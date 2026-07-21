"""Reject machine-specific paths and generated operating-system artefacts."""
from __future__ import annotations

import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {
    "", ".cff", ".csv", ".json", ".lock", ".md", ".py", ".r", ".toml",
    ".tsv", ".txt", ".yml", ".yaml",
}
IGNORED_PARTS = {".git", ".ipynb_checkpoints", ".venv", "build", "venv"}


def portability_failures() -> list[str]:
    failures: list[str] = []
    forbidden_names = {".DS_Store", "__MACOSX", "__pycache__"}
    local_markers = [
        "/" + "Users" + "/",
        "/" + "home" + "/",
        "file" + "://",
    ]
    windows_home = re.compile(r"(?i)(?:^|[\"'])?[a-z]:[\\/](?:users|documents)[\\/]")
    paths: list[Path] = []
    for directory, child_directories, filenames in os.walk(ROOT, topdown=True):
        child_directories[:] = [
            name for name in child_directories if name not in IGNORED_PARTS
        ]
        paths.extend(Path(directory) / name for name in filenames)
        paths.extend(Path(directory) / name for name in child_directories)
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        if path.name in forbidden_names or path.suffix.lower() == ".pyc":
            failures.append(f"generated artefact: {relative.as_posix()}")
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name == Path(__file__).name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in local_markers:
            if marker in text:
                failures.append(f"machine-specific path marker in {relative.as_posix()}: {marker}")
        if windows_home.search(text):
            failures.append(f"Windows home path in {relative.as_posix()}")
    return failures


def main() -> None:
    failures = portability_failures()
    if failures:
        raise RuntimeError("Portability audit failed:\n  - " + "\n  - ".join(failures))
    print("Portability audit passed: no local paths or generated OS/Python artefacts")


if __name__ == "__main__":
    main()
