"""Import helpers for stage scripts whose filenames begin with numerical prefixes."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build" / "matplotlib"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_stage(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SRC / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
