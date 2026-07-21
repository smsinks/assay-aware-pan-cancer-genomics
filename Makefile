SHELL := /bin/sh
PYTHON ?= python3

export PYTHONDONTWRITEBYTECODE := 1
export MPLCONFIGDIR := $(CURDIR)/build/matplotlib
export XDG_CACHE_HOME := $(CURDIR)/build/cache
export IPYTHONDIR := $(CURDIR)/build/ipython

.DEFAULT_GOAL := audit

.PHONY: audit frozen test portability manifest notebook full-preflight full clean help

audit: frozen test portability

frozen:
	$(PYTHON) src/run_pipeline.py

test:
	$(PYTHON) -m unittest discover -s tests -v

portability:
	$(PYTHON) src/check_portability.py

manifest:
	$(PYTHON) src/build_results_manifest.py
	$(PYTHON) src/verify_results_manifest.py

notebook:
	mkdir -p build
	jupyter nbconvert --to notebook --execute \
		notebooks/pancancer_mutational_landscape.ipynb \
		--output pancancer_mutational_landscape.executed.ipynb \
		--output-dir build --ExecutePreprocessor.timeout=900

full-preflight:
	$(PYTHON) src/run_all.py --dry-run

full:
	$(PYTHON) src/run_all.py

clean:
	find build -mindepth 1 -delete 2>/dev/null || true
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete

help:
	@echo "make audit          Verify frozen outputs, run tests and scan portability"
	@echo "make notebook       Execute the frozen-output notebook into build/"
	@echo "make full-preflight Check resources required for full reconstruction"
	@echo "make full           Run the complete raw-data reconstruction"
	@echo "make manifest       Rebuild and verify frozen-output SHA-256 hashes"
