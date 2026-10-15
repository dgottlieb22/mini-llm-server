PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: venv-cpu test bench-smoke clean

venv-cpu:
	python3 -m venv .venv
	$(PIP) install --quiet -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
	$(PIP) install --quiet -e .

test:
	$(PY) -m pytest

bench-smoke:
	$(PY) -m miniserve.bench.run --engine naive --workload smoke --out results/raw

clean:
	rm -rf .pytest_cache miniserve/__pycache__ miniserve/*/__pycache__ tests/__pycache__
