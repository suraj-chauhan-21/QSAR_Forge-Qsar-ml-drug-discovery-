.PHONY: help install install-dev test test-cov lint format clean notebook fetch-data benchmark

PYTHON := python3
PIP := pip

help:
	@echo "QSAR-ML Drug Discovery — Makefile commands"
	@echo "  make install       Install runtime dependencies"
	@echo "  make install-dev   Install package in editable mode with dev dependencies"
	@echo "  make test           Run the unit test suite"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make lint           Run flake8 + isort + black --check"
	@echo "  make format         Auto-format code with black + isort"
	@echo "  make notebook       Launch Jupyter Lab on the main notebook"
	@echo "  make fetch-data     Fetch raw ChEMBL bioactivity data (AChE by default)"
	@echo "  make benchmark      Run the full 20+ model benchmark from the CLI"
	@echo "  make clean          Remove caches, build artifacts, and __pycache__"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	isort --check-only --diff src/ tests/
	black --check --diff src/ tests/

format:
	isort src/ tests/
	black src/ tests/

notebook:
	jupyter lab notebooks/QSAR_ML_Models.ipynb

fetch-data:
	$(PYTHON) -c "from src.data.fetch_chembl import fetch_bioactivity_data; \
	fetch_bioactivity_data('CHEMBL220', 'IC50', output_path='data/raw/acetylcholinesterase_raw.csv')"

benchmark:
	$(PYTHON) -m src.models.train \
		--features data/processed/morgan_fps.csv \
		--target pIC50 \
		--output results/reports/benchmark.csv

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage build dist
	@echo "Cleaned build artifacts and caches."
