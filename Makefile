# Common developer tasks. Everything runs through uv.
.DEFAULT_GOAL := help
IMAGE_PREFIX ?= aapg

.PHONY: help setup sync lock test lint fmt type check hooks weights \
        weights-download weights-download-all build-images build-base clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the workspace venv and install all packages (editable) + dev tools
	uv sync --all-packages

sync: ## Sync the environment from uv.lock
	uv sync --all-packages

lock: ## Re-resolve and update uv.lock
	uv lock

test: ## Run the CPU-only test suite
	uv run pytest

lint: ## Lint with ruff
	uv run ruff check .

fmt: ## Auto-format with ruff
	uv run ruff format .
	uv run ruff check --fix .

type: ## Type-check first-party source with mypy
	uv run mypy packages/core/src packages/toolkit/src packages/rfd3_ellipsoid/src \
		packages/grpo_steer/src packages/mpnn_iter/src packages/validation/src \
		packages/pipelines/src packages/rfdiffusion3/src packages/proteinmpnn/src \
		packages/score_head/src

check: lint type test ## Run lint + type + test (what CI runs)

hooks: ## Install pre-commit hooks
	uv run pre-commit install

weights: ## Print the weights plan (what goes where, and how each is obtained)
	uv run python scripts/download_weights.py

weights-download: ## Download the open weights into data/weights/ (Boltz-2, RFD3; verifies ProteinMPNN)
	bash scripts/weights/download_all.sh

weights-download-all: ## Also fetch the optional open extras (Protenix v1, ESM-2)
	bash scripts/weights/download_all.sh --with-protenix-v1 --with-esm2

build-base: ## Build the shared base container image
	docker build -f containers/base.Dockerfile -t $(IMAGE_PREFIX)/base:latest .

build-images: build-base ## Build all per-tool container images
	docker build -f containers/rfd3.Dockerfile        -t $(IMAGE_PREFIX)/rfd3:latest .
	docker build -f containers/proteinmpnn.Dockerfile -t $(IMAGE_PREFIX)/proteinmpnn:latest .
	docker build -f containers/protenix.Dockerfile    -t $(IMAGE_PREFIX)/protenix:latest .
	docker build -f containers/opendde.Dockerfile     -t $(IMAGE_PREFIX)/opendde:latest .
	docker build -f containers/rosetta.Dockerfile     -t $(IMAGE_PREFIX)/rosetta:latest .

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
