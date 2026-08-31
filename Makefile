HOST ?= 0.0.0.0
PORT ?= 7799
MAX_PARALLEL ?= 2

.PHONY: help install run check

help: ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install: ## install dependencies with uv
	uv sync

run: ## run the kanban dashboard (HOST/PORT/MAX_PARALLEL overridable)
	uv run ui.py --host $(HOST) --port $(PORT) --max-parallel $(MAX_PARALLEL)

check: ## compile-check python sources
	uv run python -m py_compile ui.py pipeline.py config.py main.py
