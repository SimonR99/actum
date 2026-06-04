# Actum — common tasks. Run `make` or `make help` to list targets.
#
# Override the interpreter if you use a specific env, e.g.:
#   make install PYTHON=~/miniconda3/bin/python

PYTHON ?= python3
NPM    ?= npm
PIP    := $(PYTHON) -m pip

.DEFAULT_GOAL := help

# ── Help ────────────────────────────────────────────────────────────────────
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ───────────────────────────────────────────────────────────────────
.PHONY: install
install: ## Install the package with dev + camera extras (editable)
	$(PIP) install -e ".[dev,camera]"

.PHONY: install-all
install-all: ## Install with every stack (dev,camera,openai,whisper,mcp)
	$(PIP) install -e ".[dev,camera,openai,whisper,mcp]"

.PHONY: frontend
frontend: ## Install deps and build the dashboard into frontend/dist
	$(NPM) install --prefix frontend
	$(NPM) run build --prefix frontend

.PHONY: frontend-dev
frontend-dev: ## Run the Vite dev server (keep `make server` running too)
	$(NPM) run dev --prefix frontend

# ── Run ─────────────────────────────────────────────────────────────────────
.PHONY: server
server: ## Run the agent + dashboard at http://localhost:8000
	actum-server

.PHONY: run
run: server ## Alias for `server`

.PHONY: headless
headless: ## Run the headless mic/text loop (no web UI)
	actum

.PHONY: dev
dev: frontend server ## Build the dashboard, then run the server

# ── Quality ─────────────────────────────────────────────────────────────────
.PHONY: test
test: ## Run the test suite
	PYTHONPATH=src $(PYTHON) -m pytest -q

.PHONY: check
check: ## Compile-check all sources, then run tests
	PYTHONPATH=src $(PYTHON) -m compileall -q src tests
	$(MAKE) test

# ── Docker ──────────────────────────────────────────────────────────────────
.PHONY: docker-build
docker-build: ## Build the Docker image (bundles all stacks + models)
	./docker/build.sh

.PHONY: docker-server
docker-server: ## Run the dashboard in Docker
	./docker/run-server.sh

.PHONY: docker-headless
docker-headless: ## Run the headless loop in Docker
	./docker/run-headless.sh

.PHONY: docker-shell
docker-shell: ## Open a shell in the Docker image
	./docker/shell.sh

.PHONY: up
up: ## docker compose up (build if needed)
	docker compose up --build

.PHONY: down
down: ## docker compose down
	docker compose down

# ── Housekeeping ────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove Python/build caches and the frontend bundle
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache *.egg-info src/*.egg-info
	rm -rf frontend/dist
	rm -f default.profraw
