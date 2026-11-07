SHELL := /bin/bash
PYTHON := .venv/bin/python
PIP := $(PYTHON) -m pip
NODE_BIN := $(CURDIR)/.tools/node/bin
NPM := PATH="$(NODE_BIN):$$PATH" npm

.PHONY: install install-frontend data train api frontend test test-python test-frontend test-e2e lint typecheck build demo-export demo-build compose-config verify

install:
	$(PIP) install -e '.[dev,mlops]'

install-frontend:
	cd frontend && $(NPM) ci

data:
	$(PYTHON) -m industrial_pm.cli data prepare --subset FD001

train:
	$(PYTHON) -m industrial_pm.cli train --config configs/fd001.yaml

api:
	$(PYTHON) -m uvicorn industrial_pm.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && $(NPM) run dev -- --host 0.0.0.0

test: test-python test-frontend

test-python:
	$(PYTHON) -m pytest --cov=industrial_pm --cov-report=term-missing --cov-report=xml

test-frontend:
	cd frontend && $(NPM) run test:coverage

test-e2e:
	cd frontend && $(NPM) run test:e2e

lint:
	$(PYTHON) -m ruff check src tests
	cd frontend && $(NPM) run lint

typecheck:
	$(PYTHON) -m mypy src
	cd frontend && $(NPM) run typecheck

build:
	cd frontend && $(NPM) run build

demo-export:
	$(PYTHON) scripts/export_static_demo.py

demo-build:
	cd frontend && VITE_STATIC_DEMO=true VITE_BASE_PATH=/industrial-ai-predictive-maintenance/ $(NPM) run build

compose-config:
	docker compose config --quiet

verify: lint typecheck test build demo-build compose-config
