# Contributing

Contributions are welcome for reproducibility, leakage prevention, scientific evaluation, accessibility, and operational safety. Open an issue before a large architectural change.

Do not commit NASA raw data, generated model binaries, credentials, `.env`, MLflow databases, or PostgreSQL volumes. Synthetic fixtures must be explicitly labeled and must never be reported as NASA benchmark evidence. New model claims require a machine-readable experiment artifact, engine-grouped splits, calibration independent of official test labels, a model-card update, and exact reproduction commands.

Before a pull request, run:

```bash
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m mypy src
.venv/bin/python -m pytest --cov=industrial_pm
npm --prefix frontend run lint
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
docker compose config --quiet
```

Use deterministic seeds and keep changes scoped. Do not silently weaken validation gates or replace verified metrics with hand-edited values.
