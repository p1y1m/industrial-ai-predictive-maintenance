# Persistent Progress Log

This file is the authoritative checkpoint for resuming work after interruption.

## Current checkpoint

- Date: 2026-11-07 UTC
- Phase: public portfolio release
- Status: implementation is public and the no-cost static demo is deployed for the bounded FD001 reference scope; documented extensions remain
- Next action: maintain the verified release; retrain or republish only after deliberate model or product changes

## Preserved scientific evidence

- Official NASA C-MAPSS FD001 archive and processed train/test tables remain intact.
- Archive SHA-256: `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`.
- Run: `fd001-c9c5dec1-seed42`; seed 42; CPU-only; 70 development, 15 validation, 15 calibration, and 100 untouched official-test engines.
- Selected LSTM: validation RMSE 15.432; official test capped RMSE 15.139; uncapped RMSE 16.021; capped MAE 10.755.
- Thirty-cycle risk: Brier 0.0537, AUROC 0.9763, ECE 0.0326.
- Nominal 90% conformal interval: 85% empirical test coverage, 34.68-cycle mean width.
- No retraining occurred during the 2026-11-07 continuation.

## Continuation work completed

- Verified dependency pins: FastAPI 0.115.6, Starlette 0.41.3, HTTPX 0.27.2, AnyIO 3.7.1.
- Diagnosed the managed-environment AnyIO thread-worker hang and converted endpoints/dependencies to async; replaced `TestClient` with HTTPX ASGI transport and explicit lifespan handling.
- Made API tests hermetic so GitHub CI does not require ignored model/data artifacts.
- Passed 19 Python tests at 70.93% coverage, Ruff, and strict mypy across 23 source files.
- Passed ESLint, TypeScript, one React component test, live/static Vite builds, and two real Chromium E2E tests covering desktop interaction and a 390 px mobile viewport.
- Added deterministic static export from the real artifact repository; committed payload contains all 100 official-test predictions and explicit offline/not-live metadata.
- Added interactive navigation, queue filters, critical-queue action, per-engine interval/anomaly/risk analysis, and offline evidence banner.
- Captured `reports/screenshots/fleet-command-center.png` from the browser test.
- Added a manual-only GitHub Pages workflow and publishing guidance; nothing was pushed or deployed.
- Added `.dockerignore`, reducing build context from more than 217 MB (interrupted) to about 108 KB.
- Built backend and frontend images and smoke-tested PostgreSQL, FastAPI, MLflow, and Nginx. Fixed an absolute champion-artifact path that failed in containers.
- Verified the Compose API against real artifacts: ready, 100 assets, LSTM selected, run ID correct.
- `npm audit --omit=dev` and full `npm audit` report zero vulnerabilities after a patched `minimatch` override.
- Confirmed the prior `.git` directory contained no objects, refs, configuration, or recoverable history; initialized a fresh `main` repository in place without altering project files.
- Audited `.gitignore`: credentials, local environments, caches, raw/processed datasets, generated reports, and model binaries are excluded; the 100-engine offline evidence payload and real UI screenshot are retained.
- Re-ran the complete local publication gate on 2026-11-07: all tests, static build, dependency audit, and Compose configuration passed without retraining.
- Published the approved public repository at `https://github.com/p1y1m/industrial-ai-predictive-maintenance` and deployed the manual-only Pages workflow at `https://p1y1m.github.io/industrial-ai-predictive-maintenance/`.

## Exact verification snapshot

| Gate | Result |
| --- | --- |
| Python tests | 19 passed |
| Covered application code | 70.93% |
| Python lint/type | Ruff passed; mypy passed |
| Frontend component | 1 passed |
| Browser E2E | 2 passed in Chromium (desktop interaction and mobile layout) |
| Production/static builds | passed |
| Docker images | API and frontend passed |
| Compose runtime | PostgreSQL/API healthy; MLflow 200; Nginx `ok` |
| Dependency audit | 0 vulnerabilities |

## Known constraints

- No recoverable Git history existed in the empty prior `.git` directory; publication therefore starts with a clean initial commit.
- GitHub CI initially exposed Playwright 1.56's lack of Ubuntu 26.04 support; the frontend job is pinned to the supported Ubuntu 24.04 runner.
- FD001 is simulated and contains only one operating condition and fault mode.
- Conformal coverage is five percentage points below its nominal target.
- Permutation explanations apply to the boosted-tree challenger, not the LSTM champion.
- The individual-engine UI shows interval, anomaly status, and risk horizons; full measured per-cycle telemetry and LSTM attribution timelines remain extensions.
- The JavaScript bundle is about 648 KB minified and 195 KB gzip; route/code splitting is a future performance optimization.
- Production concurrency should use an async SQLAlchemy/PostgreSQL driver rather than synchronous sessions inside async endpoints.

## Resume/reproduce commands

```bash
.venv/bin/python -m pytest --cov=industrial_pm --cov-report=term-missing
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m mypy src
npm --prefix frontend run lint
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run test:e2e
VITE_STATIC_DEMO=true VITE_BASE_PATH=/industrial-ai-predictive-maintenance/ npm --prefix frontend run build
docker compose config --quiet
docker compose up --build -d
```

Do not retrain merely to resume. Re-export `frontend/public/data/demo.json` only after a deliberately completed and verified new experiment. Do not push, publish Pages, or provision infrastructure without explicit approval.
