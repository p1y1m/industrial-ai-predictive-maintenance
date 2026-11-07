# Verification report

Verified locally on 2026-11-07 UTC with four CPU cores and no GPU. No model was retrained during this continuation; all metrics were read from the preserved `fd001-c9c5dec1-seed42` evidence bundle.

| Gate | Result |
| --- | --- |
| Python unit + API integration | 19 passed |
| Python branch coverage | 70.93% (training orchestration and CLI excluded; the real training run is separately evidenced) |
| Ruff | passed |
| strict mypy | passed, 23 source files |
| React component tests | 1 passed |
| TypeScript + ESLint | passed |
| Chromium E2E | 2 passed; API requests blocked to prove offline operation, including a 390 px mobile viewport with no horizontal overflow |
| Production dependency audit | 0 vulnerabilities |
| Full npm audit | 0 vulnerabilities after patched `minimatch` override |
| Live and Pages-style Vite builds | passed |
| Compose parsing | passed |
| API and frontend container builds | passed |
| Compose smoke test | PostgreSQL healthy; API healthy; MLflow `/health` 200; Nginx `/health` `ok` |
| Real-artifact API | ready; 100 assets; run `fd001-c9c5dec1-seed42`; selected model `lstm` |

Core commands:

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
docker compose build api frontend
docker compose up -d postgres api mlflow frontend
docker compose ps
```

The FastAPI/Starlette `TestClient` worker-thread path hangs in this managed execution environment because AnyIO thread dispatch does not complete. Endpoints and dependencies were made async, and API integration uses HTTPX `ASGITransport` with explicit lifespan handling. This passes deterministically and the separate Uvicorn/Compose smoke test verifies a real network server.

Known residual limitations: FD001 covers one condition/fault; nominal 90% intervals achieved 85% coverage; explanations describe the boosted-tree challenger rather than the LSTM; the UI does not provide a full per-cycle LSTM attribution trace; and the optimized JavaScript bundle is approximately 648 KB before gzip (about 195 KB gzip).

The prior `.git` directory was inspected on 2026-11-07 and contained no object database, refs, configuration, or recoverable history. A fresh `main` repository was initialized in place without deleting or changing project files. GitHub remote creation, push, and Pages deployment remain pending explicit user approval.
