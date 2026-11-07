# Deployment and publishing

The source is public at <https://github.com/p1y1m/industrial-ai-predictive-maintenance> and the read-only offline demo is deployed at <https://p1y1m.github.io/industrial-ai-predictive-maintenance/>. The full application remains local-first; no paid cloud resources are provisioned.

## Full local stack

Copy `.env.example` to `.env`, replace `POSTGRES_PASSWORD`, and set `API_KEY` before any shared environment. Then run:

```bash
docker compose config --quiet
docker compose up --build -d
docker compose ps
```

The UI is at `http://localhost:8080`, FastAPI at `http://localhost:8000/docs`, and MLflow at `http://localhost:5000`. PostgreSQL is intentionally not published to the host. The API and Nginx images expose health checks, the API runs as a non-root user, artifacts are mounted read-only, and retraining through HTTP is dry-run only. Stop services without deleting the persisted PostgreSQL or MLflow volumes with `docker compose stop`.

For a remote host, terminate TLS at a managed reverse proxy, restrict MLflow to an authenticated private network, inject secrets through the platform secret store, set `APP_ENV=production`, use a unique database password and API key, narrow `ALLOWED_ORIGINS`, pin container digests, and add backups. The current API uses synchronous SQLAlchemy sessions in async handlers; move to an async PostgreSQL driver before high-concurrency production use.

## Static GitHub Pages demo

Static mode contains no credentials, database, model binaries, or write operations. It reads the committed `frontend/public/data/demo.json`, which is an offline export of precomputed official NASA FD001 predictions. The UI labels the data as offline and not live telemetry.

For an explicitly approved future release:

1. Create or select the GitHub repository and push the source normally.
2. In GitHub, choose **Settings → Pages → Source → GitHub Actions**.
3. Open **Actions → static-portfolio-demo → Run workflow**.
4. Verify the offline banner, navigation, queue filters, engine risk panel, chart, and model comparison at the emitted Pages URL.

The workflow is `workflow_dispatch` only. A push cannot publish the site. For a local equivalent:

```bash
cd frontend
VITE_STATIC_DEMO=true VITE_BASE_PATH=/industrial-ai-predictive-maintenance/ npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```

If the GitHub repository name changes, the workflow automatically derives its base path. For another static host, set `VITE_BASE_PATH=/` during build.

## Low-cost architecture

The lowest-cost public presentation is GitHub Pages for the read-only portfolio demo. A full non-safety-critical prototype can run on one small container host with Compose and a managed TLS proxy, but memory must be sized for the API model stack and MLflow. A serious operational deployment should separate PostgreSQL, artifact storage, inference, experiment tracking, and observability; add identity-aware access, encrypted backups, secret rotation, audit retention, rate limits, and a reviewed retraining/promotion workflow.
