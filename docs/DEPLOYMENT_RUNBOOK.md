# Deployment Runbook

This runbook documents EVBuddy deployment and operations with Docker Compose.

## 1. Preflight

On the target host:

```bash
docker --version
docker compose version
```

Clone repos in sibling layout:

- `/srv/evbuddy-serving` (this repo with `docker-compose.yml`)
- `/srv/evbuddy/backend`
- `/srv/evbuddy/frontend`

`docker-compose.yml` supports configurable build contexts via:

- `EVBUDDY_BACKEND_CONTEXT`
- `EVBUDDY_FRONTEND_CONTEXT`

## 2. Compose Environment File

```bash
cd /srv/evbuddy-serving
cp .env.compose.example .env.compose
```

Edit `.env.compose`:

- `VITE_API_BASE_URL`
- `VITE_MAPBOX_ACCESS_TOKEN`
- `EV_BUDDY_ALLOWED_ORIGINS`
- `EV_BUDDY_MAPBOX_GEOCODING_TOKEN` (if geocode proxy enabled)
- `DVC_REMOTE_URL` (if using pipeline container for `dvc pull/repro`)
- `EVBUDDY_BACKEND_CONTEXT` (for your host: `/srv/evbuddy/backend`)
- `EVBUDDY_FRONTEND_CONTEXT` (for your host: `/srv/evbuddy/frontend`)

## 3. Data and Model Materialization

If `data/` and `models/` are not already materialized on host, pull them via the optional pipeline container:

```bash
cd /srv/evbuddy-serving
docker compose --env-file .env.compose --profile ml run --rm evbuddy-pipeline bash -lc '
if [ -n "$DVC_REMOTE_URL" ]; then
  poetry run dvc remote add --force --local "$DVC_REMOTE_NAME" "$DVC_REMOTE_URL";
fi
poetry run dvc pull
'
```

## 4. Build and Start Containers

```bash
cd /srv/evbuddy-serving
docker compose --env-file .env.compose up -d --build evbuddy-backend evbuddy-frontend
docker compose --env-file .env.compose ps
```

Health checks:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:5173/ >/dev/null
```

## 5. Cutover From systemd

After container health is green, disable old host services.

Example (adjust names to your host):

```bash
sudo systemctl stop evbuddy-backend.service evbuddy-frontend.service
sudo systemctl disable evbuddy-backend.service evbuddy-frontend.service
```

If you had automatic serving-sync jobs:

```bash
sudo systemctl stop evbuddy-serving-sync.timer evbuddy-serving-sync.service
sudo systemctl disable evbuddy-serving-sync.timer
```

## 6. Post-Cutover Verification

```bash
docker compose --env-file .env.compose logs --tail=200 evbuddy-backend
docker compose --env-file .env.compose logs --tail=200 evbuddy-frontend
```

Validate API and UI through your tunnel/domain endpoints.

## 7. Rollback

If cutover fails:

```bash
cd /srv/evbuddy-serving
docker compose --env-file .env.compose down
sudo systemctl enable --now evbuddy-backend.service evbuddy-frontend.service
```

Re-enable any sync timer only if still needed:

```bash
sudo systemctl enable --now evbuddy-serving-sync.timer
```

## 8. Day-2 Operations

Update and redeploy:

```bash
cd /srv/evbuddy-serving && git pull
cd /srv/evbuddy/backend && git pull
cd /srv/evbuddy/frontend && git pull
cd /srv/evbuddy-serving
docker compose --env-file .env.compose up -d --build evbuddy-backend evbuddy-frontend
```

Run pipeline stages in container:

```bash
cd /srv/evbuddy-serving
docker compose --env-file .env.compose --profile ml run --rm evbuddy-pipeline poetry run dvc repro train_models
```

Current ops model:

- `evbuddy-serving-sync.service` + `evbuddy-serving-sync.timer` remain the single deployment orchestrator.
