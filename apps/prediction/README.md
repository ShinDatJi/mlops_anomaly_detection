# Prediction App

Containerized FastAPI service for anomaly prediction.

## Endpoints

- `GET /status` -> `{ "status": "ok" }`
- `GET /metrics` -> Prometheus metrics
- `POST /predict/{category}` with multipart file field `image` -> `{ "defective": bool }`

## Environment

Copy default env file:

```bash
cp apps/prediction/default.env apps/prediction/.env
```

Variables:

- `PREDICTION_PORT`: host port exposed for API (default `8000`)
- `MODELS_PATH`: host path to model directory (default `./models`)
- `VIRTUAL_MODELS_PATH`: container path for mounted models (default `./models`)
- `LOG_LEVEL`: API log level (`INFO`, `DEBUG`, ...)
- `MONITORING_REPORTS_PATH`: host path for monitoring artifacts (default `./reports/monitoring`)
- `VIRTUAL_MONITORING_REPORTS_PATH`: container mount path for monitoring artifacts
- `MONITORING_EVENTS_FILE`: file path relative to mounted monitoring path for jsonl events
- `MLFLOW_MODEL_NAME`: deployed model name label for metrics/events
- `MLFLOW_MODEL_VERSION`: deployed model version label for metrics/events
- `MLFLOW_RUN_ID`: optional run identifier label for metrics/events

## Build and Run

From repository root:

```bash
docker compose \
  -f apps/prediction/docker-compose.yml \
  --env-file .env \
  --env-file apps/prediction/.env \
  --project-directory ./ \
  up --build
```

## Test

```bash
curl http://localhost:8000/status
```

```bash
curl -X POST "http://localhost:8000/predict/bottle" \
  -F "image=@/path/to/image.png"
```

Interactive test helper (prompts for category and image number):

```bash
./apps/prediction/scripts/send_prediction_event.sh
```

Optional env vars:

- `API_URL` (default: `http://localhost:8000`)
- `MVTec_ROOT` (default: `./data/mvtec_anomaly_detection`)
- `MVTec_SPLIT` (default: `test/good`)

## Lock file

`uv.lock` is included to keep the same project structure as other apps.
Generate or refresh it after dependency changes:

```bash
cd apps/prediction
uv lock
```

If you want strict reproducible installs in Docker, switch `uv sync` to `uv sync --locked` in `apps/prediction/Dockerfile` after generating the lockfile.
