# Prediction App

Containerized FastAPI service for anomaly prediction.

## Endpoints

- `GET /status` -> `{ "status": "ok" }`
- `GET /metrics` -> Prometheus metrics
- `POST /predict/{category}/{version}` with multipart file field `image` -> `{ "defective": bool }`

## Environment

Copy default env file:

```bash
cp apps/prediction/default.env apps/prediction/.env
```

Variables:

- `PREDICTION_PORT`: host port exposed for API (default `8000`)
- `API_KEY_ADMIN`: admin API key accepted via `X-API-Key` header
- `API_KEY_TEST`: test API key accepted via `X-API-Key` header
- `LOG_LEVEL`: API log level (`INFO`, `DEBUG`, ...)
- `MONITORING_REPORTS_PATH`: host path for monitoring artifacts (default `./reports/monitoring`)
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

### curl examples

```bash
curl http://localhost:8000/status
```

```bash
curl -X POST "http://localhost:8000/predict/bottle/pretrained" \
  -H "X-API-Key: ${API_KEY_TEST}" \
  -F "image=@/path/to/image.png"
```

Unauthorized example:

```bash
curl -i -X POST "http://localhost:8000/predict/bottle/pretrained" \
  -F "image=@/path/to/image.png"
```

### bash script

Interactive test helper (prompts for category and image number):

```bash
./apps/prediction/scripts/send_prediction_event.sh
```

Optional env vars:

- `API_URL` (default: `http://localhost:8000`)
- `MVTec_ROOT` (default: `./data/mvtec_anomaly_detection`)
- `MVTec_SPLIT` (default: `test/good`)
