# Prediction App

Containerized FastAPI service for anomaly prediction.

## Endpoints

- `GET /status` -> `{ "status": "ok" }`
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

## Lock file

`uv.lock` is included to keep the same project structure as other apps.
Generate or refresh it after dependency changes:

```bash
cd apps/prediction
uv lock
```

If you want strict reproducible installs in Docker, switch `uv sync` to `uv sync --locked` in `apps/prediction/Dockerfile` after generating the lockfile.
