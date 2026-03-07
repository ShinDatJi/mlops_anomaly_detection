init:
	cp -i default.env .env

# minio
MINIO_CMD = docker compose -f apps/minio/docker-compose.yml --env-file .env --env-file apps/minio/.env --project-directory ./
init-minio:
	cp -i apps/minio/default.env apps/minio/.env
start-minio:
	$(MINIO_CMD) up -d
stop-minio:
	$(MINIO_CMD) down

# mlflow
MLFLOW_CMD = docker compose -f apps/mlflow/docker-compose.yml --env-file .env --env-file apps/mlflow/.env --project-directory ./
init-mlflow:
	cp -i apps/mlflow/default.env apps/mlflow/.env
build-mlflow:
	$(MLFLOW_CMD) build
start-mlflow:
	$(MLFLOW_CMD) up -d --wait
stop-mlflow:
	$(MLFLOW_CMD) down

# airflow
AIRFLOW_CMD = docker compose -f apps/airflow/docker-compose.yml --env-file .env --env-file apps/airflow/.env --project-directory ./
init-airflow:
	cp -i apps/airflow/default.env apps/airflow/.env
build-airflow:
	$(AIRFLOW_CMD) build
start-airflow:
	$(AIRFLOW_CMD) up -d --wait
stop-airflow:
	$(AIRFLOW_CMD) down

# data
DATA_CMD = docker compose -f apps/data/docker-compose.yml --env-file .env --env-file apps/data/.env --project-directory ./
init-data:
	cp -i apps/data/default.env apps/data/.env
build-data:
	$(DATA_CMD) build
dev-data-ingest-data:
	$(DATA_CMD) run --build ingest-data
connect-data-ingest-data:
	$(DATA_CMD) run --build ingest-data bash
run-data-ingest-data:
	$(DATA_CMD) run --rm ingest-data

# modeling
MODELING_CMD = docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./
init-modeling:
	cp -i apps/modeling/default.env apps/modeling/.env
build-modeling:
	$(MODELING_CMD) build
dev-modeling-load-initial-models:
	$(MODELING_CMD) run --build load-initial-models
dev-modeling-load-raw-data:
	$(MODELING_CMD) run --build load-raw-data
dev-modeling-load-config:
	$(MODELING_CMD) run --build load-config
dev-modeling-preprocess-data:
	$(MODELING_CMD) run --build preprocess-data
dev-modeling-train-model:
	$(MODELING_CMD) run --build train-model
dev-modeling-evaluate-model:
	$(MODELING_CMD) run --build evaluate-model
connect-modeling-load-initial-models:
	$(MODELING_CMD) run --build load-initial-models bash
connect-modeling-load-raw-data:
	$(MODELING_CMD) run --build load-raw-data bash
connect-modeling-load-config:
	$(MODELING_CMD) run --build load-config bash
connect-modeling-preprocess-data:
	$(MODELING_CMD) run --build preprocess-data bash
connect-modeling-train-model:
	$(MODELING_CMD) run --build train-model bash
connect-modeling-evaluate-model:
	$(MODELING_CMD) run --build evaluate-model bash
run-modeling-load-initial-models:
	$(MODELING_CMD) run --rm load-initial-models
run-modeling-load-raw-data:
	$(MODELING_CMD) run --rm load-raw-data
run-modeling-load-config:
	$(MODELING_CMD) run --rm load-config
run-modeling-preprocess-data:
	$(MODELING_CMD) run --rm preprocess-data
run-modeling-train-model:
	$(MODELING_CMD) run --rm train-model
run-modeling-evaluate-model:
	$(MODELING_CMD) run --rm evaluate-model
run-modeling:
	$(MAKE) run-modeling-load-raw-data
	$(MAKE) run-modeling-load-config
	$(MAKE) run-modeling-preprocess-data
	$(MAKE) run-modeling-train-model
	$(MAKE) run-modeling-evaluate-model

# prediction
PREDICTION_CMD = docker compose -f apps/prediction/docker-compose.yml --env-file .env --env-file apps/prediction/.env --project-directory ./
init-prediction:
	cp -i apps/prediction/default.env apps/prediction/.env
build-prediction:
	$(PREDICTION_CMD) build prediction
dev-prediction:
	$(PREDICTION_CMD) up --build
connect-prediction:
	$(PREDICTION_CMD) run --build prediction bash
start-prediction:
	$(PREDICTION_CMD) up -d --wait
stop-prediction:
	$(PREDICTION_CMD) down

#all
init-all:
	$(MAKE) init
	$(MAKE) init-minio
	$(MAKE) init-mlflow
	$(MAKE) init-airflow
	$(MAKE) init-data
	$(MAKE) init-modeling
	$(MAKE) init-prediction
build-all:
	$(MAKE) build-mlflow
	$(MAKE) build-airflow
	$(MAKE) build-data
	$(MAKE) build-modeling
	$(MAKE) build-prediction
start-all:
	$(MAKE) start-minio
	$(MAKE) start-mlflow
	$(MAKE) start-airflow
	$(MAKE) start-prediction
stop-all:
	$(MAKE) stop-minio
	$(MAKE) stop-mlflow
	$(MAKE) stop-airflow
	$(MAKE) stop-prediction
