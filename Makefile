init:
	cp default.env .env

# minio
MINIO_CMD = docker compose -f apps/minio/docker-compose.yml
init-minio:
	cp apps/minio/default.env apps/minio/.env
build-minio:
	$(MINIO_CMD) build
start-minio:
	$(MINIO_CMD) up -d --wait
stop-minio:
	$(MINIO_CMD) down

# mlflow
MLFLOW_CMD = docker compose -f apps/mlflow/docker-compose.yml
init-mlflow:
	cp apps/mlflow/default.env apps/mlflow/.env
build-mlflow:
	$(MLFLOW_CMD) build
start-mlflow:
	$(MLFLOW_CMD) up -d --wait
stop-mlflow:
	$(MLFLOW_CMD) down

# data
DATA_CMD = docker compose -f apps/data/docker-compose.yml --env-file .env --env-file apps/data/.env --project-directory ./
init-data:
	cp apps/data/default.env apps/data/.env
build-data:
	$(DATA_CMD) build data
dev-data:
	$(DATA_CMD) run --build data
connect-data:
	$(DATA_CMD) run --build data bash
run-data:
	$(DATA_CMD) run --rm data

# modeling
MODELING_CMD = docker compose -f apps/modeling/docker-compose.yml --env-file .env --env-file apps/modeling/.env --project-directory ./
init-modeling:
	cp apps/modeling/default.env apps/modeling/.env
build-modeling:
	$(MODELING_CMD) build
dev-modeling-loading:
	$(MODELING_CMD) run --build modeling-loading
dev-modeling-preprocessing:
	$(MODELING_CMD) run --build modeling-preprocessing
dev-modeling-training:
	$(MODELING_CMD) run --build modeling-training
dev-modeling-evaluation:
	$(MODELING_CMD) run --build modeling-evaluation
dev-modeling-setup:
	$(MODELING_CMD) run --build modeling-setup
connect-modeling-loading:
	$(MODELING_CMD) run --build modeling-loading bash
connect-modeling-preprocessing:
	$(MODELING_CMD) run --build modeling-preprocessing bash
connect-modeling-training:
	$(MODELING_CMD) run --build modeling-training bash
connect-modeling-evaluation:
	$(MODELING_CMD) run --build modeling-evaluation bash
connect-modeling-setup:
	$(MODELING_CMD) run --build modeling-setup bash
run-modeling-loading:
	$(MODELING_CMD) run --rm modeling-loading
run-modeling-preprocessing:
	$(MODELING_CMD) run --rm modeling-preprocessing
run-modeling-training:
	$(MODELING_CMD) run --rm modeling-training
run-modeling-evaluation:
	$(MODELING_CMD) run --rm modeling-evaluation
run-modeling-setup:
	$(MODELING_CMD) run --rm modeling-setup
setup-modeling:
	run-modeling-setup
run-modeling:
	run-modeling-loading
	run-modeling-preprocessing
	run-modeling-training
	run-modeling-evaluation

# prediction
PREDICTION_CMD = docker compose -f apps/prediction/docker-compose.yml --env-file .env --env-file apps/prediction/.env --project-directory ./
init-prediction:
	cp apps/prediction/default.env apps/prediction/.env
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
	init
	init-minio
	init-mlflow
	init-data
	init-modeling
	init-prediction
build-all:
	build-minio
	build-mlflow
	build-data
	build-modeling
	build-prediction
start-all:
	start-minio
	start-mlflow
	start-prediction
setup-all:
	setup-modeling
stop-all:
	stop-minio
	stop-mlflow
	stop-prediction
