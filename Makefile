init:
	cp default.env .env

# mlflow
MLFLOW_CMD = docker compose -f apps/mlflow/docker-compose.yml
init-mlflow:
	cp apps/mlflow/default.env apps/mlflow/.env
build-mlflow:
	$(MLFLOW_CMD) build
start-mlflow:
	$(MLFLOW_CMD) up -d
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
dev-modeling-preprocessing:
	$(MODELING_CMD) run --build modeling-preprocessing
dev-modeling-training:
	$(MODELING_CMD) run --build modeling-training
dev-modeling-evaluation:
	$(MODELING_CMD) run --build modeling-evaluation
dev-modeling-setup:
	$(MODELING_CMD) run --build modeling-setup
connect-modeling-preprocessing:
	$(MODELING_CMD) run --build modeling-preprocessing bash
connect-modeling-training:
	$(MODELING_CMD) run --build modeling-training bash
connect-modeling-evaluation:
	$(MODELING_CMD) run --build modeling-evaluation bash
connect-modeling-setup:
	$(MODELING_CMD) run --build modeling-setup bash
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
	run-modeling-preprocessing
	run-modeling-training
	run-modeling-evaluation

# prediction
PREDICTION_CMD = docker compose -f apps/prediction/docker-compose.yml --env-file .env --env-file apps/prediction/default.env --env-file apps/prediction/.env --project-directory ./
init-prediction:
	cp apps/prediction/default.env apps/prediction/.env
build-prediction:
	$(PREDICTION_CMD) build prediction
dev-prediction:
	$(PREDICTION_CMD) up --build
connect-prediction:
	$(PREDICTION_CMD) run --build prediction bash
start-prediction:
	$(PREDICTION_CMD) up -d
stop-prediction:
	$(PREDICTION_CMD) down

# monitoring
MONITORING_CMD = docker compose -f apps/monitoring/docker-compose.yml --env-file .env --env-file apps/monitoring/.env --project-directory ./
init-monitoring:
	cp apps/monitoring/default.env apps/monitoring/.env
build-monitoring:
	$(MONITORING_CMD) build
start-monitoring:
	$(MONITORING_CMD) up -d --build
stop-monitoring:
	$(MONITORING_CMD) down

#all
init-all:
	init
	init-mlflow
	init-data
	init-modeling
	init-prediction
	init-monitoring
build-all:
	build-mlflow
	build-data
	build-modeling
	build-prediction
start-all:
	start-mlflow
	start-prediction
	start-monitoring
setup-all:
	setup-modeling
stop-all:
	stop-mlflow
	stop-prediction
	stop-monitoring
