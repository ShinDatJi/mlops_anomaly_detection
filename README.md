# MLOps MVTec Anomaly Detection

Datascientest MLOps project using deep learning with the MVTec Anomaly Detection dataset as exemplary database.

MVTec dataset: <https://www.mvtec.com/company/research/datasets/mvtec-ad>

## Project Overview

The goal is to simulate a real world application for industrial anomaly detection. A customer is providing us with a training dataset containing labeled images (defective or not) . With that we train a model by carefully selecting preprocessing and modeling parameters to get a deep learning anomaly detection model. This model will be provided to the customer through a prediction api endpoint.
The project is designed to accept multiple projects from multiple customers. Internally a customer is a category and a project of a customer is a training data version for a specific category. So, the training pipeline, model configuration and provided model with the endpoint is specific to a category and data version.
As specific to anomaly detection the final model of a customer's project is fixed and not automatically adapted to new data. If the data on the customer side changes, the customer has to provide new training data for a new project.

![project overview](/demo/figures/introduction.png)

## Architecture

The project is designed to be deployed on multiple servers for load balancing and separation of concern. Each sub project has its own docker compose stack organized into separate folders: ```apps/``` with it's own environment configuration. For easy development setup a default.env is provided in the root directory assuming all docker compose projects are running on the same machine.

![global architecture](/demo/figures/architecture.png)

* Training data: public MVTec AD dataset (industrial anomaly detection)
* Data storage: MinIO
* Modeling: Configurable CNN with patching
* Orchestration: Airflow
* Experiment Tracking: MLFlow
* Model Registry: MLFlow
* Prediction: FastAPI service
* Drift detection and monitoring: Evidently, Prometheus and Grafana

## Development setup

### Prerequisites

* The project is solely based on the uv package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
* Also it requires docker with an installed docker compose plugin.
* The example dataset should be downloaded to `./data/raw`: <https://www.mvtec.com/company/research/datasets/mvtec-ad>

### Starting the project

To run the project go through the following steps:

* Init the environment: `make init-all`
* Adapt the development environment if necessary: `./.env`
* Build the docker images: `make build-all`
* Start the whole stack: `make start-all`
* Load initial pre-trained models: `make run-modeling-load-initial-models`
* Start a prediction demo app: `make start-demo`
* Stop the whole stack: `make stop-all`

Each sub project has also its own make commands. E.g.: `make start-mlflow`
The data and modeling projects also have their own run commands: `make run-modeling-preprocessing`

### Web-based interfaces

* MinIO: Port: 9011, Username: minio, Password: minio123
* MLFLow: Port: 5000, Username: mlflow, Password: mlflowpassword
* Airflow: Port: 8080, Username: airflow, Password: airflow
* Prometheus: Port: 9090
* Grafana: Port: 3001, Username: admin, Password: admin
* Streamlit: Port: 8501
* FastAPI: Port: 8000, API-Key: projectAdmin or projectTest
  * Status: `GET http://localhost:8000/status`
  * Predict: `POST http://localhost:8000/predict/{category}/{version}` with multipart file field `image`

## Project Organization

```text
├── .env                        < Environment for development setup         
├── LICENSE
├── Makefile                    < Commands for running the project
├── README.md
├── default.env                 < Default environment for development setup
├── pyproject.toml              < Dependencies for development
├── uv.lock
├── apps                        < Subprojects
│   ├── airflow                 < Orchestration
│   ├── data                    < Data ingestion
│   │   ├── .env                < Production environment
│   │   ├── default.env         < Default production environment
│   │   ├── docker-compose.yml
│   │   ├── Dockerfile
│   │   ├── pyproject.toml      < Production dependencies
│   │   ├── README.md
│   │   ├── src/                < Python source code
│   │   ├── tests/              < Unit tests
│   │   └── uv.lock
│   ├── minio                   < Training data storage
│   ├── mlflow                  < Experiment tracking, model registry
│   ├── modeling                < Data loading, Preprocessing, Training, Evaluation
│   ├── monitoring
│   │   ├── evidently/          < Drift detection
│   │   ├── grafana/            < Monitoring
│   │   ├── prometheus/         < Metrics collection
│   └── prediction              < Prediction API
├── data
│   ├── processed               < Preprocessed data for model training
│   └── raw                     < Raw training data
├── demo                        < Streamlit demo
├── models                      < Pre-trained models with configuration and metrics report
├── references                  < Evidently reference
└── reports                     < Modeling and Monitoring reports 
```
