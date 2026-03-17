# MLOps MVTec Anomaly Detection

Datascientest MLOps project using deep learning with the MVTec Anomaly Detection dataset as exemplary database.

MVTec dataset: <https://www.mvtec.com/company/research/datasets/mvtec-ad>

Project members: Isaak Gerber, Fernando Sotres

The project was developed with VSCode with Copilot enabled, which was used for code suggestions and repetitive tasks, but all code was written and adapted by the project members.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Development setup](#development-setup)
- [Project Organization](#project-organization)
- [Apps](#apps)
  - [MinIO](#minio)
  - [MLFlow](#mlflow)
  - [Airflow](#airflow)
  - [Data](#data)
  - [Modeling](#modeling)
  - [Prediction](#prediction)
  - [Monitoring](#monitoring)
- [Demo](#demo)
- [Tests](#tests)
- [Hardware requirements](#hardware-requirements)
- [Improvements and next steps](#improvements-and-next-steps)

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
* Start e2e tests: `make start-tests`
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
  * Predict: `POST http://localhost:8000/predict/{category}/{version}`
    * with multipart file field `image`
    * with header `x-api-key`

## Project Organization

### Apps structure (applies to all folders in `./apps`)

```text
<app_name>/
├── src/                  # (python app) Source code for the app
│   └── <app_name>/
│       └── __init__.py
├── .env                  # (not committed) Production environment
├── default.env           # Default production environment
├── docker-compose.yml    # Docker compose stack for the app
├── Dockerfile            # Dockerfile for the app if necessary
├── pyproject.toml        # (python app) Production dependencies and scripts
└── README.md
```

### Root structure

```text
├── apps/                       # Sub projects
│   ├── airflow/                # Orchestration
│   ├── data/                   # Data ingestion
│   ├── minio/                  # Training data storage
│   ├── mlflow/                 # Experiment tracking, model registry
│   ├── modeling/               # Data loading, Preprocessing, Training, Evaluation
│   ├── monitoring/             
│   │   ├── evidently/          # Drift detection
│   │   ├── grafana/            # Monitoring
│   │   ├── prometheus/         # Metrics collection
│   └── prediction/             # Prediction API
├── data/                       
│   ├── processed/              # (generated) Preprocessed data for model training
│   └── raw/                    # (local only) Raw training data
├── demo/                       # Streamlit demo
├── models/                     # Pre-trained models
├── references/                 # Evidently reference
├── reports/                    
│   ├── modeling/               # (generated) Modeling reports and configuration for each project (category/version)
│   └── monitoring/             # (generated) Monitoring reports
├── tests/                      # e2e tests
├── .env                        # (not committed) Environment for development setup         
├── LICENSE
├── Makefile                    # Commands for running the project
├── README.md
├── default.env                 # Default environment for development setup
└── pyproject.toml              # Dependencies for development
```

## Apps

Apps are organized under `./apps` folder. Each app is built with it's own docker compose stack and environment.

### MinIO

MinIO is used for storing the raw data for the training process.

Command: `make start-minio`
Port: 9011
Username: minio
Password: minio123

* Storing the raw data for each project (category/version).
* Each category has it's own bucket.
* Each version has it's own sub folder.

### MLFlow

MLFlow is used for experiment tracking and model registry.

Command: `make start-mlflow`
Port: 5000
Username: mlflow
Password: mlflowpassword

* Experiment tracking
  * loading-{CATEGORY}_{VERSION} => params, metrics and statistical plots from loading raw data
  * preprocessing-{CATEGORY}_{VERSION} => params and metrics from preprocessing (patching) of images
  * training-{CATEGORY}_{VERSION} => params, metrics and plots from model training
  * evaluation-{CATEGORY}_{VERSION} => params, metrics, plots and model from model evaluation
  * initial-{CATEGORY} => params, metrics and model from initial pre-trained models provided for each category
* Model registry
  * model name: {CATEGORY}_{VERSION}
  * model version: automatically incremented for each new model of a project
  * model alias: "champion" for the best model of a project => used for prediction

### Airflow

Airflow is used for orchestration of the data and modeling pipelines.

Command: `make start-airflow`
Port: 8080
Username: airflow
Password: airflow

* For each new project (category/version) new data and modeling DAGs are auto generated by administration DAGs.
* After running administration DAGs refresh the page or reload a DAG or wait a bit to see the new DAGs.
* data and modeling DAGs are highly configurable.

#### Airflow DAGs

* administration DAGs
  * init-airflow => creates data and modeling dags for an exemplary project (category/version)
  * add-category => creates data and modeling dags for a new project (category/version)
* data DAGs
  * ingest-data_{CATEGORY}_{VERSION} => ingests new data
* modeling DAGs
  * load-config_{CATEGORY}_{VERSION} => loads the modeling configuration
  * load-raw-data_{CATEGORY}_{VERSION} => loads the raw data
  * preprocess-data_{CATEGORY}_{VERSION} => preprocesses the data
  * train-model_{CATEGORY}_{VERSION} => trains the model
  * evaluate-model_{CATEGORY}_{VERSION} => evaluates the model
* composite DAGs
  * end-to-end_{CATEGORY}_{VERSION} => runs the whole data and modeling pipeline in the right order

### Data

Data is used to validate the raw data, create a database and upload the data to MinIO.

Command: `make run-data-ingest-data`
Environment: `CATEGORY` and `VERSION` to define the project. By default raw data is loaded from ./data/raw/{CATEGORY}

* Validates the folder and file structure of raw data path.
* Creates image database with statistical information.
* Checks for image consistency.
* Uploads images to MinIO server.
* Uploads database file to MinIO server.

### Modeling

Modeling is used for loading raw data, preprocessing, training and evaluating the model.

Environment: `CATEGORY` and `VERSION` to define the project.

* All scripts in modeling are connected to MLFLow for experiment tracking.
* All scripts are also configurable and callable via Airflow.
* Each modeling step is configurable (`./reports/modeling/{CATEGORY_VERSION}/config.json`) and should be evaluated using metrics and plots stored in MLFlow and reports.
* Each script can be run independently of the project (category/version).

The general flow should be:

* load the initial config (e.g. use the provided default config)
* load the raw data => inspecting the plots to decide on preprocessing parameters
* preprocess the data => inspect the metrics produced by patching on anomaly coverage / patch count => reiterate
* train the model => inspect the plots for over-fitting or under-fitting => reiterate
* evaluate the model => inspect the plots and metrics to define the prediction threshold

#### Modeling Scripts

* Load initial models: `make run-modeling-load-initial-models`
  * Loads initial pre-trained models for each category provided in `./models/` to MLFlow Model Registry
* Load configuration: `make run-modeling-load-config`
  * Loads configuration from a default configuration file or from the model registry using the config of an aliased model
* Load raw data: `make run-modeling-load-raw-data`
  * Loads raw data from MinIO server
  * Creates statistical plots of images
* Preprocess data: `make run-modeling-preprocess-data`
  * Creates a training and test database
  * Creates patches
* Train model: `make run-modeling-train-model`
  * Trains a configurable CNN model with the training database
* Evaluate model: `make run-modeling-evaluate-model`
  * Evaluates the trained model with the test database
  * Registers the model in MLFlow Model Registry

### Prediction

See also: [Prediction documentation](./apps/prediction/README.md)

Prediction is a FastAPI service providing a prediction endpoint for customers.

Command: `make start-prediction`
Port: 8000
API-Key: projectAdmin or projectTest

* Status endpoint
  * URI: `http://localhost:8000/status`
  * Method: GET
  * Response:

    ```json
    { "status": "ok" }
    ```

* Prediction endpoint
  * URI: `http://localhost:8000/predict/{category}/{version}`
  * Method: POST
  * Header parameters: `x-api-key`
  * Multipart file parameters: `image`
  * Response: "params" and "pred_probas" gives the customer the possibility to generate insight on where in the image an anomaly is detected and with which probability. (See: demo)

    ```json
    { 
        "defective": bool,
        "params": {
            "patches": int,
            "overlap": float,
            "height_cropping": int,
            "width_cropping": int,
            "threshold": float
        },
        "pred_probas": [float]
    }
    ```

  * Exemplary request with python requests library:

    ```python
    import requests

    url = "http://localhost:8000/predict/bottle/pretrained"
    headers = {"x-api-key": "projectTest"}
    files = {"image": open("path_to_image.jpg", "rb")}
    response = requests.post(url, headers=headers, files=files)
    print(response.json())
    ```

### Monitoring

See also: [Monitoring documentation](./apps/monitoring/README.md)

This creates a clickable link to the monitoring README file while maintaining consistency with the rest of your documentation.


Monitoring is used for drift detection and monitoring of the prediction endpoint.

Command: `make start-monitoring`
Port: 3001
Username: admin
Password: admin

* Drift detection: Evidently
* Metrics collection: Prometheus
* Metrics visualization: Grafana

## Demo

A Streamlit app containing the Project presentation and a prediction demo with visualization of the predicted anomaly areas.

Command: `make start-demo`
Port: 8501

* The demo is based on the prediction endpoint, so make sure to have the prediction app and MLFlow running.
* Also the initial pre-trained models should be loaded to MLFlow Model Registry with `make run-modeling-load-initial-models` command.
* Also the MVTec dataset should be downloaded to `./data/raw`.
* The demo allows to select a category and send a random defective or good image to the prediction endpoint. The predicted anomaly areas are visualized with a heatmap.

## Tests

For the prediction endpoint some e2e tests are implemented.

Command: `make start-tests`

* The tests are based on the prediction endpoint, so make sure to have the prediction app and MLFlow running.
* Also the initial pre-trained models should be loaded to MLFlow Model Registry with `make run-modeling-load-initial-models` command.

## Hardware requirements

50GB of free disk space, 8GB of free RAM and a SSD are the recommended requirements for running the whole project.

* The project uses tensorflow and Airflow both of which comes with heavy disk space and memory requirements, also the MVTec dataset consumes 5GB alone.
* The project stores and loads thousands of images for the patches during preprocessing and model training, so a SSD is mandatory for the project to run in a reasonable time.
* The project is based on deep learning but uses quiet small models and small patched images as input, so it can be run without GPU, but for faster training a GPU is recommended.
* Building and starting the docker compose stacks for the first time takes a while, so be patient.

## Improvements and next steps

* User management: Implementing a more sophisticated user management with authentication and authorization for the prediction endpoint and the provided tools facilitating role based access control for customers and internal users.
* Data API: Implementing a data API for uploading training data by customers.
* Tests: Implementing unit, integration, and e2e tests for the whole project.
* CI/CD: Implementing CI/CD pipelines for testing and deployment of the project.
* Scaling: Implementing nginx for load balancing and scaling of the prediction endpoint.
* Efficient model serving: Implementing model packaging with e.g. bentoml for efficient model serving.
* Customer side model: Providing the customer with a model package to run the model on their side for faster predictions.