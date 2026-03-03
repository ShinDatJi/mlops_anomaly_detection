Project Name
==============================

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

Prediction API
------------

The prediction service is maintained only under `apps/prediction`.
Root-level API duplicates were removed.

Quick start from repository root:

```bash
make init-prediction
make build-prediction
make run-prediction
```

API endpoints:

- `GET http://localhost:8000/status`
- `POST http://localhost:8000/predict/{category}` with multipart file field `image`

Monitoring modes:
- `minimal` (default): core operational metrics only (`anomaly_rate`, `output_drift_score`, drift, data quality issues, API traffic/performance)
- `full`: detailed data quality/error metrics and label-based metrics

For app-specific details, see `apps/prediction/README.md`.

Project Organization
------------
    
    ├── LICENSE
    ├── README.md
    ├── Makefile
    ├── apps
    │   ├── data
    │   │   ├── default.env
    │   │   ├── docker-compose.yml
    │   │   ├── Dockerfile
    │   │   ├── pyproject.toml
    │   │   └── src
    │   ├── modeling
    │   │   ├── default.env
    │   │   ├── docker-compose.yml
    │   │   ├── Dockerfile
    │   │   ├── pyproject.toml
    │   │   └── src
    │   └── prediction
    │       ├── default.env
    │       ├── docker-compose.yml
    │       ├── Dockerfile
    │       ├── pyproject.toml
    │       ├── uv.lock
    │       └── src
    ├── data
    │   ├── processed
    │   └── raw
    ├── models
    ├── references
    └── reports

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
