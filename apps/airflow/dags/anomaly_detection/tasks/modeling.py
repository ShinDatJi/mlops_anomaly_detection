from airflow.decorators import task
import os

project_root = os.environ["PROJECT_ROOT"]
modeling_path = os.environ["MODELING_PROJECT_PATH"]

base_command = f"docker compose -f {modeling_path}/docker-compose.yml --env-file {modeling_path}/.env --project-directory ./ run --rm"

@task.bash(
    task_id="load-raw-data",
    env={
        "CATEGORY": "{{ params.category }}",
        "VERSION": "{{ params.version }}"
    },
    cwd=project_root
)
def load_raw_data_task():
    bash_command = f"{base_command} load-raw-data"
    return bash_command

@task.bash(
    task_id="load-config",
    env={
        "CATEGORY": "{{ params.category }}",
        "VERSION": "{{ params.version }}",
        "CONFIG_FILE": "{{ params.config_file }}",
        "CONFIG_FROM_MODEL_REGISTRY": "{{ params.config_from_model_registry }}",
        "CONFIG_FROM_MODEL_REGISTRY_CATEGORY": "{{ params.config_from_model_registry_category }}",
        "CONFIG_FROM_MODEL_REGISTRY_VERSION": "{{ params.config_from_model_registry_version }}",
        "CONFIG_FROM_MODEL_REGISTRY_ALIAS": "{{ params.config_from_model_registry_alias }}",
    },
    cwd=project_root
)
def load_config_task():
    bash_command = f"{base_command} load-config"
    return bash_command

@task.bash(
    task_id="preprocess-data",
    env={
        "CATEGORY": "{{ params.category }}",
        "VERSION": "{{ params.version }}"
    },
    cwd=project_root
)
def preprocess_data_task():
    bash_command = f"{base_command} preprocess-data"
    return bash_command

@task.bash(
    task_id="train-model",
    env={
        "CATEGORY": "{{ params.category }}",
        "VERSION": "{{ params.version }}"
    },
    cwd=project_root
)
def train_model_task():
    bash_command = f"{base_command} train-model"
    return bash_command

@task.bash(
    task_id="evaluate-model",
    env={
        "CATEGORY": "{{ params.category }}",
        "VERSION": "{{ params.version }}"
    },
    cwd=project_root
)
def evaluate_model_task():
    bash_command = f"{base_command} evaluate-model"
    return bash_command

