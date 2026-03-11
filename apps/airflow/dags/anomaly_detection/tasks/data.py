from airflow.decorators import task
import os

project_root = os.environ["PROJECT_ROOT"]
data_path = os.environ["DATA_PROJECT_PATH"]

base_command = f"docker compose -f {data_path}/docker-compose.yml --env-file {data_path}/.env --project-directory ./ run --rm"

@task.bash(
    task_id="ingest-data",
    env={
        "CATEGORY": "{{ params.category }}",
        "VERSION": "{{ params.version }}",
        "DATA_RAW_PATH": "{{ params.data_raw_path }}"
    },
    cwd=project_root
)
def ingest_data_task():
    bash_command = f"{base_command} ingest-data"
    return bash_command

