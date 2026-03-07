from airflow.utils.dates import days_ago
from airflow.decorators import dag, task, task_group
from airflow.models import Param, Variable
from airflow.operators.python import get_current_context
import os
from anomaly_detection.tools.config_tools import load_default_config, load_config, save_config
from anomaly_detection.tools.config_tools import get_all_params, update_config_with_all_params
from anomaly_detection.tasks.data import ingest_data_task
from anomaly_detection.tasks.modeling import load_raw_data_task, load_config_task
from anomaly_detection.tasks.modeling import preprocess_data_task, train_model_task, evaluate_model_task

modeling_project_path = os.environ["MODELING_PROJECT_PATH"]
default_config_file = os.path.join(modeling_project_path, "default.config.json")

categories = Variable.get(key="categories", default_var=[], deserialize_json=True)
default_config = load_default_config()

@task_group(
    group_id="data"
)
def data_tasks():
    ingest_data_task() >> load_raw_data_task()

@task_group(
    group_id="config"
)
def config_tasks():
    load_config_task() >> update_config_task()

@task_group(
    group_id="modeling"
)
def modeling_tasks():
    preprocess_data_task() >> train_model_task() >> evaluate_model_task()

@task(
    task_id="update-config"
)
def update_config_task():
    params = get_current_context()["params"]
    if params["override_config_params"]:
        category = params["category"]
        version = params["version"]
        config = load_config(category, version)
        update_config_with_all_params(config, params)
        save_config(category, version, config)

def create_end_to_end_dag(category, version):
    params = get_all_params(default_config)
    @dag(
        dag_id=f'end-to-end_{category}_{version}',
        tags=['composite', f"cat_{category}_{version}"],
        default_args={
            'owner': 'airflow',
            'start_date': days_ago(0, minute=1),
        },
        catchup=False,
        params={
            "category": Param(category, const=category, type="string"),
            "version": Param(version, const=version, type="string"),
            "data_raw_path": Param(f"./data/raw/{category}", type="string"),
            "config_file": Param(default_config_file, type="string"),
            "config_from_model_registry": Param("1", type="string", enum=["0", "1"]),
            "config_from_model_registry_category": Param(category, type="string"),
            "config_from_model_registry_version": Param("pretrained", type="string"),
            "config_from_model_registry_alias": Param("champion", type="string"),
            "override_config_params": Param(False, type="boolean"),
            **params
        }
    )
    def end_to_end_dag():
        [data_tasks(),  config_tasks()] >> modeling_tasks()
    
    return end_to_end_dag()

for cat in categories:
    name = cat["name"]
    for ver in cat["versions"]:
        globals()[f"end_to_end_{name}_{ver}"] = create_end_to_end_dag(name, ver)