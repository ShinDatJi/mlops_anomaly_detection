from numpy.version import version

from airflow.utils.dates import days_ago
from airflow.decorators import dag, task
from airflow.models import Param, Variable
from airflow.operators.python import get_current_context
from anomaly_detection.tools.config_tools import load_default_config, load_config, save_config
from anomaly_detection.tools.config_tools import get_preparation_and_preprocessing_params, update_config_with_preparation_and_preprocessing_params
from anomaly_detection.tasks.modeling import preprocess_data_task

categories = Variable.get(key="categories", default_var=[], deserialize_json=True)
default_config = load_default_config()

@task(
    task_id="update-config"
)
def update_config_task():
    params = get_current_context()["params"]
    if params["override_config_params"]:
        category = params["category"]
        version = params["version"]
        config = load_config(category, version)
        update_config_with_preparation_and_preprocessing_params(config, params)
        save_config(category, version, config)

def create_preprocess_data_dag(category, version):
    params = get_preparation_and_preprocessing_params(default_config)
    @dag(
        dag_id=f'preprocess-data_{category}_{version}',
        tags=['modeling', f"cat_{category}_{version}"],
        default_args={
            'owner': 'airflow',
            'start_date': days_ago(0, minute=1),
        },
        catchup=False,
        params={
            "category": Param(category, const=category, type="string"),
            "version": Param(version, const=version, type="string"),
            "override_config_params": Param(False, type="boolean"),
            **params
        }
    )
    def preprocess_data_dag():
        update_config_task() >> preprocess_data_task()
    
    return preprocess_data_dag()

for cat in categories:
    name = cat["name"]
    for ver in cat["versions"]:
        globals()[f"preprocess_data_{name}_{ver}"] = create_preprocess_data_dag(name, ver)