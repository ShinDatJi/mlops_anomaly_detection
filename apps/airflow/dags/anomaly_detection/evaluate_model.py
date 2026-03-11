from airflow.utils.dates import days_ago
from airflow.decorators import dag, task
from airflow.models import Param, Variable
from airflow.operators.python import get_current_context
from anomaly_detection.tools.config_tools import load_default_config, load_config, save_config
from anomaly_detection.tools.config_tools import get_evaluation_params, update_config_with_evaluation_params
from anomaly_detection.tasks.modeling import evaluate_model_task

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
        update_config_with_evaluation_params(config, params)
        save_config(category, version, config)

def create_evaluate_model_dag(category, version):
    params = get_evaluation_params(default_config)
    @dag(
        dag_id=f'evaluate-model_{category}_{version}',
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
    def evaluate_model_dag():
        update_config_task() >> evaluate_model_task()
    
    return evaluate_model_dag()

for cat in categories:
    name = cat["name"]
    for ver in cat["versions"]:
        globals()[f"evaluate_model_{name}_{ver}"] = create_evaluate_model_dag(name, ver)