from airflow.utils.dates import days_ago
from airflow.decorators import dag, task
from airflow.models import Param, Variable
from airflow.operators.python import get_current_context
from anomaly_detection.tools.config_tools import load_default_config, load_config, save_config
from anomaly_detection.tools.config_tools import get_params_from_config, get_config_from_params
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
        config = load_config(category)
        config.update({
            "preparation": get_config_from_params(params["preparation"]),
            "preprocessing": get_config_from_params(params["preprocessing"])
        })
        save_config(category, config)

def create_preprocess_data_dag(category):
    preparation_params = get_params_from_config(default_config["preparation"])
    preprocessing_params = get_params_from_config(default_config["preprocessing"])
    @dag(
        dag_id=f'preprocess-data_{category}',
        tags=['modeling', f"cat_{category}"],
        default_args={
            'owner': 'airflow',
            'start_date': days_ago(0, minute=1),
        },
        catchup=False,
        params={
            "category": Param(category, const=category, type="string"),
            "override_config_params": Param(False, type="boolean"),
            "preparation": Param(preparation_params, type="object", section="Preparation"),
            "preprocessing": Param(preprocessing_params, type="object", section="Preprocessing"),
        }
    )
    def preprocess_data_dag():
        update_config_task()
        preprocess_data_task()
    
    return preprocess_data_dag()

for cat in categories:
    globals()[f"preprocess_data_{cat}"] = create_preprocess_data_dag(cat)