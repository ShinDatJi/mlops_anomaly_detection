from airflow.utils.dates import days_ago
from airflow.decorators import dag, task
from airflow.models import Param, Variable
from airflow.operators.python import get_current_context
from anomaly_detection.tools.config_tools import load_default_config, load_config, save_config
from anomaly_detection.tools.config_tools import get_params_from_config, get_config_from_params
from anomaly_detection.tasks.modeling import train_model_task

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
            "modeling": get_config_from_params(params["modeling"]),
            "training": get_config_from_params(params["training"])
        })
        save_config(category, config)

def create_train_model_dag(category):
    modeling_params = get_params_from_config(default_config["modeling"])
    training_params = get_params_from_config(default_config["training"])
    @dag(
        dag_id=f'train-model_{category}',
        tags=['modeling', f"cat_{category}"],
        default_args={
            'owner': 'airflow',
            'start_date': days_ago(0, minute=1),
        },
        catchup=False,
        params={
            "category": Param(category, const=category, type="string"),
            "override_config_params": Param(False, type="boolean"),
            "modeling": Param(modeling_params, type="object", section="Modeling"),
            "training": Param(training_params, type="object", section="Training"),
        }
    )
    def train_model_dag():
        update_config_task()
        train_model_task()
    
    return train_model_dag()

for cat in categories:
    globals()[f"train_model_{cat}"] = create_train_model_dag(cat)