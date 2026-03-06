from airflow.utils.dates import days_ago
from airflow.decorators import dag, task
from airflow.models import Param, Variable
from airflow.operators.python import get_current_context
from anomaly_detection.tools.config_tools import load_default_config, load_config, save_config
from anomaly_detection.tools.config_tools import get_params_from_config, get_config_from_params
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
        config = load_config(category)
        config.update({
            "evaluation": get_config_from_params(params["evaluation"])
        })
        save_config(category, config)

def create_evaluate_model_dag(category):
    evaluation_params = get_params_from_config(default_config["evaluation"])
    @dag(
        dag_id=f'evaluate-model_{category}',
        tags=['modeling', f"cat_{category}"],
        default_args={
            'owner': 'airflow',
            'start_date': days_ago(0, minute=1),
        },
        catchup=False,
        params={
            "category": Param(category, const=category, type="string"),
            "override_config_params": Param(False, type="boolean"),
            "evaluation": Param(evaluation_params, type="object", section="Evaluation"),
        }
    )
    def evaluate_model_dag():
        update_config_task()
        evaluate_model_task()
    
    return evaluate_model_dag()

for cat in categories:
    globals()[f"evaluate_model_{cat}"] = create_evaluate_model_dag(cat)