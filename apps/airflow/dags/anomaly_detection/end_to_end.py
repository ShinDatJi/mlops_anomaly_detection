from airflow.utils.dates import days_ago
from airflow.decorators import dag, task, task_group
from airflow.models import Param, Variable
from airflow.operators.python import get_current_context
from anomaly_detection.tools.config_tools import load_default_config, load_config, save_config
from anomaly_detection.tools.config_tools import get_params_from_config, get_config_from_params
from anomaly_detection.tasks.data import ingest_data_task
from anomaly_detection.tasks.modeling import load_raw_data_task, load_config_task
from anomaly_detection.tasks.modeling import preprocess_data_task, train_model_task, evaluate_model_task

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
        config = load_config(category)
        config.update({
            "preparation": get_config_from_params(params["preparation"]),
            "preprocessing": get_config_from_params(params["preprocessing"]),
            "modeling": get_config_from_params(params["modeling"]),
            "training": get_config_from_params(params["training"]),
            "evaluation": get_config_from_params(params["evaluation"])
        })
        save_config(category, config)

def create_end_to_end_dag(category):
    preparation_params = get_params_from_config(default_config["preparation"])
    preprocessing_params = get_params_from_config(default_config["preprocessing"])
    modeling_params = get_params_from_config(default_config["modeling"])
    training_params = get_params_from_config(default_config["training"])
    evaluation_params = get_params_from_config(default_config["evaluation"])
    @dag(
        dag_id=f'end-to-end_{category}',
        tags=['modeling', f"cat_{category}"],
        default_args={
            'owner': 'airflow',
            'start_date': days_ago(0, minute=1),
        },
        catchup=False,
        params={
            "category": Param(category, const=category, type="string"),
            "config_from_model_registry": Param("1", type="string", enum=["0", "1"]),
            "config_from_model_registry_alias": Param("champion", type="string"),
            "override_config_params": Param(False, type="boolean"),
            "preparation": Param(preparation_params, type="object", section="1 Preparation"),
            "preprocessing": Param(preprocessing_params, type="object", section="2 Preprocessing"),
            "modeling": Param(modeling_params, type="object", section="3 Modeling"),
            "training": Param(training_params, type="object", section="4 Training"),
            "evaluation": Param(evaluation_params, type="object", section="5 Evaluation"),
        }
    )
    def end_to_end_dag():
        [data_tasks(),  config_tasks()] >> modeling_tasks()
    
    return end_to_end_dag()

for cat in categories:
    globals()[f"end_to_end_{cat}"] = create_end_to_end_dag(cat)