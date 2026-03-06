from airflow.utils.dates import days_ago
from airflow.decorators import dag
from airflow.models import Variable, Param
from anomaly_detection.tasks.modeling import load_config_task

categories = Variable.get(key="categories", default_var=[], deserialize_json=True)

def create_load_config_dag(category):
    @dag(
        dag_id=f'load-config_{category}',
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
        }
    )
    def load_config_dag():
        load_config_task()

    return load_config_dag()

for cat in categories:
    globals()[f"load_config_{cat}"] = create_load_config_dag(cat)
