from airflow.utils.dates import days_ago
from airflow.decorators import dag
from airflow.models import Variable, Param
from anomaly_detection.tasks.modeling import load_raw_data_task

categories = Variable.get(key="categories", default_var=[], deserialize_json=True)

def create_load_raw_data_dag(category):
    @dag(
        dag_id=f'load-raw-data_{category}',
        tags=['modeling', f"cat_{category}"],
        default_args={
            'owner': 'airflow',
            'start_date': days_ago(0, minute=1),
        },
        catchup=False,
        params={
            "category": Param(category, const=category, type="string")
        }
    )
    def load_raw_data_dag():
        load_raw_data_task()

    return load_raw_data_dag()

for cat in categories:
    globals()[f"load_raw_data_{cat}"] = create_load_raw_data_dag(cat)
