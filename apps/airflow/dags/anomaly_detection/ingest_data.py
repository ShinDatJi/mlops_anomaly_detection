from airflow.utils.dates import days_ago
from airflow.decorators import dag
from airflow.models import Param, Variable
from anomaly_detection.tasks.data import ingest_data_task

categories = Variable.get(key="categories", default_var=[], deserialize_json=True)

def create_ingest_data_dag(category):
    @dag(
        dag_id=f'ingest-data_{category}',
        tags=['data', f"cat_{category}"],
        default_args={
            'owner': 'airflow',
            'start_date': days_ago(0, minute=1),
        },
        catchup=False,
        params={
            "category": Param(category, const=category, type="string")
        }
    )
    def ingest_data_dag():
        ingest_data_task()
    
    return ingest_data_dag()

for cat in categories:
    globals()[f"ingest_data_{cat}"] = create_ingest_data_dag(cat)
