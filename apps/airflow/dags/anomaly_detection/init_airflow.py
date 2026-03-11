from airflow.utils.dates import days_ago
from airflow.decorators import dag, task
from anomaly_detection.tools.default_data import get_default_categories
from airflow.models import Variable

@task
def init_categories_task():
    default_categories = get_default_categories()
    categories = Variable.get(key="categories", default_var=default_categories, deserialize_json=True)
    print(categories)
    Variable.set(key="categories", value=categories, serialize_json=True)

@dag(
    dag_id='init-airflow',
    tags=['administration'],
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    },
    catchup=False
)
def init_airflow_dag():
    init_categories_task()

init_airflow_dag()
