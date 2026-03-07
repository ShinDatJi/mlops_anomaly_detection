from airflow.utils.dates import days_ago
from airflow.decorators import dag
from airflow.models import Variable, Param
from anomaly_detection.tasks.modeling import load_raw_data_task

categories = Variable.get(key="categories", default_var=[], deserialize_json=True)

def create_load_raw_data_dag(category, version):
    @dag(
        dag_id=f'load-raw-data_{category}_{version}',
        tags=['modeling', f"cat_{category}_{version}"],
        default_args={
            'owner': 'airflow',
            'start_date': days_ago(0, minute=1),
        },
        catchup=False,
        params={
            "category": Param(category, const=category, type="string"),
            "version": Param(version, const=version, type="string")
        }
    )
    def load_raw_data_dag():
        load_raw_data_task()

    return load_raw_data_dag()

for cat in categories:
    name = cat["name"]
    for ver in cat["versions"]:
        globals()[f"load_raw_data_{name}_{ver}"] = create_load_raw_data_dag(name, ver)
