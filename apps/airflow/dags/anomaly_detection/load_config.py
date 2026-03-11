from airflow.utils.dates import days_ago
from airflow.decorators import dag
from airflow.models import Variable, Param
import os
from anomaly_detection.tasks.modeling import load_config_task

modeling_project_path = os.environ["MODELING_PROJECT_PATH"]
default_config_file = os.path.join(modeling_project_path, "default.config.json")

categories = Variable.get(key="categories", default_var=[], deserialize_json=True)

def create_load_config_dag(category, version):
    @dag(
        dag_id=f'load-config_{category}_{version}',
        tags=['modeling', f"cat_{category}_{version}"],
        default_args={
            'owner': 'airflow',
            'start_date': days_ago(0, minute=1),
        },
        catchup=False,
        params={
            "category": Param(category, const=category, type="string"),
            "version": Param(version, const=version, type="string"),
            "config_file": Param(default_config_file, type="string"),
            "config_from_model_registry": Param("1", type="string", enum=["0", "1"]),
            "config_from_model_registry_category": Param(category, type="string"),
            "config_from_model_registry_version": Param("pretrained", type="string"),
            "config_from_model_registry_alias": Param("champion", type="string"),
        }
    )
    def load_config_dag():
        load_config_task()

    return load_config_dag()

for cat in categories:
    name = cat["name"]
    for ver in cat["versions"]:
        globals()[f"load_config_{name}_{ver}"] = create_load_config_dag(name, ver)
