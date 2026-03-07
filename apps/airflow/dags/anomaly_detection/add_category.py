from airflow.utils.dates import days_ago
from airflow.decorators import dag, task
from airflow.models import Variable, Param
from airflow.operators.python import get_current_context

@task
def add_category_task():
    params = get_current_context()["params"]
    category = params["category"]
    version = params["version"]
    categories = Variable.get(key="categories", default_var=[], deserialize_json=True)
    category_exists = False
    for cat in categories:
        if cat["name"] == category:
            category_exists = True
            if version not in cat["versions"]:
                cat["versions"].append(version)
            break
    if not category_exists:
        categories.append({"name": category, "versions": [version]})
    print(categories)
    Variable.set(key="categories", value=categories, serialize_json=True)

@dag(
    dag_id='add-category',
    tags=['administration'],
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    },
    catchup=False,
    params={
        "category": Param("new_category", type="string"),
        "version": Param("v1", type="string")
    }
)
def add_category_dag():
    add_category_task()

add_category_dag()
