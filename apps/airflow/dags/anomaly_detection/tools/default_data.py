from airflow.models import Param, Variable
from airflow.utils.dates import days_ago

def get_default_categories():
    # return ["bottle", "metal_nut"]
    return [
        {
            "name": "bottle",
            "versions": ["v1"]
        },
        {
            "name": "metal_nut",
            "versions": ["v1"]
        }
    ]
