# import mlflow.sklearn
from mlflow import MlflowClient
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

model_version = client.get_model_version_by_alias("metal_nut", "champion")
run = client.get_run(model_version.run_id)
params = run.data.params
model = mlflow.keras.load_model("models:/metal_nut@champion")

print(params)
print(model)
