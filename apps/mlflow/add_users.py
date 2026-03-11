from mlflow.server import get_app_client
import os

port = os.environ["MLFLOW_PORT"]
user = os.environ["MLFLOW_ADMIN_USER"]
password = os.environ["MLFLOW_ADMIN_PASSWORD"]

print(f"Adding user '{user}' to MLflow")

tracking_uri = f"http://mlflow:{port}/"

try:
    auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)

    auth_client.create_user(username=user, password=password)
    auth_client.update_user_admin(username=user, is_admin=True)

    auth_client.delete_user(username="admin")

except Exception as e:
    print(f"Error creating auth client: {e}")   
