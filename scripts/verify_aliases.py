import os
from mlflow.tracking import MlflowClient

MODEL_NAME = os.getenv("MODEL_NAME", "ml-demo-model")
TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

c = MlflowClient(tracking_uri=TRACKING)

for alias in ("staging", "prod"):
    try:
        mv = c.get_model_version_by_alias(MODEL_NAME, alias)
        print(f"@{alias} -> {MODEL_NAME} v{mv.version}")
    except Exception as e:
        print(f"@{alias} not set ({e})")
