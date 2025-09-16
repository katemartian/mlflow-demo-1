import os
import mlflow
from mlflow.tracking import MlflowClient

TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "ml-demo-2")
MODEL_NAME = os.getenv("MODEL_NAME", "ml-demo-model")
ALIAS = os.getenv("ALIAS", "prod")  # you can override to 'staging' if you like

mlflow.set_tracking_uri(TRACKING)
client = MlflowClient()

# 1) Ensure experiment exists and grab most recent finished run
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Train once first.")

runs = client.search_runs(
    [exp.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=25,
)
if not runs:
    raise RuntimeError("No finished runs found to register.")
run = runs[0]
run_id = run.info.run_id
source = f"runs:/{run_id}/model"  # your training logs the model to artifact_path='model'

print(f"Using run_id={run_id}")
print(f"Source artifact: {source}")

# 2) Create or reuse registered model
try:
    client.create_registered_model(MODEL_NAME)
    print(f"Created registered model: {MODEL_NAME}")
except Exception:
    print(f"Registered model already exists: {MODEL_NAME}")

# 3) Create a new version from the run’s artifact
mv = client.create_model_version(
    name=MODEL_NAME,
    source=source,
    run_id=run_id,
    description="Auto-registered from latest successful training run.",
)
print(f"Created model version: v{mv.version}")

# 4) Optional metadata
client.set_model_version_tag(MODEL_NAME, mv.version, "registered_by", "programmatic")
client.set_model_version_tag(MODEL_NAME, mv.version, "experiment", EXPERIMENT_NAME)

# 5) Alias-based “promotion” (no stages)
client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
print(f"Set alias @{ALIAS} -> v{mv.version}")

# Tip to consumers:
print(f"Load with: mlflow.pyfunc.load_model('models:/{MODEL_NAME}@{ALIAS}')")
