import os
import mlflow
from mlflow.tracking import MlflowClient

TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "ml-demo-2")
MODEL_NAME = os.getenv("MODEL_NAME", "ml-demo-model")

mlflow.set_tracking_uri(TRACKING)
client = MlflowClient()

# 1) Get experiment id
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Train once first.")

# 2) Grab the most recent finished run that logged a `model/` artifact
#    (filter for finished status; order by start_time desc)
runs = client.search_runs(
    [exp.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=25,
)

run = None
for r in runs:
    # quick heuristic: the training script logged artifact_path='model'
    # so 'runs:/<run_id>/model' should exist
    run = r
    break

if run is None:
    raise RuntimeError("No finished runs found to register.")

run_id = run.info.run_id
source = f"runs:/{run_id}/model"  # path to the model artifact inside the run

print(f"Using run_id={run_id}")
print(f"Source artifact for registration: {source}")

# 3) Ensure registered model exists
try:
    client.create_registered_model(MODEL_NAME)
    print(f"Created registered model: {MODEL_NAME}")
except Exception:
    print(f"Registered model already exists: {MODEL_NAME}")

# 4) Create a new version from the run’s artifact
mv = client.create_model_version(
    name=MODEL_NAME,
    source=source,
    run_id=run_id,
    description="Auto-registered from latest successful training run.",
)
print(f"Created model version: {mv.version}")

# 5) Optional metadata
client.set_model_version_tag(MODEL_NAME, mv.version, "registered_by", "programmatic")
client.set_model_version_tag(MODEL_NAME, mv.version, "experiment", EXPERIMENT_NAME)

# 6) Transition to STAGING
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=mv.version,
    stage="Staging",
    archive_existing_versions=False,
)
print(f"Version {mv.version} transitioned to Staging")

# 7) (Optional) Promote to PRODUCTION and set alias
PROMOTE_TO_PROD = True
if PROMOTE_TO_PROD:
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=False,
    )
    # Aliases are handy: mlflow://models/ml-demo-model@prod
    client.set_registered_model_alias(MODEL_NAME, "prod", mv.version)
    print(f"Version {mv.version} transitioned to Production and alias @prod set")
