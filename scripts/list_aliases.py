import os
from mlflow.tracking import MlflowClient

MODEL_NAME = os.getenv("MODEL_NAME", "ml-demo-model")
TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

c = MlflowClient(tracking_uri=TRACKING)

versions = c.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    print(f"No versions found for {MODEL_NAME}")
else:
    for v in sorted(versions, key=lambda x: int(x.version)):
        # Fetch full version details to ensure aliases are populated
        full = c.get_model_version(MODEL_NAME, v.version)
        aliases = getattr(full, "aliases", None) or []
        # aliases may be list[str] or list objects; normalize to strings
        aliases = [str(a) for a in aliases]
        print(f"v{v.version}: aliases={sorted(aliases)}")
