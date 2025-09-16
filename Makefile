PY=python
TRACKING=http://127.0.0.1:5000
EXP=ml-demo-2
MODEL=ml-demo-model

.PHONY: help up down train fast tests register promote prod

help:
	@echo "make up        - start mlflow + api via docker compose"
	@echo "make down      - stop stack"
	@echo "make train     - full training (logs to MLflow, logs model artifact)"
	@echo "make fast      - fast dummy training for tests/CI"
	@echo "make tests     - run pytest"
	@echo "make register  - register latest run’s model as a new version"
	@echo "make prod      - promote latest registered version to Production"

up:
	docker compose up --build

down:
	docker compose down

train:
	MLFLOW_TRACKING_URI=$(TRACKING) MLFLOW_EXPERIMENT=$(EXP) $(PY) src/train_local.py

fast:
	FAST_TRAIN=1 $(PY) src/train_local.py

tests:
	$(PY) -m pytest -q

register:
	MLFLOW_TRACKING_URI=$(TRACKING) MLFLOW_EXPERIMENT=$(EXP) MODEL_NAME=$(MODEL) \
	$(PY) register_and_promote.py

# Only promotion (if you prefer separating it):
prod:
	MLFLOW_TRACKING_URI=$(TRACKING) MLFLOW_EXPERIMENT=$(EXP) MODEL_NAME=$(MODEL) \
	$(PY) - << 'PY'
from mlflow.tracking import MlflowClient
import os
name = os.getenv("MODEL_NAME", "ml-demo-model")
client = MlflowClient()
latest = max(client.get_latest_versions(name), key=lambda v: v.creation_timestamp)
client.transition_model_version_stage(name, latest.version, stage="Production", archive_existing_versions=False)
client.set_registered_model_alias(name, "prod", latest.version)
print(f"Promoted {name} v{latest.version} → Production and set alias @prod")
PY
