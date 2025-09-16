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
	@echo "make stage     - promote latest registered version to Staging"
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
	MLFLOW_TRACKING_URI=$(TRACKING) MLFLOW_EXPERIMENT=$(EXP) MODEL_NAME=$(MODEL) ALIAS=prod \
	$(PY) register_and_promote.py

stage:
	MLFLOW_TRACKING_URI=$(TRACKING) $(PY) - << 'PY'
from mlflow.tracking import MlflowClient
name="ml-demo-model"
c=MlflowClient()
latest=max(c.search_model_versions(f"name='{name}'"), key=lambda v: v.creation_timestamp)
c.set_registered_model_alias(name,"staging", latest.version)
print(f"@staging -> v{latest.version}")

prod:
	MLFLOW_TRACKING_URI=$(TRACKING) $(PY) - << 'PY'
from mlflow.tracking import MlflowClient
name="ml-demo-model"
c=MlflowClient()
# Pick the newest version by creation timestamp
latest=max(c.search_model_versions(f"name='{name}'"), key=lambda v: v.creation_timestamp)
c.set_registered_model_alias(name,"prod", latest.version)
print(f"@prod -> v{latest.version}")


PY
