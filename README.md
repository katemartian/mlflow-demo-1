# ML Demo Portfolio (FastAPI + MLflow + Docker)

[![Python CI](https://github.com/katemartian/mlflow-demo-1/actions/workflows/python-ci.yml/badge.svg)](https://github.com/katemartian/mlflow-demo-1/actions/workflows/python-ci.yml)

End-to-end **Machine Learning service** built with **FastAPI**, **MLflow**, and **Docker**, designed to showcase modern MLOps skills.  
This project demonstrates the full lifecycle: training, experiment tracking, containerization, testing, and serving predictions through an API.

---

## Highlights
- **FastAPI REST API** with typed `/predict` endpoint and input validation  
- **MLflow tracking** for experiments, metrics (AUC/Accuracy), and model registry  
- **Docker & Compose stack** (API + MLflow UI)  
- **Automated CI tests** via GitHub Actions (FAST mode for speed)  
- **Screenshots included**: API docs, MLflow UI, prediction examples, and passing tests  

---

## A) Run locally (WSL/Ubuntu)

```bash
python -m venv .venv1 && source .venv1/bin/activate
pip install -r requirements.txt
python src/train_local.py                   # Creates ./models/latest
uvicorn api.app:app --reload --port 8000    # http://127.0.0.1:8000/docs
```

## B) Run in Docker (single container)

```bash
docker build -t mlflow-demo-1 ./api
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models:ro" mlflow-demo-1
# Open http://127.0.0.1:8000/docs
```

## C) Run with Docker Compose (API + MLflow UI)

```bash
docker compose up --build
# MLflow UI: http://127.0.0.1:5000
# API docs:  http://127.0.0.1:8000/docs
```

## API Endpoints

.GET /health -> status + model visibility

.POST /predict -> JSON body:

```json
{
    "inputs": [
        {
            "f1": 0.1,
            "f2": -0.2,
            "f3": 1.1,
            "f4": 0.0,
            "f5": 2.3
        }
    ]
}
```

## Tests

```bash
python -m pytest -q
```

## Folder layout
```bash
.
├─ api/                 # FastAPI app + Dockerfile
├─ src/                 # training script (logs to ./mlruns, exports ./models/latest)
├─ models/latest/       # trained model (model.joblib, schema.json)
├─ mlruns/              # MLflow local tracking store (created at first run)
├─ tests/               # API tests
├─ docker-compose.yml   # API + MLflow UI stack
└─ README.md
```

## Screenshots

- FastAPI Docs  
  ![FastAPI Docs](docs/img/fastapi-docs.png)

- Predict Endpoint Response  
  ![Predict Example](docs/img/predict-example.png)

- MLflow UI (Experiments)  
  ![MLflow UI](docs/img/mlflow-ui.png)

- MLflow Run Details  
  ![MLflow Run](docs/img/mlflow-run-details.png)

- Tests Passing  
  ![Tests Passing](docs/img/tests-pass.png)


## Model Registry

This project also demonstrates **end-to-end model lifecycle management** using the **MLflow Model Registry**.

### Steps shown
1. **Log & Track**  
   Each training run logs metrics (AUC, Accuracy) and saves the trained model as an artifact.
2. **Register**  
   Logged models are registered into the **MLflow Model Registry**.
3. **Promote**  
   Versions are transitioned through stages (**Staging → Production**), simulating real-world workflows.

### Screenshots
| Step | Screenshot |
|------|-------------|
| Experiment run with metrics & artifacts | ![Experiment with metrics](docs/img/mlflow-experiment.png) |
| Model registered in Registry | ![Model registered](docs/img/mlflow-model-registered.png) |
| Model promoted to Production | ![Model in Production](docs/img/mlflow-model-prod.png) |

---

**Note on MLflow registry:** Stages (Staging/Production) are deprecated in newer MLflow versions.
This project uses **aliases** instead (e.g., `@staging`, `@prod`). Promotion is simply updating the alias to point to a new version:

```python
client.set_registered_model_alias("ml-demo-model", "prod", VERSION)
# load the production model:
mlflow.pyfunc.load_model("models:/ml-demo-model@prod")
```

### Makefile quickstart (registry)
```bash
make register   # register latest run and point alias @prod to it
make prod       # repoint @prod to the latest registered version
# (optional) make stage  # set alias @staging to latest
```

## 🏷️ Alias-Based Model Registry (MLflow 2.22+)

This project uses **MLflow aliases** (e.g., `@staging`, `@prod`) instead of legacy “Stages.”  
Aliases make promotion a simple pointer update — no deprecation warnings, easy to automate.

### Flow at a glance
1. **Train** a model and log artifacts to MLflow.  
2. **Register** the latest run as a new model version.  
3. **Promote** by pointing an alias (e.g., `@prod`) to that version.  
4. **Consume** the current production model via `models:/NAME@prod`.

---

### 📸 Screenshots
- **Run with metrics & artifacts**  
  ![Experiment run](docs/img/mlflow-experiment.png)
- **Registered from run (badge)**  
  ![Registered model from run](docs/img/mlflow-model-registered.png)
- **Aliases visible on version** (turn on *New model registry UI* toggle)  
  ![Aliases on version](docs/img/mlflow-aliases.png)

---

### 🧪 Train & Register
```bash
# start services (MLflow server + API)
make up

# full training (logs metrics + model artifact to MLflow)
make train

# register latest run and point alias @prod to it
make register
```


Under the hood, register calls `register_and_promote.py` to:

 - create a new model version from the latest finished run (`runs:/<RUN_ID>/model`)

 - set alias @prod to that version

## 🚦 Promote with Aliases

Promote (or repoint) aliases using simple Make targets:
```bash
# set alias @staging -> latest version
make stage

# set alias @prod -> latest version
make prod

# verify which versions aliases point to
make verify-aliases

# list every version with its aliases
make list-aliases
```

Behind the scenes:

- `scripts/set_alias.py` calls `MlflowClient.set_registered_model_alias(name, alias, version)`

- `scripts/verify_aliases.py` uses `get_model_version_by_alias(...)`

- `scripts/list_aliases.py` prints aliases across all versions

## 📦 Consume the Production Model

Load the model by alias and run a quick prediction:

```python
# src/load_and_predict.py
import mlflow, pandas as pd
mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.pyfunc.load_model("models:/ml-demo-model@prod")
X = pd.DataFrame([{"f1": 0.5, "f2": -1.2, "f3": 3.1, "f4": 0.7, "f5": 1.0}])
print("Prediction:", model.predict(X).tolist())
```
Or just:
```bash
make predict
```

## 💡 Notes & Tips

- MLflow stages are deprecated; aliases are the modern replacement.

- Aliases are just labels — promotion is a zero-copy pointer update.

- Want a quick status check?

```bash
make verify-aliases
# @staging -> ml-demo-model v5
# @prod    -> ml-demo-model v5
```