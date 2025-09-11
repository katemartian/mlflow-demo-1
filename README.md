# ML Demo Portfolio (FastAPI + MLflow + Docker)

End-to-end demo: train scikit-learn model, log runs to MLflow, and serve predictions via FastAPI service. Runs **locally and in Docker**.

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

## Screenshots

- MLflow UI
- FastAPI /docs page
- Example /predict response