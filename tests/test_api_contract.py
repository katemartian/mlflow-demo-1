from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_dir" in data
    assert "model_exists" in data

def test_predict():
    payload = {
        "inputs": [
            {"f1": 0.1, "f2": -0.2, "f3": 1.1, "f4": 0.0, "f5": 2.3},
            {"f1": -1.0, "f2": 0.5, "f3": 0.0, "f4": -0.7, "f5": 1.7}
        ]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert out["n"] == len(payload["inputs"])
    assert isinstance(out["preds"], list)

def test_predict_invalid_type():
    payload = {
        "inputs": [
            {"f1": "bad", "f2": 1, "f3": 2, "f4": 3, "f5": 4}
        ]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422  # Unprocessable Entity from FastAPI validation

def test_predict_missing_field():
    payload = {
        "inputs": [
            {"f1": 1, "f2": 2, "f3": 3, "f4": 4}  # Missing f5
        ]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422

def test_predict_empty():
    payload = {
        "inputs": []
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert out["n"] == 0
    assert out["preds"] == []        