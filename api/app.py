from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import os
import joblib

# Resolve a default absolute path: <project_root>/models/latest
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "latest"

MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_DIR))

app = FastAPI(tittle="ML API demo", version="0.1.0")

class Features(BaseModel):
    f1: float = Field(..., description="feature 1")
    f2: float = Field(..., description="feature 2")
    f3: float = Field(..., description="feature 3")
    f4: float = Field(..., description="feature 4")
    f5: float = Field(..., description="feature 5")

class PredictRequest(BaseModel):
    inputs: list[Features]

@app.get("/health")
def health():
    model_file = MODEL_PATH / "model.joblib"
    return {
        "status": "ok", 
        "model_path": str(MODEL_PATH),
        "model_exists": model_file.exists(),
        "cwd": os.getcwd(),
        "base_dir": str(BASE_DIR)
        }

@app.on_event("startup")
def _load_model():
    model_file = MODEL_PATH / "model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}."
                                f"Trin first or set MODEL_PATH env var.")
    try:
        app.state.model = joblib.load(model_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_file}: {e}")

@app.post("/predict")
def predict(req: PredictRequest):
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    df = pd.DataFrame([x.dict() for x in req.inputs])
    try:
        preds = app.state.model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
    
    return {"preds": preds.tolist(), "n": len(preds)}
