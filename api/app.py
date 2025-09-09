from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import os
import joblib

MODEL_PATH = os.getenv("MODEL_PATH", "./models/latest")

app = FastAPI(tittle="ML API demo", version="0.1.0")

class Features(BaseModel):
    f1: float = Field(..., description="feature 1")
    f2: float = Field(..., description="feature 2")
    f3: float = Field(..., description="feature 3")
    f4: float = Field(..., description="feature 4")
    f5: float = Field(..., description="feature 5")

class PredicRequest(BaseModel):
    inputs: list[Features]

@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}

@app.on_event("startup")
def _load_model():
    model_file = os.path.join(MODEL_PATH, "model.joblib")
    app.state.model = joblib.load(model_file)

@app.post("/predict")
def predict(req: PredicRequest):
    df = pd.DataFrame([x.dict() for x in req.inputs])
    preds = app.state.model.predict(df)
    return {"preds": preds.tolist(), "n": len(preds)}
