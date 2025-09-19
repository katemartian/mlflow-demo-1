import os
import mlflow
import pandas as pd

MODEL_NAME = os.getenv("MODEL_NAME", "ml-demo-model")
ALIAS = os.getenv("ALIAS", "prod")
TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

mlflow.set_tracking_uri(TRACKING)

# load model by alias (e.g., models:/ml-demo-model@prod)
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

# sample row with your API schema f1..f5 (adjust if you changed features)
X = pd.DataFrame([{"f1": 0.5, "f2": -1.2, "f3": 3.1, "f4": 0.7, "f5": 1.0}])

pred = model.predict(X)

print(f"Loaded: models:/{MODEL_NAME}@{ALIAS}")
print("Input:")
print(X.to_string(index=False))
print("Prediction:", pred.tolist())
