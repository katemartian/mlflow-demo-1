import os
FAST = os.getenv("FAST_TRAIN") == "1"

from pathlib import Path
import json
import joblib
import pandas as pd

# Only import mlflow & heavy sklearn bits for NORMAL path
if not FAST:
    import mlflow, mlflow.sklearn
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
else:
    from sklearn.dummy import DummyClassifier  # tiny & instant

def main():
    export_path = Path("models/latest")
    export_path.mkdir(parents=True, exist_ok=True)

    # ---------- FAST PATH (CI) ----------
    if FAST:
        X = pd.DataFrame(
            [{"f1":0,"f2":0,"f3":0,"f4":0,"f5":0},
             {"f1":1,"f2":1,"f3":1,"f4":1,"f5":1}]
        )
        y = pd.Series([0, 1])

        model = DummyClassifier(strategy="most_frequent")
        model.fit(X, y)

        mapping = {f"f{i+1}": f"f{i+1}" for i in range(5)}
        (export_path / "schema.json").write_text(json.dumps({"feature_mapping": mapping}, indent=2))
        joblib.dump(model, export_path / "model.joblib")
        print("✅ FAST_TRAIN=1: dummy model written to models/latest (no MLflow used).")
        return  # <- IMPORTANT: exit before any MLflow code

    # ---------- NORMAL PATH ----------
    # Allow override via env; default to file store if you prefer:
    # TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    EXP_NAME = os.getenv("MLFLOW_EXPERIMENT", "ml-demo-1")

    import mlflow, mlflow.sklearn
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    cols = X.columns[:5]
    X = X[cols].copy()
    mapping = {orig: f"f{i+1}" for i, orig in enumerate(cols)}
    X.rename(columns=mapping, inplace=True)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
    ])

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXP_NAME)

    with mlflow.start_run():
        # If you ever hit permission/version issues with server artifacts:
        # mlflow.sklearn.autolog(log_models=False)
        mlflow.sklearn.autolog(log_models=True)

        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_te, proba)
        acc = accuracy_score(y_te, pred)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", acc)

        (export_path / "schema.json").write_text(json.dumps({"feature_mapping": mapping}, indent=2))
        joblib.dump(pipe, export_path / "model.joblib")
        print(f"✅ Training complete. AUC={auc:.3f}, ACC={acc:.3f}.")
        
if __name__ == "__main__":
    main()
