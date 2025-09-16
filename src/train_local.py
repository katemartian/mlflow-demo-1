import os
import json
from pathlib import Path
import joblib
import pandas as pd
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--register", action="store_true", help="Register model after training")
    p.add_argument("--stage", choices=["None","Staging","Production"], default="None")
    return p.parse_args()

# detect fast mode (for CI)
FAST = os.getenv("FAST_TRAIN") == "1"

def main():
    args = parse_args()

    export_path = Path("models/latest")
    export_path.mkdir(parents=True, exist_ok=True)

    # =====================
    # FAST PATH (CI / dummy)
    # =====================
    if FAST:
        from sklearn.dummy import DummyClassifier

        X = pd.DataFrame(
            [
                {"f1": 0, "f2": 0, "f3": 0, "f4": 0, "f5": 0},
                {"f1": 1, "f2": 1, "f3": 1, "f4": 1, "f5": 1},
            ]
        )
        y = pd.Series([0, 1])

        model = DummyClassifier(strategy="most_frequent")
        model.fit(X, y)

        mapping = {f"f{i+1}": f"f{i+1}" for i in range(5)}
        (export_path / "schema.json").write_text(
            json.dumps({"feature_mapping": mapping}, indent=2)
        )
        joblib.dump(model, export_path / "model.joblib")

        print("✅ FAST_TRAIN=1: dummy model written to models/latest (no MLflow).")
        return

    # =====================
    # NORMAL PATH (with MLflow)
    # =====================
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score

    # dataset
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    cols = X.columns[:5]
    X = X[cols].copy()

    mapping = {orig: f"f{i+1}" for i, orig in enumerate(cols)}
    X.rename(columns=mapping, inplace=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ml-demo-2")

    with mlflow.start_run() as run:
        # train
        pipe.fit(X_tr, y_tr)

        # metrics
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_te, proba)
        acc = accuracy_score(y_te, pred)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_params({"model": "LogisticRegression", "max_iter": 1000})

        # log model with signature + example
        input_example = X_te.head(3)
        signature = infer_signature(X_te, pipe.predict(X_te))
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )
        if args.register:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            mv = client.create_model_version(
                name=os.getenv("MODEL_NAME","ml-demo-model"),
                source=f"runs:/{run.info.run_id}/model",
                run_id=run.info.run_id,
            )
            if args.stage != "None":
                client.set_registered_model_alias(
                    name=mv.name,
                    version=mv.version,
                    alias=args.stage
                )
                
        # save local serving copy
        schema_file = export_path / "schema.json"
        schema_file.write_text(json.dumps({"feature_mapping": mapping}, indent=2))
        mlflow.log_artifact(str(schema_file), artifact_path="schema")
        joblib.dump(pipe, export_path / "model.joblib")

        print(
            f"✅ Training complete. AUC={auc:.3f}, ACC={acc:.3f}. "
            f"Model exported to {export_path}"
        )

if __name__ == "__main__":
    main()
