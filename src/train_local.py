from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import joblib

def main():
    # Load dataset
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    cols = X.columns[:5]    # Select first 5 features to match API schema
    X = X[cols]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocessing and model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("breast_cancer_classification")

    with mlflow.start_run():
        mlflow.sklearn.autolog(log_models=True)
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predict = (probabilities >= 0.5).astype(int)

        auc = roc_auc_score(y_test, probabilities)
        accuracy = accuracy_score(y_test, predict)

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("accuracy", accuracy)

        # Export model for serving
        export_path = Path("models/latest")
        export_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, export_path / "model.joblib")

        # Write a minimal MLmodel file so that mlflow.pyfunc.load_model can load it
        py_version = ".".join(map(str, tuple(pd.__version__.split(".")[:2])))
        (export_path / "MLmodel").write_text(f"""
        flavors:
          python_function:
            loader_module: mlflow.sklearn
            model_path: model.joblib
            python_version: "{py_version}"
        """)

    print(f"Training complete. AUC={auc:.3f}, ACC={accuracy:.3f}. Model exported to {export_path}")

if __name__ == "__main__":
    main()
