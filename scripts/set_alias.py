# scripts/set_alias.py
import argparse
import os
from mlflow.tracking import MlflowClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "ml-demo-model"),
                        help="Registered model name")
    parser.add_argument("--alias", required=True, help="Alias to set (e.g., staging, prod)")
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    client = MlflowClient(tracking_uri=tracking_uri)

    # find latest model version
    versions = client.search_model_versions(f"name='{args.model}'")
    if not versions:
        raise RuntimeError(f"No versions found for model {args.model}")

    latest = max(versions, key=lambda v: v.creation_timestamp)

    client.set_registered_model_alias(args.model, args.alias, latest.version)
    print(f"Alias @{args.alias} -> {args.model} v{latest.version}")

if __name__ == "__main__":
    main()
