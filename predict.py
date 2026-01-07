from __future__ import annotations
import argparse
import json
import joblib
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Predict churn using a trained model.")
    p.add_argument("--model_path", type=str, default="artifacts/model.joblib", help="Path to saved model.joblib.")
    p.add_argument("--input_json", type=str, required=True, help="JSON string or path to a JSON file with one record.")
    return p.parse_args()

def load_record(input_json: str) -> dict:
    # If it's a file path, read it; otherwise treat as JSON string
    try:
        with open(input_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return json.loads(input_json)

def main():
    args = parse_args()
    bundle = joblib.load(args.model_path)
    pipeline = bundle["pipeline"]

    record = load_record(args.input_json)
    X = pd.DataFrame([record])

    proba = pipeline.predict_proba(X)[:, 1][0]
    pred = int(proba >= 0.5)

    print(json.dumps({"churn_probability": float(proba), "churn_prediction": pred}, indent=2))

if __name__ == "__main__":
    main()
