from __future__ import annotations
import argparse
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.features import FeatureSpec, split_xy
from src.model import ModelConfig, build_model_pipeline, evaluate_binary
from src.utils import ensure_dir, save_json

def parse_args():
    p = argparse.ArgumentParser(description="Train a churn prediction model.")
    p.add_argument("--data", type=str, default=None, help="Path to CSV file. If omitted, uses synthetic data.")
    p.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf"], help="Model type.")
    p.add_argument("--outdir", type=str, default="artifacts", help="Output directory for model + metrics.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = load_data(args.data)
    spec = FeatureSpec()
    X, y = split_xy(df, spec)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, stratify=y
    )

    cfg = ModelConfig(model_type=args.model, random_state=args.seed)
    pipeline = build_model_pipeline(spec, cfg)

    pipeline.fit(X_train, y_train)
    metrics = evaluate_binary(pipeline, X_test, y_test)

    model_path = outdir / "model.joblib"
    metrics_path = outdir / "metrics.json"

    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_spec": spec.__dict__,
            "model_config": cfg.__dict__,
        },
        model_path
    )
    save_json({k: v for k, v in metrics.items() if k != "report"}, metrics_path)

    print("\n=== Training Complete ===")
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print("\nMetrics:")
    print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"F1:       {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:\n")
    print(metrics["report"])

if __name__ == "__main__":
    main()
