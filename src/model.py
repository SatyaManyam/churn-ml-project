from __future__ import annotations
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report

from .features import FeatureSpec, build_preprocessor

@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "logreg"  # "logreg" or "rf"
    random_state: int = 42

def build_model_pipeline(spec: FeatureSpec, cfg: ModelConfig) -> Pipeline:
    pre = build_preprocessor(spec)

    if cfg.model_type == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=cfg.random_state,
        )
    elif cfg.model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=6,
            min_samples_leaf=3,
            n_jobs=-1,
            class_weight="balanced",
            random_state=cfg.random_state,
        )
    else:
        raise ValueError("model_type must be 'logreg' or 'rf'")

    return Pipeline(steps=[("preprocess", pre), ("model", clf)])

def evaluate_binary(model, X_test, y_test) -> dict:
    # proba for ROC-AUC
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback (rare)
        proba = model.decision_function(X_test)

    preds = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "report": classification_report(y_test, preds, digits=4),
    }
    return metrics
