from __future__ import annotations
from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass(frozen=True)
class FeatureSpec:
    target: str = "churn"
    numeric: List[str] = None
    categorical: List[str] = None

    def __post_init__(self):
        if self.numeric is None or self.categorical is None:
            object.__setattr__(self, "numeric", ["tenure_months", "monthly_charges", "total_charges"])
            object.__setattr__(self, "categorical", ["contract", "internet_service", "has_support", "paperless_billing"])

def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, spec.numeric),
            ("cat", categorical_pipe, spec.categorical),
        ],
        remainder="drop",
    )
    return preprocessor

def split_xy(df: pd.DataFrame, spec: FeatureSpec):
    X = df.drop(columns=[spec.target])
    y = df[spec.target].astype(int)
    return X, y
