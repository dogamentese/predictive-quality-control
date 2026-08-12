"""Leakage-safe preprocessing pipeline for the SECOM data.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def split_secom(test_size: float = 0.20, random_state: int = 42):
    # stratified train/test split 
    from src.data import load_secom

    df = load_secom()
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    X = df[sensor_cols]
    y = df["fail"]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


class DropHighMissing(BaseEstimator, TransformerMixin):
    # drops columns whose fraction of missing values exceeds "threshold"
    # to make it leakage safe, we learn which columns get dropped in "fit"
    # and then we apply it to the test fold
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        missing_frac = X.isna().mean()
        self.columns_kept_ = missing_frac[missing_frac <= self.threshold].index.tolist()
        self.columns_dropped_ = missing_frac[missing_frac > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.columns_kept_]

    def get_feature_names_out(self, input_features=None):
        # lets the pipeline keep real column names flowing through
        return np.asarray(self.columns_kept_, dtype=object)


def preprocessing_steps(missing_threshold: float = 0.5):
    """The ordered preprocessing steps, as (name, transformer) pairs
    to be used in 'build_preprocessor'.
    """
    return [
        ("drop_high_missing", DropHighMissing(threshold=missing_threshold)),
        ("impute", SimpleImputer(strategy="median")),
        ("drop_constant", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
    ]
 
 
def build_preprocessor(missing_threshold: float = 0.5) -> Pipeline:
    """Return the unfitted preprocessing pipeline
    so that preprocessing is re-fit inside every fold (leak-safe cross validation).
    """
    pipe = Pipeline(steps=preprocessing_steps(missing_threshold))
    # keep DataFrame output with real column names through every step
    pipe.set_output(transform="pandas")
    return pipe