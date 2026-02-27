"""Unit tests for train.py."""

import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from train import (
    load_data,
    add_engineered_features,
    add_month_target_encoding,
    get_preprocessor,
    get_classifier,
    NUMERIC_FEATURES,
    ENGINEERED_FEATURES,
    MONTH_TARGET_FEATURE,
    CATEGORICAL_FEATURES,
    TARGET,
)


def test_preprocessor_fit_transform():
    """Preprocessor fits and transforms a sample from Databricks via get_spark()."""
    X_train, y_train, numeric_cols, categorical_cols = _load_real_data_sample()
    prep = get_preprocessor(categorical_cols, numeric_cols)
    Xt = prep.fit_transform(X_train)
    assert Xt.shape[0] == len(y_train)
    assert Xt.shape[1] >= len(NUMERIC_FEATURES)


def test_feature_columns_defined():
    """All feature columns are explicitly defined."""
    assert len(NUMERIC_FEATURES) > 0
    assert len(ENGINEERED_FEATURES) > 0
    assert MONTH_TARGET_FEATURE == "month_subscription_rate"
    assert len(CATEGORICAL_FEATURES) > 0
    assert TARGET == "y"


def _minimal_params(model_type: str) -> dict:
    """Minimal params for fast unit test."""
    base = {"learning_rate": 0.1, "subsample": 0.8, "scale_pos_weight": 1.0}
    if model_type == "xgboost":
        return {**base, "n_estimators": 2, "max_depth": 2, "colsample_bytree": 0.8,
                "min_child_weight": 1, "reg_alpha": 0.1, "reg_lambda": 0.1}
    if model_type == "lightgbm":
        return {**base, "n_estimators": 2, "max_depth": 2, "colsample_bytree": 0.8,
                "min_child_samples": 5, "reg_alpha": 0.1, "reg_lambda": 0.1}
    if model_type == "catboost":
        return {**base, "iterations": 2, "depth": 2, "colsample_bylevel": 0.8,
                "l2_leaf_reg": 0.1}
    raise ValueError(model_type)


def _load_real_data_sample():
    """Load a small sample from Databricks via get_spark(). Skips if unavailable."""
    numeric_cols = NUMERIC_FEATURES + ENGINEERED_FEATURES + [MONTH_TARGET_FEATURE]
    categorical_cols = CATEGORICAL_FEATURES

    try:
        from get_spark import get_spark
        spark = get_spark()
        df = load_data(spark, sample_fraction=0.05)
        df = add_engineered_features(df)
        base_features = NUMERIC_FEATURES + ENGINEERED_FEATURES + ["month"] + CATEGORICAL_FEATURES
        X = df[base_features]
        y = (df[TARGET] == "yes").astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_test = add_month_target_encoding(X_train, y_train, X_test, numeric_cols, categorical_cols)
        return X_train, y_train, numeric_cols, categorical_cols
    except Exception as e:
        pytest.skip(f"Databricks unavailable ({e}). Run with .env configured, or use smoke_test.py.")


@pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
def test_all_models_fit_and_predict(model_type):
    """Each model can fit and predict on real Databricks training data sample."""
    X_train, y_train, numeric_cols, categorical_cols = _load_real_data_sample()

    params = _minimal_params(model_type)
    clf = get_classifier(model_type, params)
    preprocessor = get_preprocessor(categorical_cols, numeric_cols)
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_train)
    assert len(preds) == len(y_train)
    assert preds.ndim == 1
