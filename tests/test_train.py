"""Unit tests for train.py."""

import pytest
import pandas as pd

from train import (
    get_preprocessor,
    NUMERIC_FEATURES,
    ENGINEERED_FEATURES,
    MONTH_TARGET_FEATURE,
    CATEGORICAL_FEATURES,
    TARGET,
)


def test_preprocessor_fit_transform():
    """Preprocessor fits and transforms sample data."""
    numeric_cols = NUMERIC_FEATURES + ENGINEERED_FEATURES + [MONTH_TARGET_FEATURE]
    categorical_cols = CATEGORICAL_FEATURES
    prep = get_preprocessor(categorical_cols, numeric_cols)
    n_rows = 50
    df = pd.DataFrame({
        **{f: list(range(n_rows)) for f in NUMERIC_FEATURES},
        **{f: [0, 1] * (n_rows // 2) for f in ENGINEERED_FEATURES},
        **{MONTH_TARGET_FEATURE: [0.1] * n_rows},
        **{f: ["a", "b"] * (n_rows // 2) for f in CATEGORICAL_FEATURES},
    })
    X = df[numeric_cols + categorical_cols]
    Xt = prep.fit_transform(X)
    assert Xt.shape[0] == n_rows
    assert Xt.shape[1] >= len(NUMERIC_FEATURES)


def test_feature_columns_defined():
    """All feature columns are explicitly defined."""
    assert len(NUMERIC_FEATURES) > 0
    assert len(ENGINEERED_FEATURES) > 0
    assert MONTH_TARGET_FEATURE == "month_subscription_rate"
    assert len(CATEGORICAL_FEATURES) > 0
    assert TARGET == "y"
