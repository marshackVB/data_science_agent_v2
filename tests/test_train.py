"""Unit tests for train.py."""

import pytest
import pandas as pd

from train import (
    get_preprocessor,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)


def test_preprocessor_fit_transform():
    """Preprocessor fits and transforms sample data."""
    prep = get_preprocessor()
    n_rows = 50
    df = pd.DataFrame({
        **{f: list(range(n_rows)) for f in NUMERIC_FEATURES},
        **{f: ["a", "b"] * (n_rows // 2) for f in CATEGORICAL_FEATURES},
    })
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    Xt = prep.fit_transform(X)
    assert Xt.shape[0] == n_rows
    assert Xt.shape[1] >= len(NUMERIC_FEATURES)


def test_feature_columns_defined():
    """All feature columns are explicitly defined."""
    assert len(NUMERIC_FEATURES) > 0
    assert len(CATEGORICAL_FEATURES) > 0
    assert TARGET == "y"
