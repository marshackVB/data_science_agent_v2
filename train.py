"""
Bank Marketing classification: predict if client subscribes to term deposit (y).

Optimizes F1 score via Optuna hyperparameter tuning. Logs to MLflow.
Supports XGBoost, LightGBM, CatBoost. Includes feature engineering and threshold tuning.
"""

import os
import numpy as np
import pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import optuna

# Load .env when running locally
if not os.getenv("DATABRICKS_RUNTIME_VERSION"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

CATALOG = "shared"
SCHEMA = "mlc_schema"
TABLE = "mlc_bank_marketing_train"
MLFLOW_EXPERIMENT = "/Users/marshall.carter@databricks.com/agent_experiment_v2"

# Feature columns (exclude target y)
NUMERIC_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]
# Exclude poutcome (use engineered), contact (use engineered contact_unknown), month (use target encoding)
CATEGORICAL_FEATURES = ["job", "marital", "education", "default", "housing", "loan"]
ENGINEERED_FEATURES = ["poutcome_success", "high_season", "contact_unknown", "was_contacted_before", "has_prior_contacts"]
MONTH_TARGET_FEATURE = "month_subscription_rate"
TARGET = "y"

CV_FOLDS = 5


def load_data(spark, sample_fraction: float | None = None):
    """Load training data from Delta table, optionally sample for local runs."""
    df = spark.read.table(f"{CATALOG}.{SCHEMA}.{TABLE}")
    pdf = df.toPandas()

    if sample_fraction is not None and sample_fraction < 1.0:
        pdf = pdf.sample(frac=sample_fraction, random_state=42)

    return pdf


def add_engineered_features(df):
    """Add features from EDA and feature importance."""
    df = df.copy()
    df["poutcome_success"] = (df["poutcome"] == "success").astype(int)
    df["high_season"] = df["month"].str.lower().isin(["mar", "sep", "oct", "dec"]).astype(int)
    df["contact_unknown"] = (df["contact"].str.lower() == "unknown").astype(int)
    df["was_contacted_before"] = (df["pdays"] >= 0).astype(int)
    df["has_prior_contacts"] = (df["previous"] > 0).astype(int)
    return df


def add_month_target_encoding(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]):
    """Replace month with subscription rate per month (from train). Returns X_train, X_test."""
    train_df = X_train.assign(_y=y_train.values)
    month_map = train_df.groupby("month")["_y"].mean().to_dict()
    global_mean = float(y_train.mean())

    def encode(df):
        out = df.copy()
        out[MONTH_TARGET_FEATURE] = out["month"].map(month_map).fillna(global_mean)
        out = out.drop(columns=["month"])
        return out[numeric_cols + categorical_cols]

    return encode(X_train), encode(X_test)


def get_preprocessor(categorical_features: list[str], numeric_features: list[str]):
    """Build sklearn preprocessor for numeric + categorical features."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def get_classifier(model_type: str, params: dict, random_state: int = 42):
    """Return classifier instance for model_type with given params."""
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params, random_state=random_state, n_jobs=-1)
    if model_type == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params, random_state=random_state, n_jobs=-1, verbosity=-1)
    if model_type == "catboost":
        import catboost as cb
        return cb.CatBoostClassifier(**params, random_state=random_state, thread_count=-1, verbose=0)
    raise ValueError(f"Unknown model_type: {model_type}")


def suggest_params(trial, model_type: str):
    """Suggest hyperparameters for Optuna based on model_type."""
    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 15.0),
        }
    if model_type == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 15.0),
        }
    if model_type == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 15.0),
        }
    raise ValueError(f"Unknown model_type: {model_type}")


def tune_threshold(y_true, y_proba, thresholds: np.ndarray | None = None):
    """Find threshold that maximizes F1 macro."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.5, 41)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        y_p = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_p, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def train_with_optuna(X, y, n_trials: int, model_type: str, numeric_cols: list[str], categorical_cols: list[str]):
    """Run Optuna study to maximize F1 (macro) via cross-validation."""

    def objective(trial):
        params = suggest_params(trial, model_type)
        clf = get_classifier(model_type, params)
        preprocessor = get_preprocessor(categorical_cols, numeric_cols)
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        scores = cross_val_score(pipe, X, y, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def run_training(
    sample_fraction: float | None = None,
    n_trials: int = 50,
    model_type: str = "catboost",
):
    """
    Full training workflow: load data, tune with Optuna, fit best model, log to MLflow.

    Args:
        sample_fraction: If < 1.0, sample this fraction for faster runs.
        n_trials: Number of Optuna trials.
        model_type: "xgboost", "lightgbm", or "catboost".
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    from get_spark import get_spark
    spark = get_spark()
    df = load_data(spark, sample_fraction=sample_fraction)
    df = add_engineered_features(df)

    # Features before month encoding (need month for target encoding)
    base_features = NUMERIC_FEATURES + ENGINEERED_FEATURES + ["month"] + CATEGORICAL_FEATURES
    X = df[base_features]
    y = (df[TARGET] == "yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Month target encoding (fit on train only)
    numeric_cols = NUMERIC_FEATURES + ENGINEERED_FEATURES + [MONTH_TARGET_FEATURE]
    categorical_cols = CATEGORICAL_FEATURES
    X_train, X_test = add_month_target_encoding(X_train, y_train, X_test, numeric_cols, categorical_cols)

    # Validation split for threshold tuning
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)

        study = train_with_optuna(X_fit, y_fit, n_trials, model_type, numeric_cols, categorical_cols)
        best_params = study.best_params

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_f1_macro", study.best_value)

        preprocessor = get_preprocessor(categorical_cols, numeric_cols)
        clf = get_classifier(model_type, best_params)
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_fit, y_fit)

        # Tune decision threshold on validation set
        y_val_proba = pipe.predict_proba(X_val)[:, 1]
        best_threshold = tune_threshold(y_val.values, y_val_proba)
        mlflow.log_param("decision_threshold", best_threshold)

        # Refit on full train
        pipe.fit(X_train, y_train)

        y_train_proba = pipe.predict_proba(X_train)[:, 1]
        y_test_proba = pipe.predict_proba(X_test)[:, 1]
        y_train_pred = (y_train_proba >= best_threshold).astype(int)
        y_test_pred = (y_test_proba >= best_threshold).astype(int)

        train_f1 = f1_score(y_train, y_train_pred, average="macro")
        test_f1 = f1_score(y_test, y_test_pred, average="macro")
        mlflow.log_metric("train_f1_macro", train_f1)
        mlflow.log_metric("test_f1_macro", test_f1)

        # Feature importances (XGB/LGB/CatBoost all have feature_importances_)
        preprocessor = pipe.named_steps["preprocessor"]
        classifier = pipe.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_
        importance_dict = dict(zip(feature_names, importances.tolist()))
        mlflow.log_dict(importance_dict, "feature_importance.json")
        sorted_importances = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )
        for i, (name, imp) in enumerate(sorted_importances[:10]):
            mlflow.log_param(f"importance_rank_{i+1}", f"{name}: {imp:.4f}")

        mlflow.sklearn.log_model(pipe, "model")

        print(f"Model: {model_type}")
        print(f"Best CV F1 (macro): {study.best_value:.4f}")
        print(f"Decision threshold: {best_threshold:.3f}")
        print(f"Train F1 (macro): {train_f1:.4f}")
        print(f"Test F1 (macro): {test_f1:.4f}")
        print("\nFeature importances (all, sorted):")
        for name, imp in sorted_importances:
            print(f"  {name}: {imp:.6f}")
        print("\nTest set classification report:")
        print(classification_report(y_test, y_test_pred, target_names=["no", "yes"]))

    return study, pipe


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="catboost", choices=["xgboost", "lightgbm", "catboost"], help="Model type to train")
    args, _ = parser.parse_known_args()  # Ignore job-injected params (--job-id, --job-run-id)
    run_training(model_type=args.model)
