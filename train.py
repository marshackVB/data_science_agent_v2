"""
Bank Marketing classification: predict if client subscribes to term deposit (y).

Optimizes F1 score via Optuna hyperparameter tuning. Logs to MLflow.
"""

import os
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import optuna
from xgboost import XGBClassifier

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
# Exclude duration: known only after call; inclusion would inflate metrics unrealistically
NUMERIC_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL_FEATURES = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
TARGET = "y"


def load_data(spark, sample_fraction: float | None = None):
    """Load training data from Delta table, optionally sample for local runs."""
    df = spark.read.table(f"{CATALOG}.{SCHEMA}.{TABLE}")
    pdf = df.toPandas()

    if sample_fraction is not None and sample_fraction < 1.0:
        pdf = pdf.sample(frac=sample_fraction, random_state=42)

    return pdf


def get_preprocessor():
    """Build sklearn preprocessor for numeric + categorical features."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def train_with_optuna(X, y, n_trials: int = 50):
    """Run Optuna study to maximize F1 (macro) via cross-validation."""

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 12)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 10.0)

        pipe = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("classifier", XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        scores = cross_val_score(pipe, X, y, cv=3, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def run_training(
    sample_fraction: float | None = None,
    n_trials: int = 50,
):
    """
    Full training workflow: load data, tune with Optuna, fit best model, log to MLflow.

    Args:
        sample_fraction: If < 1.0, sample this fraction for faster runs (e.g. local smoke test).
        n_trials: Number of Optuna trials.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    from get_spark import get_spark
    spark = get_spark()
    df = load_data(spark, sample_fraction=sample_fraction)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = (df[TARGET] == "yes").astype(int)

    with mlflow.start_run():
        study = train_with_optuna(X, y, n_trials=n_trials)
        best_params = study.best_params

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_f1_macro", study.best_value)

        # Fit final model on full training data with best params
        pipe = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("classifier", XGBClassifier(
                **best_params,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        f1 = f1_score(y, y_pred, average="macro")
        mlflow.log_metric("train_f1_macro", f1)

        mlflow.sklearn.log_model(pipe, "model")

        print(f"Best CV F1 (macro): {study.best_value:.4f}")
        print(f"Train F1 (macro): {f1:.4f}")
        print(classification_report(y, y_pred, target_names=["no", "yes"]))

    return study, pipe


if __name__ == "__main__":
    run_training()
