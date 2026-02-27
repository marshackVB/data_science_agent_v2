"""
Bank Marketing classification: predict if client subscribes to term deposit (y).

Optimizes F1 score via Optuna hyperparameter tuning. Logs to MLflow.
"""

import os
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import cross_val_score
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


def train_with_optuna(X, y, n_trials: int = 20):
    """Run Optuna study to maximize F1 (macro) via cross-validation."""

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 5, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])

        pipe = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
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
    n_trials: int = 20,
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
        clf_params = {k: v for k, v in best_params.items() if k != "class_weight"}
        class_weight = best_params.get("class_weight")
        if class_weight is not None:
            clf_params["class_weight"] = class_weight
        pipe = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("classifier", RandomForestClassifier(
                **clf_params,
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
