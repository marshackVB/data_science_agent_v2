# Bank Marketing Classification

Binary classification to predict whether a client will subscribe to a term deposit. Optimizes **F1 score** via Optuna hyperparameter tuning.

## Data

- **Table**: `shared.mlc_schema.mlc_bank_marketing_train`
- **Target**: `y` (yes/no)
- **Features**: age, job, balance, contact type, campaign metrics, etc.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install numpy pandas scikit-learn optuna mlflow-skinny pyspark databricks-connect python-dotenv zstandard pytest
```

For local runs, create a `.env` file with Databricks Connect credentials:
- `DATABRICKS_HOST`
- `DATABRICKS_CLIENT_ID`
- `DATABRICKS_CLIENT_SECRET`

## Usage

**Full training** (Databricks or local; uses full table):

```bash
python train.py
```

**Smoke test** (10% sample, 3 Optuna trials):

```bash
python smoke_test.py
```

**Unit tests**:

```bash
pytest tests/ -v
```

## MLflow

Results are logged to: `/Users/marshall.carter@databricks.com/agent_experiment_v2`
