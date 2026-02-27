import os

# Load .env when running locally (no DATABRICKS_RUNTIME_VERSION)
if not os.getenv("DATABRICKS_RUNTIME_VERSION"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession


def get_spark():
    """Return SparkSession on Databricks, DatabricksSession via Connect when local."""
    # Running inside Databricks (jobs/notebooks/serverless)
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        return SparkSession.builder.getOrCreate()

    # Running locally — use Databricks Connect (OAuth via DATABRICKS_HOST, etc.)
    # Expects DATABRICKS_HOST, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET in .env
    # Enable serverless compute if not already set (e.g. in .databrickscfg)
    if not os.getenv("DATABRICKS_SERVERLESS_COMPUTE_ID"):
        os.environ["DATABRICKS_SERVERLESS_COMPUTE_ID"] = "auto"
    try:
        return DatabricksSession.builder.serverless().getOrCreate()
    except Exception:
        raise RuntimeError(
            "Databricks spark session creation failed. "
            "Ensure DATABRICKS_HOST, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET are set."
        )
