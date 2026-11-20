from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

project_root=Path(__file__).resolve().parents[1]

max_date = "2024-01-31"
min_date = "2024-01-01"

# Training configuration
num_epochs = 5
batch_size = 32
learning_rate = 0.003
threshold = 0.5
early_stop_patience = 2
scheduler_patience = 3
random_state = 42

train_ratio=0.70
val_ratio=0.15
test_ratio=0.15

# Image specifications
image_size = (64, 64)
original_image_size = (1600, 768)
num_classes = 2
transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]
class_names = {0: "clean", 1: "defective"}  # must match the metadata's class/class_name columns



# Paths (relative to the project root)
data_dir = "data"
image_data_dir = "./data/images/raw"
metadata_path = "./data/images/metadata.csv"
test_data_dir = "./data/test"
test_metadata_path = test_data_dir + "/metadata.csv"

# Object store (public Hetzner bucket)
s3_endpoint = "https://itudsip.hel1.your-objectstorage.com"
s3_bucket = "data"

# Artifacts (relative to the project root)
artifact_dir = "artifacts"
pytorch_model_path = artifact_dir + "/vial_cnn_model.pth"
model_results_path = artifact_dir + "/model_results.json"
drift_baseline_stats_path = artifact_dir + "/drift_baseline_stats.json"
class_names_path = artifact_dir + "/class_names.json"
image_statistics_path = artifact_dir + "/image_statistics.json"
inference_log_path = artifact_dir + "/inference_log.json"


# MLflow - we log to a local sqlite db so no tracking server has to be running

mlflow_state_dir = "mlflow-state"
experiment_name = "vial_image_classification"
model_name = "cnn_model"
mlflow_tracking_uri = f"sqlite:///{mlflow_state_dir}/mlflow.db"
mlflow_artifact_location = "./mlflow-state/artifacts"
artifact_path = "model_pyfunc"
training_run_name = "training_notebook"

# deployment
# Where the exported servable model goes (this is what gets baked into the Docker image)
export_dir = "models/model"
model_path = export_dir

# Serving ports
port = 5001
upstream_port = 5002
upstream_host = "127.0.0.1"

# Where the endpoint lives in production
prod_endpoint = "https://mlops-itu-jtk.ondigitalocean.app"

# Drift detection
drift_default_days = 7
drift_ks_pvalue_threshold = 0.05
drift_mean_diff_threshold = 0.1

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

