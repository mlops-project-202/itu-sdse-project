from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Vial images
IMAGE_DATA_DIR = DATA_DIR / "images" / "raw"
IMAGE_METADATA_PATH = DATA_DIR / "images" / "metadata.csv"

# Artifacts loaded together with the model later
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
IMAGE_STATISTICS_PATH = ARTIFACTS_DIR / "image_statistics.json"

# Preprocessing
IMAGE_SIZE = (64, 64)
TRANSFORM_MEAN = [0.485, 0.456, 0.406]  # ImageNet statistics
TRANSFORM_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
RANDOM_STATE = 42

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
