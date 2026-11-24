import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
 
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from scripts.config import (
    image_data_dir,
    metadata_path,
    project_root,
    s3_bucket,
    s3_endpoint,
    test_data_dir,
    test_metadata_path,
)
 
app = typer.Typer()

def setup_s3_client():
    """Create a boto3 S3 client configured for Hetzner's public bucket."""
    return boto3.client(
        's3',
        endpoint_url= s3_endpoint,
        aws_access_key_id='',
        aws_secret_access_key='',
        config=Config(signature_version=UNSIGNED)
    )

def download_file(s3_client, key, output_path):
    """Download a single file from S3 using boto3. Returns True only if it was downloaded."""
    output_path = Path(output_path)
    if output_path.exists():
        return False
    try:
        s3_client.download_file(s3_bucket, key, str(output_path))
        return True
    except Exception as e:
        logger.warning(f"Failed to download {key}: {e}")
        return False

def download_files_from_metadata(metadata_path, output_dir, prefix=""):
    """Download all files listed in a metadata csv from the object store."""
    s3_client = setup_s3_client()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        rows = list(csv.DictReader(f))

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for row in rows:
            filename = row["filename"]
            key = f"{prefix}{filename}"
            output_path = Path(output_dir) / filename
            future = executor.submit(download_file, s3_client, key, output_path)
            futures[future] = (key, output_path)

        for future in tqdm(as_completed(futures), total=len(futures)):
            key, output_path = futures[future]
            if future.result():
                downloaded_count += 1
            elif output_path.exists():
                skipped_count += 1
            else:
                failed_count += 1

    logger.info(
        f"Downloaded {downloaded_count} images, "
        f"skipped {skipped_count} existing, failed {failed_count}"
    )

def pull_data():
    """Fetch the metadata csvs with DVC, then downloading the images."""
    # sys.executable + "-m dvc" uses the same environment as this script;
    # cwd=PROJ_ROOT makes sure DVC runs from the project root
    
    subprocess.run([sys.executable, "-m", "dvc", "pull"], check=True, cwd=project_root)
    download_files_from_metadata(metadata_path, image_data_dir, "images/raw/")
    download_files_from_metadata(test_metadata_path, test_data_dir, "test/")

def load_image_dataset(
    metadata_path=metadata_path, image_data_dir=image_data_dir
) -> list[tuple[Path, int]]:
    """(image_path, label) pairs for every metadata row whose image exists on disk."""
    if not Path(metadata_path).exists():
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}. Run `python -m scripts.dataset pull` first."
        )
    metadata = pd.read_csv(metadata_path)

    image_dataset = []
    missing = 0
    for filename, label in zip(metadata["filename"], metadata["class"]):
        image_path = Path(image_data_dir) / filename
        if image_path.exists():
            image_dataset.append((image_path, int(label)))
        else:
            missing += 1

    logger.info(f"Loaded {len(image_dataset)} labelled images from {image_data_dir}")
    if missing:
        logger.warning(f"{missing} image files listed in {metadata_path} were not found")
    return image_dataset


@app.command()
def pull():
    """Pull the metadata with DVC and download all images."""
    logger.info("Pulling data...")
    pull_data()
    logger.success("Data pull complete.")
 
 
@app.command()
def load():
    """Load the image dataset and report how many images were found."""
    image_dataset = load_image_dataset()
    logger.success(f"Dataset ready: {len(image_dataset)} images.")
 
 
if __name__ == "__main__":
    app()
