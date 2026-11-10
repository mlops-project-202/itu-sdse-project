"""Turn the labelled vial images into model inputs.

Covers the "Feature artifacts" and "Data preprocessing" sections of
notebooks/01_training.ipynb: image transforms, the PyTorch dataset, the
stratified train/val/test split, DataLoaders, and the dataset artifacts
(class names and image statistics).
"""

import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import typer

from scripts.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    IMAGE_DATA_DIR,
    IMAGE_METADATA_PATH,
    IMAGE_SIZE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TRANSFORM_MEAN,
    TRANSFORM_STD,
)

app = typer.Typer()

def save_class_names(metadata: pd.DataFrame, path: Path) -> dict[int, str]:
    """Save the label -> class name mapping, taken from the metadata so it always matches the data."""
    mapping = metadata[["class", "class_name"]].drop_duplicates().sort_values("class")
    if mapping["class"].duplicated().any():
        raise ValueError(f"A class label has more than one class name in the metadata:\n{mapping}")
    class_names = {int(label): name for label, name in zip(mapping["class"], mapping["class_name"])}
    with open(path, "w") as f:
        json.dump(class_names, f, indent=2)
    logger.info(f"Class names {class_names} saved to {path}")
    return class_names


def calculate_image_statistics(data_loader: DataLoader) -> dict:
    """Mean and standard deviation per channel, averaged over the images in the loader.

    The std is the mean of the per-image stds, not the std over all pixels.
    """
    total_mean = np.zeros(3)
    total_std = np.zeros(3)
    total_samples = 0

    for images, _ in data_loader:
        n = images.shape[0]
        pixels = images.numpy().transpose(0, 2, 3, 1).reshape(n, -1, 3)
        total_mean += pixels.mean(axis=1).sum(axis=0)
        total_std += pixels.std(axis=1).sum(axis=0)
        total_samples += n

    stats = {
        "mean": (total_mean / total_samples).tolist(),
        "std": (total_std / total_samples).tolist(),
        "total_samples": total_samples,
    }
    logger.info(f"Image statistics: mean={stats['mean']}, std={stats['std']}")
    return stats


def define_transforms(image_size=IMAGE_SIZE, mean=TRANSFORM_MEAN, std=TRANSFORM_STD):
    """Return (train_transforms, val_transforms). Augmentation is applied to training only."""
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transforms, val_transforms


class GlassVialDataset(Dataset):
    """Loads vial images and their labels and applies the transform."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            image = img.convert("RGB") if img.mode != "RGB" else img.copy()

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


def load_image_dataset(metadata: pd.DataFrame, image_dir: Path) -> list[tuple[Path, int]]:
    """(image_path, label) pairs for every metadata row whose image exists on disk."""
    image_dataset = []
    missing = 0
    for filename, label in zip(metadata["filename"], metadata["class"]):
        image_path = Path(image_dir) / filename
        if image_path.exists():
            image_dataset.append((image_path, int(label)))
        else:
            missing += 1

    logger.info(f"Loaded {len(image_dataset)} labelled images from {image_dir}")
    if missing:
        logger.warning(f"{missing} image files listed in the metadata were not found")
    return image_dataset


def split_dataset(image_dataset, train_ratio=0.70, val_ratio=0.15, random_seed=RANDOM_STATE):
    """Stratified split so both classes keep the same balance in every set.

    The test set gets whatever is left after train and val.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    class_0 = [(path, label) for path, label in image_dataset if label == 0]
    class_1 = [(path, label) for path, label in image_dataset if label == 1]
    np.random.shuffle(class_0)
    np.random.shuffle(class_1)

    def split_class(data):
        train_end = int(len(data) * train_ratio)
        val_end = train_end + int(len(data) * val_ratio)
        return data[:train_end], data[train_end:val_end], data[val_end:]

    train_0, val_0, test_0 = split_class(class_0)
    train_1, val_1, test_1 = split_class(class_1)

    splits = (train_0 + train_1, val_0 + val_1, test_0 + test_1)
    for name, split in zip(("train", "val", "test"), splits):
        np.random.shuffle(split)
        labels = [label for _, label in split]
        logger.info(
            f"{name}: {len(split)} images (class 0: {labels.count(0)}, class 1: {labels.count(1)})"
        )
    return splits


def make_dataloader(pairs, transform, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
    """DataLoader over (image_path, label) pairs."""
    paths, labels = zip(*pairs) if pairs else ([], [])
    dataset = GlassVialDataset(list(paths), list(labels), transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    train_transforms,
    val_transforms,
    batch_size=BATCH_SIZE,
    num_workers=0,
):
    """Train loader (augmented, shuffled) plus val and test loaders (no augmentation)."""
    train_loader = make_dataloader(train_dataset, train_transforms, batch_size, True, num_workers)
    val_loader = make_dataloader(val_dataset, val_transforms, batch_size, False, num_workers)
    test_loader = make_dataloader(test_dataset, val_transforms, batch_size, False, num_workers)
    logger.info(
        f"DataLoaders (batch_size={batch_size}): train {len(train_loader)}, "
        f"val {len(val_loader)}, test {len(test_loader)} batches"
    )
    return train_loader, val_loader, test_loader


def save_split(pairs, path: Path) -> None:
    """Write (image_path, label) pairs to a csv so the training step can rebuild the loaders."""
    pd.DataFrame(
        [(str(image_path), label) for image_path, label in pairs], columns=["image_path", "label"]
    ).to_csv(path, index=False)
    

@app.command()
def main(
    metadata_path: Path = IMAGE_METADATA_PATH,
    image_dir: Path = IMAGE_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
):
    """Split the images, save the splits to csv and save the dataset artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    save_class_names(metadata, artifacts_dir / "class_names.json")

    image_dataset = load_image_dataset(metadata, image_dir)
    splits = split_dataset(image_dataset)
    for name, split in zip(("train", "val", "test"), splits):
        save_split(split, output_dir / f"{name}.csv")
    logger.info(f"Splits saved to {output_dir}")

    # Statistics of the real training images: no augmentation, so they are the same every run
    _, val_transforms = define_transforms()
    image_stats = calculate_image_statistics(make_dataloader(splits[0], val_transforms))
    stats_path = artifacts_dir / "image_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(image_stats, f, indent=2)
    logger.info(f"Image statistics saved to {stats_path}")

    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()
