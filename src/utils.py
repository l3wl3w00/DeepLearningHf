import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import pandas as pd
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms


def setup_logger(name=__name__, log_file="log/run.log"):
    """Return a logger that writes to stdout and to log/run.log."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    # stdout
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_config(module):
    """
    Convert a config module with uppercase constants into a simple namespace.

    Parameters
    ----------
    module: Python module
        A module object (e.g., imported ``config``) containing configuration
        constants in uppercase. This helper mirrors them into an object for
        convenient attribute access and validation.
    """

    settings = {
        key: getattr(module, key)
        for key in dir(module)
        if key.isupper() and not key.startswith("__")
    }
    return SimpleNamespace(**settings)


def build_image_transform():
    """Return the default image preprocessing pipeline used across scripts."""

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class AnkleDataset(Dataset):
    """Dataset that reads images based on the manifest and majority labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path,
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.class_to_idx = class_to_idx
        self.transform = transform or build_image_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.data_root / row["relative_path"]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        label = self.class_to_idx[row["majority_label"]]
        return tensor, label


def load_manifest_and_labels(
    manifest_path: Path,
    labels_path: Path,
    data_root: Path,
    class_to_idx: Optional[Dict[str, int]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load the merged manifest/labels and return usable rows with class mapping."""

    logger = logger or setup_logger(__name__)

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        raise FileNotFoundError(manifest_path)

    if not labels_path.exists():
        logger.error("Labels not found: %s", labels_path)
        raise FileNotFoundError(labels_path)

    manifest = pd.read_csv(manifest_path)
    labels = pd.read_csv(labels_path)
    merged = manifest.merge(labels, on="file_upload", how="inner")
    merged = merged.dropna(subset=["majority_label"])

    if class_to_idx is None:
        if merged.empty:
            raise ValueError("No labeled samples available after merging manifest and labels.")

        class_names = sorted(merged["majority_label"].unique())
        class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    else:
        merged = merged[merged["majority_label"].isin(class_to_idx.keys())]

    missing_files = [
        row["relative_path"]
        for _, row in merged.iterrows()
        if not (data_root / row["relative_path"]).exists()
    ]
    if missing_files:
        logger.warning("%d files listed in manifest are missing on disk", len(missing_files))
        merged = merged[~merged["relative_path"].isin(missing_files)]

    if merged.empty:
        raise ValueError("No usable samples found for the provided manifest and labels.")

    return merged, class_to_idx


def build_efficientnet_b0(num_classes: int):
    """Create an EfficientNet-B0 classifier head configured for the task."""

    base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = base.classifier[-1].in_features
    base.classifier[-1] = nn.Linear(in_features, num_classes)
    return base
