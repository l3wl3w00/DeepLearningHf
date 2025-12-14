"""Fine-tune an image classifier for the AnkleAlign dataset."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

import config
from utils import load_config, setup_logger

logger = setup_logger(__name__)


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    early_stopping_patience: int
    data_root: Path
    manifest_path: Path
    labels_path: Path
    model_output: Path
    val_split: float = 0.2
    num_workers: int = 2


class AnkleDataset(Dataset):
    """Dataset that reads images based on the manifest and majority labels."""

    def __init__(self, df: pd.DataFrame, data_root: Path, class_to_idx: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.class_to_idx = class_to_idx
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.data_root / row["relative_path"]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        label = self.class_to_idx[row["majority_label"]]
        return tensor, label


def _load_dataset(manifest_path: Path, labels_path: Path, data_root: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
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

    if merged.empty:
        raise ValueError("No labeled samples available after merging manifest and labels.")

    class_names = sorted(merged["majority_label"].unique())
    class_to_idx = {label: idx for idx, label in enumerate(class_names)}

    missing_files = [
        row["relative_path"]
        for _, row in merged.iterrows()
        if not (data_root / row["relative_path"]).exists()
    ]
    if missing_files:
        logger.warning("%d files listed in manifest are missing on disk", len(missing_files))
        merged = merged[~merged["relative_path"].isin(missing_files)]

    if merged.empty:
        raise ValueError("All referenced files are missing from disk after filtering.")

    return merged, class_to_idx


def _build_model(num_classes: int):
    base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = base.classifier[-1].in_features
    base.classifier[-1] = nn.Linear(in_features, num_classes)
    return base


def train(cfg: TrainingConfig):
    logger.info("Starting training with config: %s", cfg)
    df, class_to_idx = _load_dataset(cfg.manifest_path, cfg.labels_path, cfg.data_root)

    train_df, val_df = train_test_split(
        df,
        test_size=cfg.val_split,
        stratify=df["majority_label"],
        random_state=42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_to_idx)
    model = _build_model(num_classes).to(device)

    train_loader = DataLoader(
        AnkleDataset(train_df, cfg.data_root, class_to_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        AnkleDataset(val_df, cfg.data_root, class_to_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total else 0.0

        logger.info(
            "Epoch %d/%d - train_loss: %.4f - val_loss: %.4f - val_acc: %.4f",
            epoch,
            cfg.epochs,
            epoch_loss,
            val_loss,
            val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    if best_state is None:
        best_state = model.state_dict()

    cfg.model_output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state,
        "class_to_idx": class_to_idx,
        "architecture": "efficientnet_b0",
    }
    torch.save(checkpoint, cfg.model_output)
    logger.info("Saved model checkpoint to %s", cfg.model_output)


def parse_args():
    defaults = load_config(config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(defaults.DATA_DIR))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/app/output/image_manifest.csv"),
        help="Path to the image manifest generated during preprocessing",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("/app/output/majority_labels.csv"),
        help="Path to the majority labels CSV",
    )
    parser.add_argument("--model-output", type=Path, default=Path(defaults.MODEL_SAVE_PATH))
    parser.add_argument("--epochs", type=int, default=defaults.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=defaults.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=defaults.LEARNING_RATE)
    parser.add_argument("--early-stopping", type=int, default=defaults.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping,
        data_root=args.data_root,
        manifest_path=args.manifest,
        labels_path=args.labels,
        model_output=args.model_output,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )
    train(cfg)
