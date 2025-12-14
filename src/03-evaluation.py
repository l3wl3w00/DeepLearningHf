"""Simple evaluation for the trained AnkleAlign classifier.

This script loads the saved checkpoint from the training step and compares
its accuracy against a simple baseline that always predicts the most common
label in the evaluation set.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from utils import (
    AnkleDataset,
    build_efficientnet_b0,
    load_config,
    load_manifest_and_labels,
    setup_logger,
)

logger = setup_logger(__name__)


@dataclass
class EvalConfig:
    data_root: Path
    manifest_path: Path
    labels_path: Path
    checkpoint_path: Path
    batch_size: int = 64
    num_workers: int = 0


def _compute_baseline_accuracy(df: pd.DataFrame) -> float:
    label_counts = Counter(df["majority_label"])
    most_common_label, most_common_count = label_counts.most_common(1)[0]
    logger.info("Baseline always predicts '%s' (%d occurrences)", most_common_label, most_common_count)
    return most_common_count / len(df)


def _evaluate_model(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total if total else 0.0


def evaluate(cfg: EvalConfig):
    logger.info("Starting evaluation with checkpoint %s", cfg.checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]

    df, class_to_idx = load_manifest_and_labels(
        cfg.manifest_path,
        cfg.labels_path,
        cfg.data_root,
        class_to_idx=class_to_idx,
        logger=logger,
    )

    dataset = AnkleDataset(df, cfg.data_root, class_to_idx)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    baseline_accuracy = _compute_baseline_accuracy(df)

    model = build_efficientnet_b0(len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model_accuracy = _evaluate_model(model, dataloader, device)

    logger.info("Baseline accuracy: %.4f", baseline_accuracy)
    logger.info("Model accuracy: %.4f", model_accuracy)
    return {"baseline_accuracy": baseline_accuracy, "model_accuracy": model_accuracy}


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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(defaults.MODEL_SAVE_PATH),
        help="Model checkpoint created by the training script",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = EvalConfig(
        data_root=args.data_root,
        manifest_path=args.manifest,
        labels_path=args.labels,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    evaluate(cfg)
