"""Data preprocessing script for AnkleAlign annotations."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from exceptiongroup import catch

from utils import setup_logger

logger = setup_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FILENAME_PATTERN = re.compile(
    r"^(?P<source>sajat|internet)_(?P<identifier>[\w-]+)_(?P<sequence>\d{2})\.(?P<ext>[A-Za-z0-9]+)$"
)


def _parse_filename(filename: str):
    """Parse filenames that follow the forras_azonosito_sorszam schema.

    Parameters
    ----------
    filename : str
        The basename of the file (without directory components).

    Returns
    -------
    dict | None
        Parsed fields when the pattern matches, otherwise ``None``.
    """

    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None

    parsed = match.groupdict()
    parsed["sequence"] = int(parsed["sequence"])
    parsed["ext"] = parsed["ext"].lower()
    return parsed


def _discover_images(input_dir: Path):
    """Return a manifest of images that comply with the naming convention."""

    manifest_rows = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_dir():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parsed = _parse_filename(path.name)
        if not parsed:
            logger.warning("Filename does not match convention: %s", path)
            continue

        manifest_rows.append(
            {
                "file_upload": path.name,
                "relative_path": path.relative_to(input_dir).as_posix(),
                "folder": path.parent.relative_to(input_dir).as_posix(),
                "source": parsed["source"],
                "identifier": parsed["identifier"],
                "sequence": parsed["sequence"],
                "extension": parsed["ext"],
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    if manifest.empty:
        logger.warning("No images discovered under %s", input_dir)
    else:
        logger.info("Discovered %d images across %d folders", len(manifest), manifest["folder"].nunique())

    return manifest


def _extract_label_from_annotation(annotation):
    """Return the first choice label from a single annotation entry."""

    for result in annotation.get("result", []):
        value = result.get("value", {})
        choices = value.get("choices")
        if choices:
            return choices[0]
    return None


def _load_annotation_file(path: Path):
    """Load a JSON annotation export and yield (file_upload, label) pairs."""

    logger.info("Loading annotations from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    for entry in payload:
        file_upload = entry.get("file_upload")
        if not file_upload:
            continue

        label = None
        for annotation in entry.get("annotations", []):
            label = _extract_label_from_annotation(annotation)
            if label:
                break

        if label:
            yield {
                "file_upload": file_upload,
                "label": label,
                "annotator": path.stem,
            }
        else:
            logger.warning("No label found for %s in %s", file_upload, path.name)


def preprocess(input_dir: Path, output_dir: Path):
    """Parse all annotation JSON files and export consolidated labels."""

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Acquire and profile image inventory
    manifest = _discover_images(input_dir)
    manifest_path = output_dir / "image_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Saved image manifest to %s", manifest_path)

    json_files = sorted(input_dir.rglob("*.json"))

    if not json_files:
        logger.warning("No annotation files found under %s", input_dir)
        return

    records = []
    for json_path in json_files:
        try:
            records.extend(list(_load_annotation_file(json_path)))
        except Exception as e:
            logger.warning("Failed to load annotations from %s: %s", json_path, e)
    if not records:
        logger.warning("No labels extracted from annotation files.")
        return

    annotations_df = pd.DataFrame(records)
    annotations_path = output_dir / "all_annotations.csv"
    annotations_df.to_csv(annotations_path, index=False)
    logger.info("Saved raw annotations to %s", annotations_path)

    majority_rows = []
    for file_upload, group in annotations_df.groupby("file_upload"):
        label_counts = Counter(group["label"])
        majority_label, majority_votes = label_counts.most_common(1)[0]
        majority_rows.append(
            {
                "file_upload": file_upload,
                "majority_label": majority_label,
                "annotations_count": int(sum(label_counts.values())),
                "majority_votes": int(majority_votes),
            }
        )

    majority_df = pd.DataFrame(majority_rows).sort_values("file_upload")
    majority_path = output_dir / "majority_labels.csv"
    majority_df.to_csv(majority_path, index=False)
    logger.info("Saved majority labels to %s", majority_path)

    dataset_df = manifest.merge(majority_df, on="file_upload", how="left")
    labeled_count = int(dataset_df["majority_label"].notna().sum())
    unlabeled_count = int(len(dataset_df) - labeled_count)

    distribution = (
        majority_df["majority_label"].value_counts().sort_index().to_dict()
    )
    summary = {
        "total_images": int(len(manifest)),
        "labeled_images": labeled_count,
        "unlabeled_images": unlabeled_count,
        "label_coverage_rate": float(labeled_count / len(manifest)) if len(manifest) else 0.0,
        "class_distribution": distribution,
        "source_distribution": manifest["source"].value_counts().sort_index().to_dict(),
    }

    summary_path = output_dir / "label_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Class distribution: %s", distribution)
    logger.info("Summary saved to %s", summary_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/data"), help="Input data root directory containing annotations")
    parser.add_argument("--output", type=Path, default=Path("/app/output"), help="Output directory for processed labels")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.input, args.output)
