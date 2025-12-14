import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from utils import setup_logger

logger = setup_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# forras_azonosito_sorszam.ext  (e.g. sajat_abc-123_01.jpg)
FILENAME_PATTERN = re.compile(
    r"^(?P<source>sajat|internet)_(?P<identifier>[\w-]+)_(?P<sequence>\d{2})\.(?P<ext>[A-Za-z0-9]+)$"
)

# Label Studio-style prefix sometimes added in exports: "<hex>-<original_filename>"
HEX_PREFIX_RE = re.compile(r"^(?P<prefix>[0-9a-fA-F]{8,})-(?P<rest>.+)$")


def _parse_filename(filename: str):
    """Parse filenames that follow the forras_azonosito_sorszam schema."""
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None

    parsed = match.groupdict()
    parsed["sequence"] = int(parsed["sequence"])
    parsed["ext"] = parsed["ext"].lower()
    return parsed


def _discover_images(input_dir: Path) -> pd.DataFrame:
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
                "file_upload": path.name,  # basename on disk
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
        logger.info(
            "Discovered %d images across %d folders",
            len(manifest),
            manifest["folder"].nunique(),
        )

    return manifest


def _extract_label_from_annotation(annotation: dict) -> str | None:
    """Return the first choice label from a single annotation entry."""
    for result in annotation.get("result", []):
        value = result.get("value", {})
        choices = value.get("choices")
        if choices:
            return choices[0]
    return None


def _normalize_file_upload(value: str | None) -> str | None:
    """Make annotation `file_upload` comparable to manifest basenames.

    Handles:
    - full paths like "data/upload/.../name.jpg"
    - Windows backslashes
    - URL-encoded pieces (%20)
    - query strings (?v=...)
    - case differences
    - Label Studio hex prefix: "b581baad-sajat_resztvevo04_02.jpg" -> "sajat_resztvevo04_02.jpg"
    """
    if not value:
        return None

    s = str(value).strip()
    if not s:
        return None

    s = unquote(s)
    s = s.split("?", 1)[0]
    s = s.replace("\\", "/")
    s = s.rsplit("/", 1)[-1]  # basename

    # Strip LS hex prefix if present
    m = HEX_PREFIX_RE.match(s)
    if m:
        s = m.group("rest")

    s = s.strip().lower()
    return s or None


def _strip_accents(text: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def _normalize_label(label: str | None) -> str | None:
    """Map all label variants into exactly 3 classes.

    Output classes:
      - "Pronation"
      - "Neutral"
      - "Supination"
    """
    if not label:
        return None

    s = str(label).strip()
    if not s:
        return None

    s = _strip_accents(s).lower()
    s = re.sub(r"^\s*\d+\s*[_-]\s*", "", s)  # drop leading "1_", "2-", etc.
    s = re.sub(r"\s+", " ", s).strip()

    if "prona" in s:  # pronacio / pronation / pronalo / pronáló
        return "Pronation"
    if "neutr" in s:  # neutral / neutralis / neutrális
        return "Neutral"
    if "szup" in s or "supin" in s:  # szupinacio / supination / szupinalo
        return "Supination"

    return "UNKNOWN"


def _iter_labels_from_entry(entry: dict) -> list[str]:
    """Extract all found labels from an entry (may include multiple annotations)."""
    labels: list[str] = []
    for annotation in entry.get("annotations", []) or []:
        raw = _extract_label_from_annotation(annotation)
        norm = _normalize_label(raw)
        if norm:
            labels.append(norm)
    return labels


def _load_annotation_file(path: Path) -> list[dict]:
    """Load a JSON annotation export and return rows: (file_upload, label, annotator)."""
    logger.info("Loading annotations from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[dict] = []

    for entry in payload:
        raw_file_upload = entry.get("file_upload")
        file_upload = _normalize_file_upload(raw_file_upload)

        # fallback: some exports store the filename under entry["data"][...]
        if not file_upload:
            data = entry.get("data") or {}
            for key in ("image", "file_upload", "file", "filename", "img"):
                if key in data:
                    file_upload = _normalize_file_upload(data.get(key))
                    if file_upload:
                        break

        if not file_upload:
            continue

        labels = _iter_labels_from_entry(entry)
        if not labels:
            logger.warning("No label found for %s in %s", raw_file_upload, path.name)
            continue

        # choose first label found for this entry
        rows.append(
            {
                "file_upload": file_upload,
                "label": labels[0],
                "annotator": path.stem,
            }
        )

    return rows


def preprocess(input_dir: Path, output_dir: Path) -> None:
    """Parse all annotation JSON files and export consolidated labels."""
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Acquire and profile image inventory
    manifest = _discover_images(input_dir)
    if manifest.empty:
        logger.warning("Manifest is empty; nothing to do.")
        return

    # Normalize manifest join key to match normalized annotation keys
    manifest["file_upload"] = (
        manifest["file_upload"].astype(str).str.strip().str.lower()
    )

    manifest_path = output_dir / "image_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Saved image manifest to %s", manifest_path)

    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        logger.warning("No annotation files found under %s", input_dir)
        return

    records: list[dict] = []
    for json_path in json_files:
        try:
            records.extend(_load_annotation_file(json_path))
        except Exception as e:
            logger.warning("Failed to load annotations from %s: %s", json_path, e)

    if not records:
        logger.warning("No labels extracted from annotation files.")
        return

    annotations_df = pd.DataFrame(records)

    annotations_path = output_dir / "all_annotations.csv"
    annotations_df.to_csv(annotations_path, index=False)
    logger.info("Saved raw annotations to %s", annotations_path)

    # Majority vote per file_upload
    majority_rows: list[dict] = []
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

    # Merge labels onto manifest to compute coverage
    dataset_df = manifest.merge(majority_df, on="file_upload", how="left")
    labeled_count = int(dataset_df["majority_label"].notna().sum())
    unlabeled_count = int(len(dataset_df) - labeled_count)

    logger.info("Merge match rate: %d/%d", labeled_count, int(len(dataset_df)))

    # IMPORTANT: distribution must be computed from the merged dataset
    # so totals cannot exceed total_images.
    merged_labeled = dataset_df.loc[dataset_df["majority_label"].notna(), "majority_label"]

    class_distribution = (
        merged_labeled.value_counts().sort_index().to_dict()
    )

    summary = {
        "total_images": int(len(manifest)),
        "labeled_images": labeled_count,
        "unlabeled_images": unlabeled_count,
        "label_coverage_rate": float(labeled_count / len(manifest))
        if len(manifest)
        else 0.0,
        "class_distribution": class_distribution,
        "source_distribution": manifest["source"].value_counts().sort_index().to_dict(),
        "unknown_labels_present": bool((annotations_df["label"] == "UNKNOWN").any()),
    }

    summary_path = output_dir / "label_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Class distribution (3-class, merged): %s", class_distribution)
    logger.info("Summary saved to %s", summary_path)

    # Extra debugging: show a few unmatched examples if still bad
    if labeled_count == 0:
        ann_keys = set(majority_df["file_upload"].astype(str))
        man_keys = set(manifest["file_upload"].astype(str))
        only_in_ann = list(sorted(ann_keys - man_keys))[:10]
        only_in_man = list(sorted(man_keys - ann_keys))[:10]
        logger.warning("No merge matches. Example keys only in annotations: %s", only_in_ann)
        logger.warning("No merge matches. Example keys only in manifest: %s", only_in_man)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/data"),
        help="Input data root directory containing images and annotation JSON exports",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/app/output"),
        help="Output directory for processed labels",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.input, args.output)
