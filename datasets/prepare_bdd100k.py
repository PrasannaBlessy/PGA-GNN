"""
PGA-GNN - BDD100K Dataset Preparation

Purpose
-------
Prepare a deterministic subset of BDD100K lane-marking data for
the PGA-GNN experiments.

The script:
1. Reads BDD100K frame-level JSON annotations.
2. Selects images containing lane annotations.
3. Selects a deterministic subset of 3,000 images.
4. Copies the selected images.
5. Extracts lane poly2d annotations.
6. Stores the extracted annotations in JSON format.
7. Creates a manifest and summary file.

BDD100K uses the Scalabel-compatible annotation format.
Lane markings are represented using:
    category = "lane"
    poly2d = lane geometry

The original BDD100K dataset is NOT redistributed by this repository.
Users must obtain BDD100K separately and comply with its license.
"""

from pathlib import Path
import argparse
import json
import random
import shutil
from typing import Dict, List, Any


DEFAULT_NUM_SAMPLES = 3000
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic BDD100K subset for PGA-GNN."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Root directory containing the downloaded BDD100K dataset."
    )

    parser.add_argument(
        "--labels",
        required=True,
        help="Path to the BDD100K lane-label JSON file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the prepared PGA-GNN subset."
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of images to select. Default: 3000."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic selection. Default: 42."
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_frames(data: Any) -> List[Dict[str, Any]]:
    """
    Extract frame records from BDD100K JSON.

    BDD100K label files are commonly represented as a list of frame
    dictionaries. This function also supports a dictionary containing
    a 'frames' field.
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "frames" in data and isinstance(data["frames"], list):
            return data["frames"]

        # Some exported files may contain one frame directly.
        if "name" in data:
            return [data]

    raise ValueError(
        "Unsupported BDD100K annotation structure. "
        "Expected a list of frames or a dictionary containing 'frames'."
    )


def get_lane_labels(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract lane labels from a BDD100K frame.

    A valid lane annotation has:
        category == 'lane'
        poly2d containing lane geometry
    """

    labels = frame.get("labels", [])

    if labels is None:
        return []

    lanes = []

    for label in labels:
        if not isinstance(label, dict):
            continue

        if label.get("category") != "lane":
            continue

        poly2d = label.get("poly2d")

        if not poly2d:
            continue

        lanes.append(label)

    return lanes


def find_image(root: Path, image_name: str) -> Path:
    """
    Locate an image inside the BDD100K directory.

    The search supports the common BDD100K organization where images
    are stored under train/val folders or another nested directory.
    """

    direct_candidates = [
        root / "images" / "100k" / "train" / image_name,
        root / "images" / "100k" / "val" / image_name,
        root / "images" / "100k" / image_name,
        root / "images" / image_name,
    ]

    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    # Fallback recursive search.
    matches = list(root.rglob(image_name))

    if matches:
        return matches[0]

    return None


def clean_lane_annotation(label: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep the fields required for the PGA-GNN dataset.

    The original BDD100K label is not copied in full. Only the
    lane geometry and relevant attributes are retained.
    """

    result = {
        "id": label.get("id"),
        "category": label.get("category"),
        "poly2d": label.get("poly2d", []),
        "attributes": label.get("attributes", {}),
    }

    return result


def prepare_dataset(
    input_root: Path,
    label_file: Path,
    output_root: Path,
    num_samples: int,
    seed: int,
):
    """Prepare the BDD100K subset."""

    print("=" * 70)
    print("PGA-GNN BDD100K Dataset Preparation")
    print("=" * 70)

    print(f"Input dataset : {input_root}")
    print(f"Label file    : {label_file}")
    print(f"Output        : {output_root}")
    print(f"Samples       : {num_samples}")
    print(f"Seed          : {seed}")
    print()

    if not input_root.exists():
        raise FileNotFoundError(
            f"BDD100K input directory does not exist: {input_root}"
        )

    if not label_file.exists():
        raise FileNotFoundError(
            f"BDD100K label file does not exist: {label_file}"
        )

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------

    image_output = output_root / "images"
    annotation_output = output_root / "annotations"

    image_output.mkdir(parents=True, exist_ok=True)
    annotation_output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load annotations
    # ------------------------------------------------------------------

    print("Loading BDD100K annotations...")

    data = load_json(label_file)
    frames = extract_frames(data)

    print(f"Total annotated frames: {len(frames)}")

    # ------------------------------------------------------------------
    # Find frames containing lane annotations
    # ------------------------------------------------------------------

    candidates = []

    for frame in frames:

        image_name = frame.get("name")

        if not image_name:
            continue

        lanes = get_lane_labels(frame)

        if not lanes:
            continue

        candidates.append(
            {
                "name": image_name,
                "frame": frame,
                "lanes": lanes,
            }
        )

    print(f"Frames containing lane annotations: {len(candidates)}")

    if len(candidates) < num_samples:
        raise RuntimeError(
            f"Only {len(candidates)} lane-labelled images were found, "
            f"but {num_samples} were requested."
        )

    # ------------------------------------------------------------------
    # Deterministic selection
    # ------------------------------------------------------------------

    rng = random.Random(seed)

    rng.shuffle(candidates)

    selected = candidates[:num_samples]

    # Sort after selection so the output order is stable and easy to inspect.
    selected.sort(key=lambda item: item["name"])

    # ------------------------------------------------------------------
    # Process selected images
    # ------------------------------------------------------------------

    manifest = []

    total_lanes = 0
    missing_images = []

    for index, item in enumerate(selected, start=1):

        image_name = item["name"]
        frame = item["frame"]
        lanes = item["lanes"]

        source_image = find_image(input_root, image_name)

        if source_image is None:
            missing_images.append(image_name)
            continue

        destination_image = image_output / image_name

        shutil.copy2(
            source_image,
            destination_image
        )

        cleaned_lanes = [
            clean_lane_annotation(label)
            for label in lanes
        ]

        annotation_record = {
            "image": image_name,
            "source": "BDD100K",
            "lanes": cleaned_lanes,
            "frame_attributes": frame.get(
                "attributes",
                {}
            ),
        }

        annotation_file = annotation_output / (
            Path(image_name).stem + ".json"
        )

        with annotation_file.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                annotation_record,
                f,
                indent=2
            )

        total_lanes += len(cleaned_lanes)

        manifest.append(
            {
                "index": index,
                "image": image_name,
                "image_path": str(
                    Path("images") / image_name
                ),
                "annotation_path": str(
                    Path("annotations")
                    / annotation_file.name
                ),
                "num_lanes": len(cleaned_lanes),
            }
        )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    manifest_path = output_root / "manifest.json"

    with manifest_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary = {
        "dataset": "BDD100K",
        "task": "lane_detection",
        "requested_samples": num_samples,
        "selected_samples": len(selected),
        "successfully_copied": len(manifest),
        "missing_images": len(missing_images),
        "total_lane_instances": total_lanes,
        "random_seed": seed,
        "annotation_format": "Scalabel-compatible poly2d",
        "lane_category": "lane",
    }

    summary_path = output_root / "summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ------------------------------------------------------------------
    # Missing images log
    # ------------------------------------------------------------------

    if missing_images:

        missing_path = output_root / "missing_images.txt"

        with missing_path.open(
            "w",
            encoding="utf-8"
        ) as f:

            for name in missing_images:
                f.write(name + "\n")

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("BDD100K preparation completed")
    print("=" * 70)

    print(f"Requested images       : {num_samples}")
    print(f"Selected images        : {len(selected)}")
    print(f"Copied images          : {len(manifest)}")
    print(f"Missing images         : {len(missing_images)}")
    print(f"Lane instances         : {total_lanes}")
    print()
    print(f"Output directory       : {output_root}")
    print(f"Manifest               : {manifest_path}")
    print(f"Summary                : {summary_path}")
    print("=" * 70)


def main():

    args = parse_args()

    prepare_dataset(
        input_root=Path(args.input),
        label_file=Path(args.labels),
        output_root=Path(args.output),
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
