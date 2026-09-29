"""
PGA-GNN - BDD100K Dataset Preparation

Purpose:
    Prepare a deterministic subset of 3,000 BDD100K images
    for the PGA-GNN experiments.

Input:
    Officially downloaded BDD100K dataset.

Expected structure:

    bdd100k/
    ├── images/
    │   └── 100k/
    └── labels/
        └── lane/
            └── polylines/

Output:

    pga_gnn_data/bdd100k/
    ├── images/
    ├── annotations/
    ├── manifest.json
    └── summary.json

Important:
    The script does not redistribute BDD100K.
    Users must obtain BDD100K from the official source.
"""

from pathlib import Path
import argparse
import json
import shutil


def find_images(dataset_root):
    """Find BDD100K images."""

    image_root = dataset_root / "images" / "100k"

    if not image_root.exists():
        raise FileNotFoundError(
            f"BDD100K image directory not found: {image_root}"
        )

    images = sorted(
        image_root.rglob("*.jpg")
    )

    if not images:
        raise FileNotFoundError(
            "No BDD100K JPG images were found."
        )

    return images


def find_lane_labels(dataset_root):
    """
    Locate BDD100K lane-marking labels.

    The script searches common BDD100K lane-label locations.
    """

    possible_directories = [
        dataset_root / "labels" / "lane" / "polylines",
        dataset_root / "labels" / "lane",
        dataset_root / "labels"
    ]

    for directory in possible_directories:

        if directory.exists():

            json_files = sorted(
                directory.rglob("*.json")
            )

            if json_files:
                return json_files

    return []


def load_lane_labels(label_files):

    labels = {}

    for label_file in label_files:

        try:

            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

        except Exception as exc:

            print(
                f"WARNING: could not read {label_file}: {exc}"
            )

            continue

        # BDD100K lane annotations may be stored as
        # a list of frame-level records.
        if isinstance(data, list):

            for record in data:

                if not isinstance(record, dict):
                    continue

                name = record.get("name")

                if name:
                    labels[name] = record

        elif isinstance(data, dict):

            name = data.get("name")

            if name:
                labels[name] = data

    return labels


def prepare_bdd100k(
    dataset_root,
    output_root,
    num_samples=3000
):

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    image_output = output_root / "images"
    annotation_output = output_root / "annotations"

    image_output.mkdir(
        parents=True,
        exist_ok=True
    )

    annotation_output.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # 1. Find images
    # --------------------------------------------------

    images = find_images(dataset_root)

    print(
        f"Total BDD100K images found: {len(images)}"
    )

    images = sorted(images)

    selected = images[:num_samples]

    if len(selected) < num_samples:

        raise RuntimeError(
            f"Only {len(selected)} images were found, "
            f"but {num_samples} were requested."
        )

    # --------------------------------------------------
    # 2. Find lane labels
    # --------------------------------------------------

    label_files = find_lane_labels(
        dataset_root
    )

    print(
        f"Lane annotation files found: "
        f"{len(label_files)}"
    )

    labels = load_lane_labels(
        label_files
    )

    print(
        f"Indexed lane-label records: "
        f"{len(labels)}"
    )

    # --------------------------------------------------
    # 3. Copy selected images
    # --------------------------------------------------

    manifest = []

    copied = 0
    missing_labels = 0

    for idx, image_file in enumerate(selected):

        image_name = (
            f"bdd100k_{idx:05d}.jpg"
        )

        destination_image = (
            image_output / image_name
        )

        shutil.copy2(
            image_file,
            destination_image
        )

        # Original BDD100K filename
        original_name = image_file.name

        lane_record = labels.get(
            original_name
        )

        annotation_name = (
            f"bdd100k_{idx:05d}.json"
        )

        annotation_path = (
            annotation_output /
            annotation_name
        )

        if lane_record is not None:

            with open(
                annotation_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    lane_record,
                    f,
                    indent=4
                )

        else:

            missing_labels += 1

            # Keep an explicit record rather than
            # silently fabricating an annotation.
            with open(
                annotation_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    {
                        "name": original_name,
                        "lane_annotation_available": False
                    },
                    f,
                    indent=4
                )

        manifest.append(
            {
                "id": idx,
                "image": (
                    f"images/{image_name}"
                ),
                "annotation": (
                    f"annotations/"
                    f"{annotation_name}"
                ),
                "source": "BDD100K",
                "original_image": original_name,
                "lane_annotation_available":
                    lane_record is not None
            }
        )

        copied += 1

    # --------------------------------------------------
    # 4. Save manifest
    # --------------------------------------------------

    with open(
        output_root / "manifest.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=4
        )

    # --------------------------------------------------
    # 5. Save summary
    # --------------------------------------------------

    summary = {

        "dataset": "BDD100K",

        "requested_samples":
            num_samples,

        "selected_images":
            len(selected),

        "copied_images":
            copied,

        "images_with_lane_annotations":
            copied - missing_labels,

        "images_without_lane_annotations":
            missing_labels,

        "note":
            "BDD100K must be obtained from "
            "the official dataset source."
    }

    with open(
        output_root / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print()
    print("----------------------------------------")
    print("BDD100K preparation completed")
    print("----------------------------------------")
    print(
        f"Requested samples       : "
        f"{num_samples}"
    )
    print(
        f"Selected images         : "
        f"{len(selected)}"
    )
    print(
        f"Copied images           : "
        f"{copied}"
    )
    print(
        f"Images with lane labels : "
        f"{copied - missing_labels}"
    )
    print(
        f"Missing lane labels     : "
        f"{missing_labels}"
    )
    print(
        f"Output directory        : "
        f"{output_root}"
    )
    print("----------------------------------------")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare BDD100K subset "
            "for PGA-GNN."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to the downloaded "
            "BDD100K dataset."
        )
    )

    parser.add_argument(
        "--output",
        default=(
            "pga_gnn_data/bdd100k"
        ),
        help="Output directory."
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=3000,
        help=(
            "Number of images to prepare."
        )
    )

    args = parser.parse_args()

    prepare_bdd100k(
        dataset_root=args.input,
        output_root=args.output,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
