"""
PGA-GNN - ACDC Dataset Preparation

Purpose:
    Prepare a deterministic subset of 1,006 ACDC images
    for the PGA-GNN experiments.

ACDC contains driving scenes captured under adverse
conditions including fog, nighttime, rain and snow.

Input:
    Officially downloaded ACDC dataset.

Output:
    pga_gnn_data/acdc/
    ├── images/
    ├── annotations/
    ├── manifest.json
    └── summary.json

Important:
    The ACDC dataset is not redistributed by this repository.
    Users must obtain the dataset from the official ACDC source.
"""

from pathlib import Path
import argparse
import json
import shutil


SUPPORTED_CONDITIONS = {
    "fog",
    "night",
    "rain",
    "snow"
}


def find_acdc_images(dataset_root):
    """
    Locate ACDC RGB images.

    Expected ACDC structure:

        root/
        └── rgb_anon/
            ├── fog/
            ├── night/
            ├── rain/
            └── snow/
    """

    rgb_root = dataset_root / "rgb_anon"

    if not rgb_root.exists():
        raise FileNotFoundError(
            f"ACDC rgb_anon directory not found: {rgb_root}"
        )

    image_files = []

    for condition in sorted(SUPPORTED_CONDITIONS):

        condition_dir = rgb_root / condition

        if not condition_dir.exists():
            print(
                f"WARNING: condition directory not found: "
                f"{condition_dir}"
            )
            continue

        files = sorted(
            condition_dir.rglob("*.png")
        )

        image_files.extend(files)

    if not image_files:
        raise FileNotFoundError(
            "No ACDC RGB images were found."
        )

    return sorted(image_files)


def find_annotation(image_file, dataset_root):
    """
    Locate the corresponding ACDC semantic annotation.

    ACDC ground-truth annotations are stored under gt/
    and use the corresponding sequence/frame naming.
    """

    try:
        relative_path = image_file.relative_to(
            dataset_root / "rgb_anon"
        )
    except ValueError:
        return None

    # Replace rgb_anon with gt
    candidate = (
        dataset_root
        / "gt"
        / relative_path
    )

    # Typical semantic annotation suffix
    candidates = [
        candidate.with_name(
            candidate.stem + "_labelIds.png"
        ),
        candidate.with_name(
            candidate.stem + "_labelTrainIds.png"
        ),
        candidate.with_name(
            candidate.stem + "_labelColor.png"
        )
    ]

    for annotation in candidates:

        if annotation.exists():
            return annotation

    return None


def get_condition(image_file):
    """
    Determine the adverse condition from the path.
    """

    for part in image_file.parts:

        if part in SUPPORTED_CONDITIONS:
            return part

    return "unknown"


def prepare_acdc(
    dataset_root,
    output_root,
    num_samples=1006
):

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    image_output = (
        output_root / "images"
    )

    annotation_output = (
        output_root / "annotations"
    )

    image_output.mkdir(
        parents=True,
        exist_ok=True
    )

    annotation_output.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # 1. Locate ACDC images
    # --------------------------------------------------

    images = find_acdc_images(
        dataset_root
    )

    print(
        f"Total ACDC images found: {len(images)}"
    )

    # --------------------------------------------------
    # 2. Deterministic ordering
    # --------------------------------------------------

    images = sorted(images)

    selected = images[:num_samples]

    if len(selected) < num_samples:

        raise RuntimeError(
            f"Only {len(selected)} ACDC images "
            f"were found, but {num_samples} "
            f"were requested."
        )

    # --------------------------------------------------
    # 3. Copy selected images and annotations
    # --------------------------------------------------

    manifest = []

    copied = 0
    missing_annotations = 0

    condition_counts = {
        "fog": 0,
        "night": 0,
        "rain": 0,
        "snow": 0
    }

    for idx, image_file in enumerate(selected):

        condition = get_condition(
            image_file
        )

        image_name = (
            f"acdc_{idx:05d}.png"
        )

        destination_image = (
            image_output / image_name
        )

        shutil.copy2(
            image_file,
            destination_image
        )

        annotation_file = find_annotation(
            image_file,
            dataset_root
        )

        annotation_name = (
            f"acdc_{idx:05d}_label.png"
        )

        destination_annotation = (
            annotation_output /
            annotation_name
        )

        annotation_available = (
            annotation_file is not None
        )

        if annotation_available:

            shutil.copy2(
                annotation_file,
                destination_annotation
            )

        else:

            missing_annotations += 1

        if condition in condition_counts:
            condition_counts[condition] += 1

        manifest.append(
            {
                "id": idx,
                "image":
                    f"images/{image_name}",
                "annotation":
                    (
                        f"annotations/"
                        f"{annotation_name}"
                        if annotation_available
                        else None
                    ),
                "source": "ACDC",
                "condition": condition,
                "original_image":
                    str(
                        image_file.relative_to(
                            dataset_root
                        )
                    ),
                "annotation_available":
                    annotation_available
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
    # 5. Save dataset summary
    # --------------------------------------------------

    summary = {

        "dataset": "ACDC",

        "requested_samples":
            num_samples,

        "selected_images":
            len(selected),

        "copied_images":
            copied,

        "images_with_annotations":
            copied - missing_annotations,

        "images_without_annotations":
            missing_annotations,

        "condition_distribution":
            condition_counts,

        "note":
            (
                "This repository prepares a project-specific "
                "subset of ACDC. The original ACDC dataset "
                "is not redistributed."
            )
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

    # --------------------------------------------------
    # 6. Print summary
    # --------------------------------------------------

    print()
    print("----------------------------------------")
    print("ACDC preparation completed")
    print("----------------------------------------")
    print(
        f"Requested samples : "
        f"{num_samples}"
    )
    print(
        f"Selected images   : "
        f"{len(selected)}"
    )
    print(
        f"Copied images     : "
        f"{copied}"
    )
    print(
        f"Missing labels    : "
        f"{missing_annotations}"
    )
    print()
    print("Condition distribution:")

    for condition, count in (
        condition_counts.items()
    ):

        print(
            f"  {condition:>5}: {count}"
        )

    print()
    print(
        f"Output directory  : "
        f"{output_root}"
    )

    print("----------------------------------------")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare an ACDC subset "
            "for PGA-GNN."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to the downloaded "
            "ACDC dataset."
        )
    )

    parser.add_argument(
        "--output",
        default=(
            "pga_gnn_data/acdc"
        ),
        help="Output directory."
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1006,
        help=(
            "Number of ACDC images "
            "to prepare."
        )
    )

    args = parser.parse_args()

    prepare_acdc(
        dataset_root=args.input,
        output_root=args.output,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
