"""
PGA-GNN - CULane Dataset Preparation

Purpose:
    Prepare a deterministic subset of 2,000 CULane samples.

Input:
    Officially downloaded CULane dataset.

Output:
    pga_gnn_data/culane/
    ├── images/
    ├── annotations/
    └── summary.json
"""

from pathlib import Path
import argparse
import shutil


def find_image_files(dataset_root):
    """Find CULane image files."""
    image_files = sorted(dataset_root.rglob("*.jpg"))

    if not image_files:
        raise FileNotFoundError(
            "No JPG images were found in the CULane directory."
        )

    return image_files


def find_lane_annotation(image_file):
    """
    CULane lane annotations normally use the same relative path
    and a .lines.txt extension.

    Example:
        laneseg_label_w16/driver_23_30frame/06030820_0446.MP4/00000.jpg

    Corresponding annotation:
        .../00000.lines.txt
    """

    candidates = [
        image_file.with_suffix(".lines.txt"),
        Path(str(image_file).replace(".jpg", ".lines.txt"))
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def prepare_culane(dataset_root, output_root, num_samples=2000):

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    image_output = output_root / "images"
    annotation_output = output_root / "annotations"

    image_output.mkdir(parents=True, exist_ok=True)
    annotation_output.mkdir(parents=True, exist_ok=True)

    images = find_image_files(dataset_root)

    print(f"Total CULane images found: {len(images)}")

    # Deterministic ordering
    images = sorted(images)

    selected = images[:num_samples]

    if len(selected) < num_samples:
        raise RuntimeError(
            f"Only {len(selected)} images found, "
            f"but {num_samples} were requested."
        )

    copied = 0
    missing_annotations = 0

    manifest = []

    for idx, image_file in enumerate(selected):

        lane_file = find_lane_annotation(image_file)

        if lane_file is None:
            missing_annotations += 1
            print(
                f"WARNING: lane annotation not found for "
                f"{image_file}"
            )
            continue

        image_name = f"culane_{idx:05d}.jpg"
        annotation_name = f"culane_{idx:05d}.lines.txt"

        destination_image = image_output / image_name
        destination_annotation = annotation_output / annotation_name

        shutil.copy2(image_file, destination_image)
        shutil.copy2(lane_file, destination_annotation)

        manifest.append({
            "id": idx,
            "image": f"images/{image_name}",
            "annotation": f"annotations/{annotation_name}",
            "source": "CULane",
            "original_image": str(
                image_file.relative_to(dataset_root)
            )
        })

        copied += 1

    # Save manifest
    manifest_file = output_root / "manifest.json"

    import json

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    summary = {
        "dataset": "CULane",
        "requested_samples": num_samples,
        "selected_images": len(selected),
        "copied_images": copied,
        "missing_annotations": missing_annotations
    }

    with open(
        output_root / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\n----------------------------------------")
    print("CULane preparation completed")
    print("----------------------------------------")
    print(f"Requested samples     : {num_samples}")
    print(f"Selected images       : {len(selected)}")
    print(f"Copied images         : {copied}")
    print(f"Missing annotations   : {missing_annotations}")
    print(f"Output directory      : {output_root}")
    print("----------------------------------------")


def main():

    parser = argparse.ArgumentParser(
        description="Prepare CULane subset for PGA-GNN."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the downloaded CULane dataset."
    )

    parser.add_argument(
        "--output",
        default="pga_gnn_data/culane",
        help="Output directory."
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of samples to prepare."
    )

    args = parser.parse_args()

    prepare_culane(
        dataset_root=args.input,
        output_root=args.output,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
