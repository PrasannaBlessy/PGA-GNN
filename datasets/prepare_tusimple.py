"""
PGA-GNN - TuSimple Dataset Preparation

Purpose:
    Prepare a deterministic subset of 1,000 TuSimple samples for PGA-GNN.

Input:
    Officially downloaded TuSimple dataset.

Expected structure:
    tusimple/
    ├── clips/
    ├── label_data_0313.json
    ├── label_data_0531.json
    └── label_data_0601.json

Output:
    pga_gnn_data/tusimple/
    ├── images/
    └── annotations.jsonl

Each annotation record contains:
    - image path
    - lane coordinates
    - h_samples
    - source dataset
"""

from pathlib import Path
import argparse
import json
import shutil


def find_annotation_files(dataset_root):
    """Find TuSimple label_data*.json files."""
    files = sorted(dataset_root.glob("label_data*.json"))

    if not files:
        raise FileNotFoundError(
            "No TuSimple label_data*.json files found in the dataset directory."
        )

    return files


def load_samples(annotation_files):
    """Read all TuSimple annotation records."""
    samples = []

    for annotation_file in annotation_files:
        print(f"Reading: {annotation_file}")

        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)
                samples.append(record)

    return samples


def prepare_tusimple(dataset_root, output_root, num_samples=1000):
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    image_output = output_root / "images"
    image_output.mkdir(parents=True, exist_ok=True)

    annotation_files = find_annotation_files(dataset_root)
    samples = load_samples(annotation_files)

    print(f"\nTotal annotation records found: {len(samples)}")

    # Deterministic ordering
    samples = sorted(
        samples,
        key=lambda x: x.get("raw_file", "")
    )

    # Select the required number of samples
    selected = samples[:num_samples]

    if len(selected) < num_samples:
        raise RuntimeError(
            f"Only {len(selected)} valid samples were found, "
            f"but {num_samples} were requested."
        )

    output_records = []

    copied = 0
    missing = 0

    for idx, sample in enumerate(selected):

        raw_file = sample.get("raw_file")

        if not raw_file:
            continue

        source_image = dataset_root / raw_file

        if not source_image.exists():
            print(f"WARNING: image not found: {source_image}")
            missing += 1
            continue

        # Give every image a deterministic sequential filename.
        output_name = f"tusimple_{idx:05d}{source_image.suffix}"

        destination = image_output / output_name

        shutil.copy2(source_image, destination)

        output_records.append(
            {
                "id": idx,
                "image": f"images/{output_name}",
                "source": "TuSimple",
                "raw_file": raw_file,
                "lanes": sample.get("lanes", []),
                "h_samples": sample.get("h_samples", [])
            }
        )

        copied += 1

    annotation_output = output_root / "annotations.jsonl"

    with open(annotation_output, "w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record) + "\n")

    # Dataset summary
    summary = {
        "dataset": "TuSimple",
        "requested_samples": num_samples,
        "selected_samples": len(selected),
        "copied_images": copied,
        "missing_images": missing,
        "annotation_file": "annotations.jsonl"
    }

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n----------------------------------------")
    print("TuSimple preparation completed")
    print("----------------------------------------")
    print(f"Requested samples : {num_samples}")
    print(f"Selected samples  : {len(selected)}")
    print(f"Copied images     : {copied}")
    print(f"Missing images    : {missing}")
    print(f"Output directory  : {output_root}")
    print("----------------------------------------")


def main():

    parser = argparse.ArgumentParser(
        description="Prepare TuSimple subset for PGA-GNN."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the downloaded TuSimple dataset."
    )

    parser.add_argument(
        "--output",
        default="pga_gnn_data/tusimple",
        help="Output directory."
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to prepare."
    )

    args = parser.parse_args()

    prepare_tusimple(
        dataset_root=args.input,
        output_root=args.output,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
