# Dataset Preparation

This directory contains the dataset preparation scripts used for the
PGA-GNN experiments.

The original datasets are not redistributed in this repository.
Users should obtain the datasets from their respective official sources
and run the provided preparation scripts.

## Datasets Used

| Dataset | Type | Samples Used |
|---------|------|--------------|
| TuSimple | Real-world | 1,000 |
| CULane | Real-world | 2,000 |
| BDD100K | Real-world | 3,000 |
| ACDC | Adverse-condition | 1,006 |
| CARLA | Synthetic | 3,000 |

**Total: 10,006 samples**

## Dataset Preparation

The preparation scripts select and organize the samples required for
the PGA-GNN experiments from the downloaded datasets.

The original datasets are not included in this repository due to their
respective distribution and licensing conditions.

### TuSimple

Obtain the TuSimple dataset from its official source and extract it
locally.

Run:

```bash
python datasets/prepare_tusimple.py \
    --input /path/to/tusimple \
    --output ./data/tusimple \
    --num-samples 1000

CULane

Obtain the CULane dataset from its official source and extract it
locally.

Run:

python datasets/prepare_culane.py \
    --input /path/to/culane \
    --output ./data/culane \
    --num-samples 2000

The script prepares the subset used in the PGA-GNN experiments.

BDD100K

Obtain the BDD100K dataset and the required lane annotations from their
official source.

Run:

python datasets/prepare_bdd100k.py \
    --input /path/to/bdd100k \
    --labels /path/to/bdd100k_labels.json \
    --output ./data/bdd100k \
    --num-samples 3000

The script prepares the 3,000 samples used in the PGA-GNN experiments.
ACDC

Obtain the ACDC dataset from its official source and extract it
locally.

Run:

python datasets/prepare_acdc.py \
    --input /path/to/acdc \
    --output ./data/acdc \
    --num-samples 1006

The script prepares the subset used in the PGA-GNN experiments.

CARLA

Synthetic driving scenes are generated using CARLA.

The provided script can be used to generate the synthetic samples
used as training augmentation:

python datasets/generate_carla_dataset.py
