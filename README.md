# PGA-GNN

## Physics-Informed, Reliability-Aware Graph Network for Adverse-Condition Lane Detection

This repository provides the dataset preparation resources associated with the
PGA-GNN framework for adverse-condition lane detection.

PGA-GNN represents lane observations as graph nodes and spatial relationships
as graph edges. A reliability-aware propagation mechanism regulates the
contribution of uncertain lane observations, while physics-informed constraints
encourage geometrically consistent lane predictions.

## Dataset

The experiments use four real-world driving datasets together with synthetic
data generated using CARLA.

| Dataset | Type | Samples Used |
|---------|------|--------------|
| TuSimple | Real-world | 1,000 |
| CULane | Real-world | 2,000 |
| BDD100K | Real-world | 3,000 |
| ACDC | Adverse-condition | 1,006 |
| CARLA | Synthetic | 3,000 |
| **Total** | | **10,006** |

The original datasets are not redistributed in this repository. Users should
obtain the datasets from their respective official sources and comply with
their licenses and terms of use.

## Dataset Preparation

The repository provides scripts for selecting and organizing the samples used
in the PGA-GNN experiments.

The scripts operate on datasets downloaded independently by the user. The
selected samples are determined by the corresponding preparation scripts and
the specified sample counts.

### Repository Structure

```text
PGA-GNN/
│
├── datasets/
│   ├── README.md
│   ├── prepare_tusimple.py
│   ├── prepare_culane.py
│   ├── prepare_bdd100k.py
│   ├── prepare_acdc.py
│   └── generate_carla_dataset.py
│
├── README.md
├── requirements.txt
├── configs/
├── src/
└── scripts/


1. TuSimple

Obtain the TuSimple dataset from its official source and extract it locally.

The preparation script selects the 1,000 samples used in the PGA-GNN
experiments.

Example:

python datasets/prepare_tusimple.py \
    --input /path/to/tusimple \
    --output ./data/tusimple \
    --num-samples 1000

2. CULane

Obtain the CULane dataset from its official source and extract it locally.

The preparation script selects the 2,000 samples used in the PGA-GNN
experiments.

Example:

python datasets/prepare_culane.py \
    --input /path/to/culane \
    --output ./data/culane \
    --num-samples 2000
3. BDD100K

Obtain the BDD100K dataset and the required lane annotations from their
official source.

The preparation script selects the 3,000 samples used in the PGA-GNN
experiments.

Example:

python datasets/prepare_bdd100k.py \
    --input /path/to/bdd100k \
    --labels /path/to/bdd100k_labels.json \
    --output ./data/bdd100k \
    --num-samples 3000
4. ACDC

Obtain the ACDC dataset from its official source and extract it locally.

The preparation script selects the 1,006 samples used in the PGA-GNN
experiments.

Example:

python datasets/prepare_acdc.py \
    --input /path/to/acdc \
    --output ./data/acdc \
    --num-samples 1006
5. CARLA Synthetic Data

Synthetic driving scenes are generated using the CARLA simulator.

The repository provides a dataset-generation script for producing synthetic
samples used as part of the PGA-GNN dataset.

Example:

python datasets/generate_carla_dataset.py
Important Notes
The original TuSimple, CULane, BDD100K, and ACDC datasets are not included
in this repository.
Users must obtain these datasets from their respective official sources.
The preparation scripts select the required subsets from the locally
downloaded datasets.
The sample counts reported in this repository correspond to the datasets
used in the PGA-GNN experiments.
Dataset redistribution is avoided in accordance with the respective
dataset distribution and licensing conditions.
The implementation and complete experimental pipeline may be released
separately.
