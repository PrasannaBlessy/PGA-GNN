# PGA-GNN

## Physics-Informed, Reliability-Aware Graph Network for Adverse-Condition Lane Detection

This repository contains the dataset preparation scripts and experimental resources for PGA-GNN, a graph-based lane-detection framework designed for adverse visual conditions.

PGA-GNN represents lane observations as graph nodes and spatial relationships as graph edges. A reliability-aware propagation mechanism regulates the contribution of uncertain lane observations, while physics-informed constraints encourage geometrically consistent lane predictions.

---

## Dataset

The experiments use four real-world driving datasets together with synthetic data generated using CARLA.

| Dataset | Type | Samples used |
|---|---|---:|
| TuSimple | Real-world | 1,000 |
| CULane | Real-world | 2,000 |
| BDD100K | Real-world | 3,000 |
| ACDC | Real-world adverse conditions | 1,006 |
| CARLA | Synthetic | 3,000 |
| **Total** | | **10,006** |

The original datasets are not redistributed in this repository. Users should obtain them from their respective official sources and comply with their licenses and terms of use.

---

## Repository Structure

```text
PGA-GNN/
│
├── datasets/
│   ├── README.md
│   ├── prepare_tusimple.py
│   ├── prepare_culane.py
│   ├── prepare_bdd100k.py
│   ├── prepare_acdc.py
│   ├── prepare_bdd100k.py
│   └── generate_carla_dataset.py
│
├── README.md
├── requirements.txt
├── configs/
├── src/
└── scripts/
