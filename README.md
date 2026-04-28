# TIMS-Bench: Towards community standards for benchmarking untargeted trapped ion mobility metabolomics tools and datasets

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14618408.svg)](TBD)) -->

This repository contains code and data described in detail in our paper (Rajkumar *et al.*, 2026).
DOI: TDB

## Table of Contents

* [Citation](#citation)
* [Requirements](#requirements)
* [Data](#data)
* [Repository structure](#repository-structure)
* [How to run](#how-to-run)

### Citation

If you have found our manuscript useful in your work, please consider citing:

> [Rajkumar, P, et al. (2026). TIMS-Bench: Towards community standards for benchmarking untargeted trapped ion mobility metabolomics tools and datasets](TBD).

## Requirements

* Python >= 3.13.5
* [UV](https://docs.astral.sh/uv/) for environment management
* A Linux machine is recommended for running the DreaMS embeddings (see note in [How to run](#how-to-run))

## Data

Datasets are publicly available and can be directly downloaded from **Zenodo** (DOI: TBD).
Unzip the downloaded files and place them under the `data/` directory as described in [Repository structure](#repository-structure).

## Repository structure

```text
.
├── benchmarking/               # Python package with shared utilities
│   ├── harmonizer/             # Tool-specific parsers (MetaboScape, MS-DIAL, MZmine)
│   ├── metrics/                # Benchmarking metrics (base metrics, clique analysis)
│   ├── similarity/             # Spectral similarity methods (cosine, entropy, DreaMS)
│   ├── constants.py
│   ├── loader.py
│   ├── plots.py
│   └── utils.py
├── data/                       # Downloaded from Zenodo (not tracked by git)
│   ├── groundtruth_dataset/    # MSV000098263, plant_spikein
│   ├── library_spectra/        # Reference library files (.parquet, .pq)
│   └── public_dataset/         # MSV000084402, MSV000090327
├── figures/                    # Output figures from the manuscript
├── notebooks/                  # Analysis notebooks — run in numbered order
│   ├── 01a_harmonization.ipynb
│   ├── 01b_annotations.ipynb
│   ├── 01b_annotations.py
│   ├── 01c_dataset_qc.ipynb
│   ├── 02_tolerance_selection.ipynb
│   ├── 03_base_metrics.ipynb
│   ├── 03_groundtruth_metrics.ipynb
│   ├── 04_reframe_based_metrics.ipynb
│   ├── 04b_reframe_css_evaluation.ipynb
│   ├── 05_nist_srm_based_metrics.ipynb
│   ├── 06_plant_spikein_base_metrics.ipynb
│   └── 06_plant_spikein_overlap.ipynb
└── pyproject.toml
```

Each dataset folder under `groundtruth_dataset/` and `public_dataset/` follows the same layout:

```text
{dataset}/
├── raw/                        # Original tool exports (MetaboScape, MS-DIAL, MZmine)
├── harmonized/                 # Unified parquet files per tool
├── annotated_cosine_similarity/
├── annotated_spectral_entropy/
├── annotated_dreams_similarity/
└── embeddings/                 # DreaMS embedding files (.npz)
```

## How to run

1. Clone the repository:

```bash
git clone https://github.com/enveda/benchmarking-untargeted-metabolomics-software.git
cd benchmarking-untargeted-metabolomics-software
```

1. Prepare the `data/` directory as described in the [Data](#data) section.

1. Install dependencies using UV:

```bash
uv sync
```

1. Run the notebooks in numbered order. Select the UV virtual environment as the kernel, or launch Jupyter directly:

```bash
uv run jupyter notebook
```

For standalone Python scripts:

```bash
uv run python notebooks/01b_annotations.py
```

> ***NOTE:*** The DreaMS embeddings and matching were run independently on a Linux server. Ensure you have the correct environment configuration as per their [GitHub](https://github.com/pluskal-lab/DreaMS).
