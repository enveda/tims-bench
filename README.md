# TIMS-Bench: Towards community standards for benchmarking untargeted trapped ion mobility metabolomics tools and datasets

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14618408.svg)](TBD)) -->

This repository contains code and data described in detail in our paper (Rajkumar *et al.*, 2026).

## Table of Contents

* [Citation](#citation)
* [Requirements](#requirements)
* [Data](#data)
* [Repository structure](#repository-structure)
* [How to run](#how-to-run)
* [Notebook overview](#notebook-overview)
* [How to benchmark your own tool?](#how-to-benchmark-your-own-tool)

## Citation

If you have found our manuscript useful in your work, please consider citing:

> [Rajkumar, P, et al. (2026). TIMS-Bench: Towards community standards for benchmarking untargeted trapped ion mobility metabolomics tools and datasets](https://doi.org/10.64898/2026.05.23.724673).

## Requirements

* Python >= 3.13.5
* [UV](https://docs.astral.sh/uv/) for environment management
* A Linux machine is recommended for running the DreaMS embeddings (see note in [How to run](#how-to-run))

## Data

Datasets are publicly available and can be directly downloaded from **Zenodo** (DOI: TBD).
Unzip the downloaded files and place them under the `data/` directory as described in [Repository structure](#repository-structure).

### Repository structure

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
│   ├── groundtruth_dataset/    # MSV000098263, plant_spikein, nist_srm
│   ├── library_spectra/        # Reference library files (.parquet, .pq)
│   └── public_dataset/         # Eg. MSV000084402, MSV000090327, ..
├── figures/                    # Output figures for the manuscript
├── notebooks/                  # Analysis notebooks — run in numbered order
│   ├── 01a_harmonization.ipynb
│   ├── 01b_annotations.ipynb
│   ├── 01b_annotations.py
│   ├── 01c_dataset_qc.ipynb
│   ├── 02_tolerance_selection.ipynb
│   ├── 03_base_metrics.ipynb
│   ├── 03_groundtruth_metrics.ipynb
│   ├── 04a_reframe_based_metrics.ipynb
│   ├── 04b_reframe_css_evaluation.ipynb
│   ├── 04c_reframe_mirror_plots.ipynb
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

The `library spectra` folder looks like this:
```text
.
├── all_sorted_library_spectra.parquet
├── all_sorted_library_spectra.npz      # DreaMS embedding files (.npz)
├── nist_srm_spikein_lib.pq
├── plant_spikein_lib.pq
├── reframe_ms2s_with_ccs.parquet
└── reframe_spikein_lib.pq
```

## How to run

1. Clone the repository:

```bash
git clone https://github.com/enveda/benchmarking-untargeted-metabolomics-software.git
cd benchmarking-untargeted-metabolomics-software
```

1. Prepare the `data/` directory as described in the [Data and repository section](#data) section.

1. Install dependencies using UV:

```bash
uv sync
```

1. Run the notebooks in numbered order. Select the UV virtual environment as the kernel, or launch Jupyter directly:

```bash
uv run jupyter notebook
```

For standalone Python scripts (used only for running DreaMS matching):

```bash
uv run python notebooks/01b_annotations.py
```

> ***NOTE:*** The DreaMS embeddings and matching were run independently on a Linux server. Ensure you have the correct environment configuration as per their [GitHub](https://github.com/pluskal-lab/DreaMS).


## Notebook overview

* [01a_harmonization](notebooks/01a_harmonization.ipynb) - Code to read the raw output files of the tool and generate feature tables for analysis.
* [01b_annotations](notebooks/01b_annotations.ipynb) - Annotates harmonized feature tables using multiple similarity approaches (Spectral Entropy and Cosine) against a spectral library.
    * [01b_annotations.py](notebooks/01b_annotations.py) - Python script used to run DreaMS similarity search. Works only on Linux environment.
* [01c_dataset_qc](notebooks/01c_dataset_qc.ipynb) - Merges and performs quality control on feature tables from public and internal datasets across multiple tools with configurable similarity thresholds.
* [02_tolerance_selection](notebooks/02_tolerance_selection.ipynb) - Identifies optimal MS1 and MS2 tolerance parameters by testing varied tolerance values on the ReFRAME library dataset and comparing annotation results.
* [03a_base_metrics](notebooks/03a_base_metrics.ipynb) - Computes and visualizes base performance metrics across 10 public metabolomics datasets, comparing detection and annotation performance across analysis tools.
* [03b_groundtruth_metrics](notebooks/03b_groundtruth_metrics.ipynb) - Calculates and visualizes base metrics across three ground-truth datasets (ReFRAME, NIST SRM, plant spike-in) with radar plots comparing tool performance.
* [04a_reframe_based_metrics](notebooks/04a_reframe_based_metrics.ipynb) - Analyzes ReFRAME spike-in library performance using precision-recall curves, F1 scores, and CCS error distributions across different similarity thresholds and annotation methods.
* [04b_reframe_css_evaluation](notebooks/04b_reframe_css_evaluation.ipynb) - Evaluates CCS-based discrimination of structural isomers from the ReFRAME library using relative CCS differences and ion mobility separation thresholds.
* [04c_reframe_mirror_plots](notebooks/04c_reframe_mirror_plots.ipynb) - Generates spectral mirror plots comparing experimental MS2 spectra against ReFRAME library reference spectra to visually validate annotations.
* [05_nist_srm_based_metrics](notebooks/05_nist_srm_based_metrics.ipynb) - Computes precision-recall curves and R² distributions for the NIST SRM spike-in dataset to evaluate annotation accuracy and correlation with expected concentrations.
* [06a_plant_spikein_base_metrics](notebooks/06a_plant_spikein_base_metrics.ipynb) - Analyzes plant spike-in dataset performance using precision-recall metrics, R² distributions, and concentration-dependent recovery curves across analysis tools.
* [06b_plant_spikein_overlap](notebooks/06b_plant_spikein_overlap.ipynb) - Visualizes compound detection overlap across analysis tools at different spike-in concentrations using Venn diagrams and identifies compounds detected at all concentration levels.

## How to benchmark your own tool

The benchmarking pipeline is built around a harmonization step that converts any tool's raw export into a standard schema. To add a new tool, you need to: (1) write a harmonizer, (2) register it, (3) update the constants, and (4) run the pipeline.

### 1. Write a harmonizer

Create `benchmarking/harmonizer/yourtool.py` with a function that reads the tool's raw export and returns a `pandas.DataFrame` in the standard schema. The harmonized DataFrame must have the following fixed columns, followed by one column per sample containing peak area/intensity values:

| Column | Description |
|---|---|
| `FEATURE_ID` | Unique feature identifier |
| `M/Z` | Precursor m/z |
| `RT` | Retention time (minutes) |
| `ION_MOBILITY` | Ion mobility value (or `"N/A"` if unavailable) |
| `CCS` | Collision cross section (or `"N/A"` if unavailable) |
| `ADDUCT` | Adduct type (e.g. `[M+H]+`) |
| `MS/MS_ASSIGNED` | Boolean — whether an MS2 spectrum is assigned |
| `MS/MS_MZS` | MS2 fragment m/z values as a list |
| `MS/MS_INTENSITIES` | MS2 fragment intensities as a list (normalized to sum to 1) |

The existing harmonizers (`benchmarking/harmonizer/mzmine.py`, `metaboscape.py`, `msdial.py`) are the best reference for how to wrangle tool-specific export formats into this schema.

### 2. Register the harmonizer

In `benchmarking/harmonizer/harmonization.py`, import your function and add it to `HARMONIZATION_FUNCTIONS`:

```python
from benchmarking.harmonizer.yourtool import harmonize_yourtool

HARMONIZATION_FUNCTIONS = {
    "mzmine": harmonize_mzmine,
    "metaboscape": harmonize_metaboscape,
    "msdial-multi": harmonize_msdial_multi_sample,
    "msdial-multi-combined": harmonize_msdial_multi_sample_combined,
    "msdial-single": harmonize_msdial_single_sample,
    "yourtool": harmonize_yourtool,          # add this
}
```

Then add a corresponding `case` block in the `match input_tool_type:` statement in the same file to handle file discovery for your tool's expected input format (e.g., which file extensions to look for in the raw directory).

### 3. Update constants and tool detection

In `benchmarking/constants.py`, add your tool's display name and plot color:

```python
TOOL_NAMES = {
    ...,
    "yourtool": "YourTool",
}

TOOL_COLORS = {
    ...,
    "yourtool": "#AABBCC",
    "YourTool": "#AABBCC",
}
```

The tool detection logic in `benchmarking/metrics/base_metrics.py` (`load_all_feature_tables` and `get_base_metrics`) and `benchmarking/loader.py` infers tool identity from the harmonized filename. Each function contains an `if/elif` chain that checks for `"mzmine"`, `"msdial"`, and `"metaboscape"` in the filename. Add your tool there:

```python
if "mzmine" in file.lower():
    tool = "mzmine"
elif "msdial" in file.lower():
    tool = "msdial"
elif "metaboscape" in file.lower():
    tool = "metaboscape"
elif "yourtool" in file.lower():          # add this
    tool = "yourtool"
else:
    raise ValueError(f"Unknown tool type in file name: {file}")
```

### 4. Organize your data and run the pipeline

Place your tool's raw export files under the dataset directory following the same layout used by the existing tools:

```text
data/{dataset}/raw/yourtool/
    ├── features.csv      # quantification table
    └── spectra.mgf       # MS2 spectra
```

Then run the notebooks in order, passing `"yourtool"` as the tool type wherever `input_tool_type` is specified:

```python
harmonize(
    input_directory="data/{dataset}/raw/yourtool/",
    input_tool_type="yourtool",
    output_path="data/{dataset}/harmonized/{dataset}_yourtool_harmonized.parquet",
)
```

After harmonization, the rest of the pipeline (annotation in `01b`, QC in `01c`, and all downstream metric notebooks) works unchanged — it reads from the harmonized parquet files and treats your tool identically to the three tools evaluated in the paper.
