# -*- coding: utf-8 -*-

"""Codes to generate the DreaMS embeddings.

These were run indepdently as a script, but the code is kept here for reproducibility.
The generated embeddings are saved in the data folder and can be loaded for downstream
analyses without needing to run this code.

A linux machine with Python 3.10 or higher is required to run this code, as well as the DreaMS package.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from dreams.api import dreams_embeddings

from benchmarking.utils import filter_spectrum_peaks_in_df
from benchmarking.similarity.dreams_similarity_utils import (
    df_to_feature_id_dict,
    feature_id_dict_to_mgf,
)


def get_library_embeddings():
    """Build DreaMS embeddings for the library spectra and save them as .npz files."""
    library_data_filtered = pd.read_parquet(
        "../../data/library_spectra/all_sorted_library_spectra.pq"
    )

    library_data_filtered["FEATURE_ID"] = library_data_filtered.index.astype(str)
    library_feature_id_dict = df_to_feature_id_dict(
        library_data_filtered,
        "FEATURE_ID",
        ms2_column=None,
        mzs_column="normalized_mzs",
        intensities_column="normalized_intensities",
        precursor_mz_column="precursor_mz",
    )

    library_mgf_path = "../../data/library_spectra/all_sorted_library_spectra.mgf"
    if not os.path.exists(library_mgf_path):
        feature_id_dict_to_mgf(library_feature_id_dict, library_mgf_path)

    lib_embeddings_path = (
        "../../data/library_spectra/all_sorted_library_spectra_dreams_embeddings.npz"
    )

    if not os.path.exists(lib_embeddings_path):
        mgf_embeddings = dreams_embeddings(library_mgf_path, in_mem=True)
        np.savez(
            lib_embeddings_path,
            dreams_embeddings=mgf_embeddings,
        )

    print("Library embeddings generated and saved at:", lib_embeddings_path)


def get_dataset_embeddings(dir_list: list[str]):
    """Build DreaMS embeddings for the dataset spectra and save them as .npz files.

    Args:
        dir_list: List of paths to the input parquet files containing the harmonized dataset spectra.

    """
    output_mgfs = []

    for i in range(len(dir_list)):
        output_mgfs.append(
            dir_list[i]
            .replace("_harmonized.parquet", ".mgf")
            .replace("harmonized/", "mgf/")
        )

        if not os.path.exists(output_mgfs[i]):
            os.makedirs(os.path.dirname(output_mgfs[i]), exist_ok=True)

    output_embeddings = []

    for i in range(len(dir_list)):
        output_embeddings.append(
            dir_list[i]
            .replace("_harmonized.parquet", "_embeddings.npz")
            .replace("harmonized/", "embeddings/")
        )
        if not os.path.exists(output_embeddings[i]):
            os.makedirs(os.path.dirname(output_embeddings[i]), exist_ok=True)

    # Build mgf files
    for input_path, output_path in tqdm(
        zip(dir_list, output_mgfs), total=len(dir_list)
    ):
        harmonized_dataset_df = pd.read_parquet(input_path)
        harmonized_dataset_df_spectra_available = harmonized_dataset_df[
            harmonized_dataset_df["MS/MS_ASSIGNED"] == True
        ]
        harmonized_filtered = filter_spectrum_peaks_in_df(
            harmonized_dataset_df_spectra_available,
            "MS/MS_INTENSITIES",
            "MS/MS_MZS",
            10.0,
            0.01,
        )
        feature_id_dict = df_to_feature_id_dict(
            harmonized_filtered, ms2_column="MS/MS_ASSIGNED"
        )
        feature_id_dict_to_mgf(feature_id_dict, output_path)

    # Build embeddings from mgf files
    for mgf_file, embedding_file in zip(output_mgfs, output_embeddings):
        if os.path.exists(embedding_file):
            continue

        embeddings = dreams_embeddings(mgf_file, in_mem=True)
        np.savez(embedding_file, dreams_embeddings=embeddings)


def main():
    get_library_embeddings()

    # For ReFrame Drug Library dataset
    base_dir_internal = "../../data/groundtruth_dataset"
    internal_datasets = ["MSV000098263"]

    input_directories = []

    for dataset in internal_datasets:
        input_dir = f"{base_dir_internal}/{dataset}/harmonized/"
        if not os.path.exists(input_dir):
            print(f"Directory {input_dir} does not exist. Skipping dataset {dataset}.")
            continue

        for file in os.listdir(input_dir):
            input_directories.append(f"{input_dir}/{file}")

    get_dataset_embeddings(input_directories)
