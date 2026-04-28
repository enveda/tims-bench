# -*- coding: utf-8 -*-

"""Cosine tolerance study utils."""

import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np

from ms_entropy import clean_spectrum
from benchmarking.utils import to_typed_bank, filter_spectrum_peaks_in_df
from benchmarking.similarity.cosine_similarity import flash_cosine_search
from benchmarking.metrics.base_metrics import compute_max_precision_recall_data
import benchmarking.constants as cons


def generate_ms_tolerance_data(
    input_feature_tables: list,
    output_feature_dirs: list,
    ms_tolerances: list,
    ms_tolerance_type: str = "ms1",
) -> None:
    """Generate data for MS tolerance plot.

    Parameters:
    input_feature_tables : list
        List of paths to input feature tables (parquet files).
    output_feature_dirs : list
        List of directories where output feature tables (parquet files) will be saved.
    ms_tolerances : list
        List of MS tolerance values to test (  e.g., [10, 20, 50] for ppm or Da).
    ms_tolerance_type : str, optional
        Type of MS tolerance to use ("ms1" or "ms2"), by default "ms1".

    Returns:
    None
    """
    # Load the library
    library_data_filtered = pd.read_parquet(
        "../data/library_spectra/all_sorted_library_spectra.parquet"
    )

    # Prepare library for matching
    reference_spectra = library_data_filtered.apply(
        lambda x: np.column_stack((x["normalized_mzs"], x["normalized_intensities"])),
        axis=1,
    ).to_numpy()

    reference_spectra = [clean_spectrum(spectrum) for spectrum in reference_spectra]
    reference_spectra = np.array(reference_spectra, dtype=object)

    reference_precursors = library_data_filtered["precursor_mz"].to_numpy(
        dtype=np.float64
    )
    reference_inchis = library_data_filtered["inchikey_2d"].to_numpy(dtype=object)
    reference_ids = library_data_filtered.index.to_numpy(dtype=object)

    reference_spectra_list = to_typed_bank(reference_spectra)
    reference_precursors = np.asarray(
        reference_precursors, dtype=np.float64, order="C"
    ).reshape(-1)
    reference_inchis = np.asarray(reference_inchis, dtype=object, order="C").reshape(-1)
    reference_ids = np.asarray(reference_ids, dtype=int, order="C").reshape(-1)

    references_mzs = library_data_filtered["normalized_mzs"].to_numpy(dtype=object)
    references_intensities = library_data_filtered["normalized_intensities"].to_numpy(
        dtype=object
    )

    print("Library loaded and processed.")

    total = len(input_feature_tables) * len(ms_tolerances)
    counter = 0

    for input_feature_table, output_feature_table in zip(
        input_feature_tables, output_feature_dirs
    ):
        harmonized_feature_table = pd.read_parquet(input_feature_table)
        harmonized_feature_table_spectra_available = harmonized_feature_table[
            harmonized_feature_table["MS/MS_ASSIGNED"] == True
        ]

        harmonized_feature_table_filtered = filter_spectrum_peaks_in_df(
            harmonized_feature_table_spectra_available,
            "MS/MS_INTENSITIES",
            "MS/MS_MZS",
            10.0,
            0.01,
        )

        query_spectra = harmonized_feature_table_filtered.apply(
            lambda x: np.column_stack((x["MS/MS_MZS"], x["MS/MS_INTENSITIES"])), axis=1
        ).to_numpy()

        query_bank = to_typed_bank(query_spectra)

        query_precursors = harmonized_feature_table_filtered["M/Z"].to_numpy()
        query_precursors_reshaped = np.asarray(
            query_precursors, dtype=np.float32, order="C"
        ).reshape(-1)

        for ms_tolerance in tqdm(
            ms_tolerances,
            desc=f"Processing MS tolerances for {input_feature_table.split('/')[-1]}",
        ):
            tool_name = input_feature_table.split("/")[-1].split("_")[1]
            output_feature_table_varied = (
                "/".join(output_feature_table.split("/")[:-1])
                + f"/{tool_name}_{ms_tolerance_type}_{ms_tolerance}"
                + ".parquet"
            )

            if os.path.isfile(output_feature_table_varied):
                counter += 1
                continue

            if ms_tolerance_type == "ms1":
                ms2_da_tolerance = 20.0
                ms1_da_tolerance = None
                ms1_ppm_tolerance = ms_tolerance
            elif ms_tolerance_type == "ms2":
                ms2_da_tolerance = ms_tolerance
                ms1_da_tolerance = None
                ms1_ppm_tolerance = 20

            scores, idx = flash_cosine_search(
                query_bank,
                query_precursors_reshaped,
                reference_spectra_list,
                reference_precursors,
                ms1_da_tolerance=ms1_da_tolerance,
                ms1_ppm_tolerance=ms1_ppm_tolerance,
                ms2_da_tolerance=ms2_da_tolerance,
                min_matched_peaks=3,
                return_argmax=True,
            )

            feature_ids = harmonized_feature_table_filtered["FEATURE_ID"].to_numpy(
                dtype=str
            )

            library_ids = reference_ids[idx]
            library_ids = np.where(idx == -1, -1, library_ids)
            library_ids = library_ids.astype(int)
            output = pd.DataFrame(
                {
                    cons.FEATURE_ID_COLUMN: feature_ids,
                    cons.LIBRARY_ID_COLUMN: library_ids,
                    cons.INCHIKEY_COLUMN: reference_inchis[idx],
                    cons.SCORE_COLUMN: scores,
                    cons.METHOD_COLUMN: f"cosine_similarity_{ms_tolerance_type}_{ms_tolerance}",
                    cons.LIBRARY_PRECURSOR_MZ_COLUMN: reference_precursors[idx],
                    cons.LIBRARY_MZS_COLUMN: references_mzs[idx],
                    cons.LIBRARY_INTENSITIES_COLUMN: references_intensities[idx],
                }
            )

            os.makedirs(os.path.dirname(output_feature_table_varied), exist_ok=True)
            output.to_parquet(output_feature_table_varied, index=False, engine="pyarrow")

    print(
        f"Processing completed. {counter}/{total} outputs already existed and were skipped."
    )


def generate_cosine_variance_outputs(
    varied_cosine_output_path: str,
    input_feature_tables: list,
    ms1_ppm_tolerance: np.array,
    ms2_ppm_tolerances: np.array,
    ground_truth_library_path: str,
):
    """Generate outputs for cosine variance analysis.

    Parameters:
    -----------
    varied_cosine_output_path : str
        Path to save the output JSON file containing cosine variance results.
    input_feature_tables : list
        List of paths to input feature tables (parquet files) that have been annotated with varied cosine scores.
    ms1_ppm_tolerance : np.array
        Array of MS1 ppm tolerance values that were tested.
    ms2_ppm_tolerances : np.array
        Array of MS2 ppm tolerance values that were tested.
    ground_truth_library_path : str
        Path to the ground truth library data (parquet file) to use for computing metrics.

    Returns:
    --------
    None
    """
    # Load existing output if it exists, otherwise initialize an empty dictionary
    if os.path.exists(varied_cosine_output_path):
        varied_cosine_output = json.load(open(varied_cosine_output_path, "r"))
    else:
        varied_cosine_output = {}

    # Load ground truth library data
    ground_truth_library_data = pd.read_parquet(ground_truth_library_path)

    # Iterate through input feature tables and compute metrics for each MS tolerance, storing results in varied_cosine_output
    for harmonized_file_path in input_feature_tables:
        for ms_tol_type in ["ms1", "ms2"]:
            for ms_tol in tqdm(
                (ms1_ppm_tolerance if ms_tol_type == "ms1" else ms2_ppm_tolerances),
                desc=f"Processing {ms_tol_type} tolerances for {harmonized_file_path.split('/')[-1]}",
            ):
                tool_name = harmonized_file_path.split("/")[-1].split("_")[1]

                # Check if output exists in varied_cosine_output, if so, skip computation
                if (
                    tool_name in varied_cosine_output
                    and ms_tol_type in varied_cosine_output[tool_name]
                    and str(ms_tol) in varied_cosine_output[tool_name][ms_tol_type]
                ):
                    continue

                output_feature_table_varied = (
                    "/".join(harmonized_file_path.split("/")[:-1]).replace(
                        "harmonized", "annotated_cosine_varied"
                    )
                    + f"/{tool_name}_{ms_tol_type}_{ms_tol}"
                    + ".parquet"
                )

                feature_table = pd.read_parquet(harmonized_file_path)
                annotations = pd.read_parquet(output_feature_table_varied)
                annotations["FEATURE_ID"] = annotations["FEATURE_ID"].astype(
                    feature_table["FEATURE_ID"].dtype
                )

                updated_feature_table = feature_table.merge(
                    annotations, on="FEATURE_ID", how="left"
                )

                # Compute metrics
                best_metrics = compute_max_precision_recall_data(
                    updated_feature_table, ground_truth_library_data
                )

                varied_cosine_output.setdefault(tool_name, {}).setdefault(
                    ms_tol_type, {}
                )[str(ms_tol)] = best_metrics

    with open(varied_cosine_output_path, "w") as f:
        json.dump(varied_cosine_output, f, indent=2, ensure_ascii=False)
