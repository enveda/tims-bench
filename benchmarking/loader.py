# -*- coding: utf-8 -*-

"""Loader utils code for loading plot related data."""

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from benchmarking.constants import (
    INCHIKEY_COLUMN,
    LIBRARY_ID_COLUMN,
    LIBRARY_INTENSITIES_COLUMN,
    LIBRARY_MZS_COLUMN,
    LIBRARY_PRECURSOR_MZ_COLUMN,
    METHOD_COLUMN,
    SCORE_COLUMN,
    TOOL_NAMES,
)
from benchmarking.metrics.base_metrics import get_euler_data


def get_upset_plot_data(
    input_dataset_directories,
    similarity_threshold=0.7,
    annotation_subfolder="annotated_spectral_entropy",
):
    # Aggregate InChIKeys across all datasets per tool
    combined_sets = defaultdict(set)

    for dataset_path in tqdm(input_dataset_directories, desc="Processing datasets"):
        labels, sets = get_euler_data(
            dataset_path,
            similarity_threshold=similarity_threshold,
            annotation_subfolder=annotation_subfolder,
        )

        dataset_dict = dict(zip(labels, sets))

        # update the combined sets with the current dataset's sets
        for label, s in dataset_dict.items():
            combined_sets[label] |= s

    if not combined_sets:
        print("No data to plot")
        return None

    return combined_sets


def get_intensity_plot_data(
    ground_truth_data: pd.DataFrame,
    merged_feature_tables: dict,
    score_threshold: float = 0.7,
    score_method: str = "spectral_entropy",
):
    """Get intensity distributions for all features, true positives, and false positives for each tool.

    Parameters
    ----------
    ground_truth_data : pd.DataFrame
        DataFrame containing the ground truth annotations, including the "inchikey_2d" column.
    merged_feature_tables : dict
        Dictionary where keys are tool names and values are DataFrames containing the merged feature tables for each tool.

    Returns
    -------
    list of tuples
        A list of tuples, where each tuple contains the tool name, an array of all feature intensities, and an array of true positive feature intensities.
    """
    true_inchikeys = set(
        ground_truth_data["inchikey_2d"].apply(lambda x: x.split("-")[0]).to_list()
    )

    intensity_list = []
    tp_intensity_list = []
    tool_list = []
    for tool, df in merged_feature_tables.items():
        df_annotated = df[df[METHOD_COLUMN] == score_method]
        df_annotated = df_annotated[
            (df_annotated[SCORE_COLUMN].notna())
            & (df_annotated[SCORE_COLUMN] >= score_threshold)
        ]
        df_annotated[INCHIKEY_COLUMN] = df_annotated[INCHIKEY_COLUMN].apply(
            lambda x: x.split("-")[0]
        )
        df_true_positives = df_annotated[
            df_annotated[INCHIKEY_COLUMN].isin(true_inchikeys)
        ]
        df_false_positives = df_annotated[
            ~df_annotated[INCHIKEY_COLUMN].isin(true_inchikeys)
        ]

        cols_to_drop = [
            c
            for c in [
                "FEATURE_ID",
                "M/Z",
                "RT",
                "ION_MOBILITY",
                "CCS",
                "ADDUCT",
                "MS/MS_ASSIGNED",
                "MS/MS_MZS",
                "MS/MS_INTENSITIES",
                LIBRARY_ID_COLUMN,
                INCHIKEY_COLUMN,
                SCORE_COLUMN,
                METHOD_COLUMN,
                LIBRARY_PRECURSOR_MZ_COLUMN,
                LIBRARY_MZS_COLUMN,
                LIBRARY_INTENSITIES_COLUMN,
                "CLIQUE_ID",
            ]
            if c in df.columns
        ]

        df = df.drop(columns=cols_to_drop)  # Original df
        df_true_positives = df_true_positives.drop(columns=cols_to_drop)
        df_false_positives = df_false_positives.drop(columns=cols_to_drop)

        all_intensities = df.max(axis=1).values
        all_intensities = all_intensities[all_intensities > 0]
        true_positive_intensities = df_true_positives.max(axis=1).values
        true_positive_intensities = true_positive_intensities[
            true_positive_intensities > 0
        ]
        false_positive_intensities = df_false_positives.max(axis=1).values
        false_positive_intensities = false_positive_intensities[
            false_positive_intensities > 0
        ]
        intensity_list.append(all_intensities)
        tp_intensity_list.append(true_positive_intensities)
        tool_list.append(tool)

    return list(zip(tool_list, intensity_list, tp_intensity_list))


def get_confident_tps(
    ground_truth_library_data: pd.DataFrame,
    merged_feature_tables: dict,
    spectral_score_threshold: float = 0.7,
):
    """
    Get confident true positives for each tool based on a spectral score threshold of 0.7.
    Parameters
    ----------
    ground_truth_library_data : pd.DataFrame
        DataFrame containing the ground truth library annotations, including the "inchikey_2d" column.
    merged_feature_tables : dict
        Dictionary where keys are tool names and values are DataFrames containing the merged feature tables for each
        tool.
    spectral_score_threshold : float, optional
        Minimum spectral score to consider a feature as a confident true positive, by default 0.7

    Returns
    -------
    dict
        Dictionary where keys are tool names and values are sets of confident true positive InChIKeys
    """
    true_inchikeys = set(
        ground_truth_library_data["inchikey_2d"]
        .apply(lambda x: x.split("-")[0])
        .to_list()
    )

    tool_inchikeys = {}

    for tool, df in zip(merged_feature_tables.keys(), merged_feature_tables.values()):
        df_annotated = df[
            (df[SCORE_COLUMN].notna()) & (df[SCORE_COLUMN] >= spectral_score_threshold)
        ]
        df_annotated[INCHIKEY_COLUMN] = df_annotated[INCHIKEY_COLUMN].apply(
            lambda x: x.split("-")[0]
        )
        df_true_positives = df_annotated[
            df_annotated[INCHIKEY_COLUMN].isin(true_inchikeys)
        ]
        tool_inchikeys[TOOL_NAMES[tool]] = set(
            df_true_positives[INCHIKEY_COLUMN].to_list()
        )

    return tool_inchikeys


def get_ccs_error_distributions(
    merged_feature_tables,
    ground_truth_ccs_data,
    score_threshold=0.7,
    output_type="relative",
    ccs_column: str = "CCS",
    ground_truth_ccs_column: str = "ccs",
    match_precursor: bool = True,
    ground_truth_precursor_column: str = "precursor_mz",
    feature_precursor_column: str = "M/Z",
):
    """Compute CCS error distributions for each tool.

    Parameters
    ----------
    merged_feature_tables : dict
        Dictionary where keys are tool names and values are DataFrames containing the merged feature tables for each tool.
    ground_truth_ccs_data : pd.DataFrame
        DataFrame containing the ground truth CCS values, including the "inchikey_2d" column and the column specified by ground_truth_ccs_column.
    score_threshold : float, optional
        Minimum spectral score to consider a feature as annotated, by default 0.7
    output_type : str, optional
        Whether to return "relative" or "absolute" CCS errors, by default "relative"
    ccs_column : str, optional
        Name of the column in the feature tables that contains the predicted CCS values, by default "CCS"
    ground_truth_ccs_column : str, optional
        Name of the column in the ground truth data that contains the true CCS values, by default "ccs"
    match_precursor : bool, optional
        Whether to match features to ground truth CCS values based on precursor m/z, by default True
    ground_truth_precursor_column : str, optional
        Name of the column in the ground truth data that contains the precursor m/z values, required if match_precursor is True, by default "precursor_mz"
    feature_precursor_column : str, optional
        Name of the column in the feature tables that contains the precursor m/z values, required if match_precursor is True, by default "M/Z"

    Returns
    -------
    dict
        Dictionary where keys are tool names and values are arrays of CCS errors for the annotated features of that tool.
    """
    if output_type not in ["relative", "absolute"]:
        raise ValueError("output_type must be 'relative' or 'absolute'")

    true_inchikeys = set(
        ground_truth_ccs_data["inchikey_2d"].apply(lambda x: x.split("-")[0]).to_list()
    )
    ccs_df = ground_truth_ccs_data[["inchikey_2d", ground_truth_ccs_column]].copy()
    ccs_df["inchikey_2d"] = ccs_df["inchikey_2d"].apply(lambda x: x.split("-")[0])

    if match_precursor:
        if ground_truth_precursor_column not in ground_truth_ccs_data.columns:
            raise ValueError(
                f"ground_truth_precursor_column '{ground_truth_precursor_column}' not found in ground_truth_ccs_data"
            )
        ccs_df = ground_truth_ccs_data[
            ["inchikey_2d", ground_truth_ccs_column, ground_truth_precursor_column]
        ].copy()
        ccs_df["inchikey_2d"] = ccs_df["inchikey_2d"].apply(lambda x: x.split("-")[0])
    else:
        # Collapse ccs values taking mean across inchikey
        ccs_df = (
            ccs_df.groupby("inchikey_2d")[ground_truth_ccs_column].mean().reset_index()
        )

    if not match_precursor:
        true_ccs_values = ccs_df.set_index(ccs_df["inchikey_2d"])[
            ground_truth_ccs_column
        ].to_dict()

    ccs_error_distributions = {}
    for tool, df in zip(merged_feature_tables.keys(), merged_feature_tables.values()):

        df_annotated = df[
            (df[SCORE_COLUMN].notna()) & (df[SCORE_COLUMN] >= score_threshold)
        ]
        df_annotated[INCHIKEY_COLUMN] = df_annotated[INCHIKEY_COLUMN].apply(
            lambda x: x.split("-")[0]
        )
        df_true_positives = df_annotated[
            df_annotated[INCHIKEY_COLUMN].isin(true_inchikeys)
        ]

        if match_precursor:
            if feature_precursor_column not in df_true_positives.columns:
                raise ValueError(
                    f"feature_precursor_column '{feature_precursor_column}' not found in feature table"
                )

            predicted_ccs_values = []#df_true_positives[ccs_column].to_numpy()
            actual_ccs_values = []

            for _, row in df_true_positives.iterrows():
                inchikey = row[INCHIKEY_COLUMN]
                precursor = row[feature_precursor_column]
                gt_subset = ccs_df[ccs_df["inchikey_2d"] == inchikey]
                if gt_subset.empty:
                    continue
                diffs = (gt_subset[ground_truth_precursor_column] - precursor).abs()
                best_idx = diffs.idxmin()
                actual_ccs_values.append(
                    gt_subset.loc[best_idx, ground_truth_ccs_column]
                )
                predicted_ccs_values.append(row[ccs_column])

            actual_ccs_values = np.array(actual_ccs_values)
            predicted_ccs_values = np.array(predicted_ccs_values)
            #predicted_ccs_values = predicted_ccs_values[: len(actual_ccs_values)]
        else:
            predicted_ccs_values = df_true_positives[ccs_column].to_numpy()
            actual_ccs_values = (
                df_true_positives[INCHIKEY_COLUMN].map(true_ccs_values).to_numpy()
            )

        ccs_errors = predicted_ccs_values - actual_ccs_values

        if output_type == "relative":
            ccs_errors = (
                (predicted_ccs_values - actual_ccs_values) / actual_ccs_values * 100
            )
        ccs_error_distributions[tool] = ccs_errors

    return ccs_error_distributions
