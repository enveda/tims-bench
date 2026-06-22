# -*- coding: utf-8 -*-

"""Functions to calcuate the base metrics."""

import os
from collections import defaultdict
from tqdm import tqdm
from typing import Dict
import numpy as np
import pandas as pd
from itertools import product as itertools_product
from scipy import stats

from benchmarking.constants import INCHIKEY_COLUMN, SCORE_COLUMN, TRANSFORM_SETTINGS
from benchmarking.utils import apply_rclr_transform, apply_clr_transform


def retrieve_sample_number(path_to_dataset: str) -> int:
    """Helper function to retrieve the number of samples in a dataset based on the harmonized feature table.

    Parameters:
    -----------
    path_to_dataset: str
        Path to the dataset directory containing the 'harmonized'

    Returns:
    --------
    int: Number of samples in the dataset
    """
    harmonized_path = os.path.join(path_to_dataset, "harmonized")
    columns_to_drop = [
        "FEATURE_ID",
        "M/Z",
        "RT",
        "ION_MOBILITY",
        "CCS",
        "ADDUCT",
        "MS/MS_ASSIGNED",
        "MS/MS_MZS",
        "MS/MS_INTENSITIES",
    ]

    for file in os.listdir(harmonized_path):
        if "harmonized" in file:
            feature_table = pd.read_parquet(os.path.join(harmonized_path, file))
            feature_table = feature_table.drop(columns=columns_to_drop, errors="ignore")
            n_cols = len(feature_table.columns)
            return n_cols


def load_all_feature_tables(
    input_directories: list,
    annotation_subfolder: str = "annotated_spectral_entropy",
) -> Dict[str, pd.DataFrame]:
    """
    Helper function to load and merge feature tables across all datasets.
    Returns a dict mapping tool -> DataFrame (one combined dataset).

    Parameters:
    -----------
    input_directories: list of str
        List of dataset directories to process. Each directory should contain an 'annotated_spectral_entropy' subfolder with the relevant files.
    annotation_subfolder: str
        Name of the subfolder within each dataset directory that contains the annotated and harmonized files.
        Default is 'annotated_spectral_entropy'.
    Returns:
    --------
    Dict[str, pd.DataFrame]
        A dictionary where keys are tool names (e.g., 'mzmine', 'msdial', 'metaboscape') and values are
        DataFrames, each representing a combined dataset.
    """
    all_tables = defaultdict(list)

    for dataset_path in tqdm(input_directories):
        annotation_path = os.path.join(dataset_path, annotation_subfolder)

        # Define paths and tools
        annotation_paths = []
        harmonized_paths = []
        tools = []
        for file in os.listdir(annotation_path):

            # skip .DS_Store files that can appear on MacOS
            if file.startswith(".DS_Store"):
                continue

            annotation_paths.append(os.path.join(annotation_path, file))
            harmonized_filepath = os.path.join(annotation_path, file).replace(
                annotation_subfolder, "harmonized"
            )
            harmonized_paths.append(harmonized_filepath)
            if "mzmine" in file.lower():
                tool = "mzmine"
            elif "msdial" in file.lower():
                tool = "msdial"
            elif "metaboscape" in file.lower():
                tool = "metaboscape"
            else:
                raise ValueError(f"Unknown tool type in file name: {file}")
            tools.append(tool)

        for annotated_path, harmonized_path, tool_type in zip(
            annotation_paths, harmonized_paths, tools
        ):
            if not os.path.exists(annotated_path) or not os.path.exists(
                harmonized_path
            ):
                continue

            annotations = pd.read_parquet(annotated_path)
            feature_table = pd.read_parquet(harmonized_path)

            annotations["FEATURE_ID"] = annotations["FEATURE_ID"].astype(
                feature_table["FEATURE_ID"].dtype
            )
            feature_table = feature_table.merge(
                annotations, on="FEATURE_ID", how="left"
            )

            if tool_type not in all_tables:
                all_tables[tool_type] = []

            if len(input_directories) > 1:
                all_tables[tool_type].append(feature_table)
            else:
                all_tables[tool_type] = feature_table

    updated_tables = defaultdict(pd.DataFrame)
    if len(input_directories) > 1:
        print("Merging feature tables across datasets...")
        for tool_type in all_tables:
            updated_tables[tool_type] = pd.concat(
                all_tables[tool_type], ignore_index=True
            )
    else:
        updated_tables = all_tables

    return updated_tables


def compute_base_metrics(
    path_to_dataset: str,
    similarity_threshold=0.7,
    annotation_subfolder="annotated_spectral_entropy",
):
    """
    Compute base metrics for a given dataset with entropy annotations.

    Parameters:
    -----------
    path_to_dataset: str
        Path to the dataset directory containing the 'annotated_spectral_entropy' subfolder with the relevant files.
    similarity_threshold: float
        Threshold for considering a feature as annotated based on the similarity score. Default is 0.7.
    annotation_subfolder: str
        Name of the subfolder within the dataset directory that contains the annotated and harmonized files.
        Default is 'annotated_spectral_entropy'.

    Returns:
    --------
    dict
        A dictionary containing the computed metrics for each tool, including MS1 Count, MS2 Count, and Annotated Count.

    """

    annotation_path = os.path.join(path_to_dataset, annotation_subfolder)

    # Define paths and tools
    annotation_paths = []
    harmonized_paths = []
    tools = []
    for file in os.listdir(annotation_path):
        annotation_paths.append(os.path.join(annotation_path, file))
        harmonized_filepath = os.path.join(annotation_path, file).replace(
            annotation_subfolder, "harmonized"
        )
        harmonized_paths.append(harmonized_filepath)
        if "mzmine" in file.lower():
            tool = "mzmine"
        elif "msdial" in file.lower():
            tool = "msdial"
        elif "metaboscape" in file.lower():
            tool = "metaboscape"
        else:
            raise ValueError(f"Unknown tool type in file name: {file}")
        tools.append(tool)

    # Merge all feature tables
    merged_feature_tables = {}
    for annotated_path, harmonized_path, tool_type in zip(
        annotation_paths, harmonized_paths, tools
    ):
        if not os.path.exists(annotated_path) or not os.path.exists(harmonized_path):
            continue

        annotations = pd.read_parquet(annotated_path)
        feature_table = pd.read_parquet(harmonized_path)

        annotations["FEATURE_ID"] = annotations["FEATURE_ID"].astype(
            feature_table["FEATURE_ID"].dtype
        )
        feature_table = feature_table.merge(annotations, on="FEATURE_ID", how="left")
        merged_feature_tables[tool_type] = feature_table

    # Compute metrics
    metrics = {}
    for tool, feature_table in merged_feature_tables.items():
        detected_inchikey_14s = feature_table[
            (feature_table[SCORE_COLUMN] >= similarity_threshold)
        ][INCHIKEY_COLUMN].apply(lambda x: x[:14])

        metrics[tool] = {
            "MS1 Count": len(feature_table),
            "MS2 Count": len(feature_table[feature_table["MS/MS_ASSIGNED"]]),
            "Annotated Count": len(detected_inchikey_14s),
        }
    return metrics


def compute_precision_recall_data(
    merged_feature_tables: Dict[str, pd.DataFrame],
    ground_truth_data: pd.DataFrame,
    precision_type="regular",
    recall_type="regular",
    f1_type="regular",
    score_threshold=0.7,
):
    """
    Compute precision/recall data for all tools given merged feature tables and ground truth data.

    Parameters:
    -----------
    merged_feature_tables: dict
        A dictionary where keys are tool names and values are DataFrames containing the merged feature tables for each tool.
    ground_truth_data: DataFrame
        A DataFrame containing the ground truth annotations with a column 'inchikey_2d'.
    precision_type: str
        Type of precision calculation: 'regular' or 'pseudo'. Default is 'regular'.
    recall_type: str
        Type of recall calculation: 'regular'. Default is 'regular'.
    f1_type: str
        Type of F1 calculation: 'regular' or 'pseudo'. Default is 'regular'.
    score_threshold: float
        Threshold for considering a feature as annotated based on the similarity score. Default is 0.7.

    Returns:
    --------
    graph_range: numpy array
        An array of thresholds used for computing precision and recall.
    precision_recall_dict: dict
        A dictionary where keys are tool names and values are dictionaries containing lists of precision,
        recall, and F1 scores for each threshold.
    confusion_matrix_dict: dict
        A dictionary where keys are tool names and values are dictionaries containing the confusion matrix
        components (TP, FP, FN) for each threshold.
    """

    if precision_type not in ["regular", "pseudo"]:
        raise ValueError("precision_type must be 'regular' or 'pseudo'")
    if recall_type not in ["regular"]:
        raise ValueError("recall_type must be 'regular'")
    if f1_type not in ["regular", "pseudo"]:
        raise ValueError("f1_type must be 'regular' or 'pseudo'")

    true_inchikeys = set(
        ground_truth_data["inchikey_2d"].apply(lambda x: x.split("-")[0]).to_list()
    )
    print(f"Number of unique true InChIKeys (14-character): {len(true_inchikeys)}")

    graph_range = np.arange(0, 1.01, 0.01)

    precision_recall_dict = {}
    confusion_matrix_dict = {}
    for tool, df in merged_feature_tables.items():
        df = df[df[SCORE_COLUMN].notna()]
        precision_recall_dict[tool] = {"precision": [], "recall": [], "f1": []}

        for threshold in graph_range:
            predicted_inchikeys = (
                df[df[SCORE_COLUMN] >= threshold][INCHIKEY_COLUMN]
                .apply(lambda x: x.split("-")[0])
                .to_list()
            )

            unique_true_positives = len(
                set(predicted_inchikeys).intersection(true_inchikeys)
            )
            unique_false_positives = len(
                set(predicted_inchikeys).difference(true_inchikeys)
            )
            unique_false_negatives = len(
                true_inchikeys.difference(set(predicted_inchikeys))
            )

            if precision_type == "pseudo" or f1_type == "pseudo":
                duplicate_true_positives = len(
                    [item for item in predicted_inchikeys if item in true_inchikeys]
                )
                duplicate_false_positives = (
                    len(predicted_inchikeys) - duplicate_true_positives
                )

            if precision_type == "regular":
                current_precision = (
                    unique_true_positives
                    / (unique_true_positives + unique_false_positives)
                    if (unique_true_positives + unique_false_positives) > 0
                    else 0
                )
            elif precision_type == "pseudo":
                current_precision = (
                    unique_true_positives
                    / (duplicate_true_positives + duplicate_false_positives)
                    if (duplicate_true_positives + duplicate_false_positives) > 0
                    else 0
                )

            if recall_type == "regular":
                current_recall = (
                    unique_true_positives
                    / (unique_true_positives + unique_false_negatives)
                    if (unique_true_positives + unique_false_negatives) > 0
                    else 0
                )

            if f1_type == "regular":
                f1_precision = current_precision
            elif f1_type == "pseudo":
                f1_precision = (
                    unique_true_positives
                    / (duplicate_true_positives + duplicate_false_positives)
                    if (duplicate_true_positives + duplicate_false_positives) > 0
                    else 0
                )

            precision_recall_dict[tool]["precision"].append(current_precision)
            precision_recall_dict[tool]["recall"].append(current_recall)

            current_f1 = (
                2 * (f1_precision * current_recall) / (f1_precision + current_recall)
                if (f1_precision + current_recall) > 0
                else 0
            )

            precision_recall_dict[tool]["f1"].append(current_f1)

        confusion_predicted_inchikeys = (
            df[df[SCORE_COLUMN] >= score_threshold][INCHIKEY_COLUMN]
            .apply(lambda x: x.split("-")[0])
            .to_list()
        )
        confusion_unique_true_positives = len(
            set(confusion_predicted_inchikeys).intersection(true_inchikeys)
        )
        confusion_unique_false_positives = len(
            set(confusion_predicted_inchikeys).difference(true_inchikeys)
        )
        confusion_unique_false_negatives = len(
            true_inchikeys.difference(set(confusion_predicted_inchikeys))
        )

        if precision_type == "pseudo" or f1_type == "pseudo":
            confusion_duplicate_true_positives = len(
                [
                    item
                    for item in confusion_predicted_inchikeys
                    if item in true_inchikeys
                ]
            )
            confusion_duplicate_false_positives = (
                len(confusion_predicted_inchikeys) - confusion_duplicate_true_positives
            )

        if precision_type == "regular":
            confusion_precision = (
                confusion_unique_true_positives
                / (confusion_unique_true_positives + confusion_unique_false_positives)
                if (confusion_unique_true_positives + confusion_unique_false_positives)
                > 0
                else 0
            )
        elif precision_type == "pseudo":
            confusion_precision = (
                confusion_unique_true_positives
                / (
                    confusion_duplicate_true_positives
                    + confusion_duplicate_false_positives
                )
                if (
                    confusion_duplicate_true_positives
                    + confusion_duplicate_false_positives
                )
                > 0
                else 0
            )

        if f1_type == "regular":
            confusion_f1_precision = (
                confusion_unique_true_positives
                / (confusion_unique_true_positives + confusion_unique_false_positives)
                if (confusion_unique_true_positives + confusion_unique_false_positives)
                > 0
                else 0
            )
        elif f1_type == "pseudo":
            confusion_f1_precision = (
                confusion_unique_true_positives
                / (
                    confusion_duplicate_true_positives
                    + confusion_duplicate_false_positives
                )
                if (
                    confusion_duplicate_true_positives
                    + confusion_duplicate_false_positives
                )
                > 0
                else 0
            )

        if f1_type == "regular" or f1_type == "pseudo":
            confusion_recall = (
                confusion_unique_true_positives
                / (confusion_unique_true_positives + confusion_unique_false_negatives)
                if (confusion_unique_true_positives + confusion_unique_false_negatives)
                > 0
                else 0
            )

        confusion_f1 = (
            2
            * (confusion_f1_precision * confusion_recall)
            / (confusion_f1_precision + confusion_recall)
            if (confusion_f1_precision + confusion_recall) > 0
            else 0
        )

        confusion_matrix_dict[tool] = {
            "TP": confusion_unique_true_positives,
            "FP": confusion_unique_false_positives,
            "FN": confusion_unique_false_negatives,
            "precision": confusion_precision,
            "recall": confusion_recall,
            "f1": confusion_f1,
        }
    return graph_range, precision_recall_dict, confusion_matrix_dict


def compute_precision_recall_data_all(
    merged_feature_tables,
    ground_truth_data,
    precision_type="regular",
    recall_type="regular",
    f1_type="regular",
    confusion_matrix_threshold=0.7,
):
    """
    Compute precision/recall data for all annotation types and tools.

    Parameters:
    -----------
    merged_feature_tables: dict
        A dictionary where keys are annotation types and values are dictionaries mapping tool names to their merged feature tables.
    ground_truth_data: DataFrame
        A DataFrame containing the ground truth annotations with a column 'inchikey_2d'.
    precision_type: str
        Type of precision calculation: 'regular' or 'pseudo'. Default is 'regular'.
    recall_type: str
        Type of recall calculation: 'regular'. Default is 'regular'.
    f1_type: str
        Type of F1 calculation: 'regular' or 'pseudo'. Default is 'regular'.
    confusion_matrix_threshold: float
        Threshold for computing the confusion matrix. Default is 0.7.

    Returns:
        graph_range: numpy array of thresholds
        precision_recall_dict: {annotation_type: {tool: {"precision": [...], "recall": [...], "f1": [...]}}}
        confusion_matrix_dict: {annotation_type: {tool: {"TP": ..., "FP": ..., "FN": ..., ...}}}
    """
    if precision_type not in ["regular", "pseudo"]:
        raise ValueError("precision_type must be 'regular' or 'pseudo'")
    if recall_type not in ["regular"]:
        raise ValueError("recall_type must be 'regular'")
    if f1_type not in ["regular", "pseudo"]:
        raise ValueError("f1_type must be 'regular' or 'pseudo'")

    graph_range = np.arange(0, 1.01, 0.01)

    graph_range_all = {}
    all_precision_recall = {}
    all_confusion_matrix = {}

    for annotation_type, tool_tables in merged_feature_tables.items():
        graph_range, precision_recall_dict, confusion_matrix_dict = (
            compute_precision_recall_data(
                tool_tables,
                ground_truth_data,
                precision_type=precision_type,
                recall_type=recall_type,
                f1_type=f1_type,
                score_threshold=confusion_matrix_threshold,
            )
        )

        all_precision_recall[annotation_type] = precision_recall_dict
        all_confusion_matrix[annotation_type] = confusion_matrix_dict
        graph_range_all[annotation_type] = graph_range

    return graph_range_all, all_precision_recall, all_confusion_matrix


def get_euler_data(
    path_to_dataset: str,
    similarity_threshold: float = 0.7,
    annotation_subfolder="annotated_spectral_entropy",
) -> dict:

    annotation_path = os.path.join(path_to_dataset, annotation_subfolder)

    # Define paths and tools
    annotation_paths = []
    harmonized_paths = []
    tools = []
    for file in os.listdir(annotation_path):
        annotation_paths.append(os.path.join(annotation_path, file))
        harmonized_filepath = os.path.join(annotation_path, file).replace(
            annotation_subfolder, "harmonized"
        )
        harmonized_paths.append(harmonized_filepath)
        if "mzmine" in file.lower():
            tool = "mzmine"
        elif "msdial" in file.lower():
            tool = "msdial"
        elif "metaboscape" in file.lower():
            tool = "metaboscape"
        else:
            raise ValueError(f"Unknown tool type in file name: {file}")
        tools.append(tool)

    result_sets = {}
    for i, annotated_path in enumerate(annotation_paths):
        if not os.path.exists(annotated_path):
            continue

        annotations = pd.read_parquet(annotated_path)
        harmonized_path = harmonized_paths[i]
        if not os.path.exists(harmonized_path):
            continue

        feature_table = pd.read_parquet(harmonized_path)
        annotations["FEATURE_ID"] = annotations["FEATURE_ID"].astype(
            feature_table["FEATURE_ID"].dtype
        )
        feature_table = feature_table.merge(annotations, on="FEATURE_ID", how="left")
        tool_type = annotated_path.split("_")[-4]

        if tool_type not in {"mzmine", "msdial", "metaboscape"}:
            raise ValueError(f"Unknown tool type in file name: {annotated_path}")

        feature_table = feature_table[
            feature_table[SCORE_COLUMN] >= similarity_threshold
        ]

        result_set = (
            feature_table[INCHIKEY_COLUMN].apply(lambda x: x[:14]).dropna().to_list()
        )
        result_sets[tool_type] = set(result_set)

    return list(result_sets.keys()), list(result_sets.values())


def compute_max_precision_recall_data(
    feature_table,
    ground_truth_data,
    fixed_threshold=0.7,
    precision_type="regular",
    recall_type="regular",
    f1_type="regular",
):
    """
    Compute max precision, recall, and F1 metrics with their thresholds for a single feature table.
    """
    if precision_type not in ["regular", "pseudo"]:
        raise ValueError("precision_type must be 'regular' or 'pseudo'")
    if recall_type not in ["regular"]:
        raise ValueError("recall_type must be 'regular'")
    if f1_type not in ["regular", "pseudo"]:
        raise ValueError("f1_type must be 'regular' or 'pseudo'")

    true_inchikeys = set(
        ground_truth_data["inchikey_2d"].apply(lambda x: x.split("-")[0]).to_list()
    )
    graph_range = np.arange(0, 1.01, 0.01)

    df = feature_table[feature_table[SCORE_COLUMN].notna()]

    precision_list = []
    recall_list = []
    f1_list = []

    for threshold in graph_range:
        predicted_inchikeys = (
            df[df[SCORE_COLUMN] >= threshold][INCHIKEY_COLUMN]
            .apply(lambda x: x.split("-")[0])
            .to_list()
        )

        unique_true_positives = len(
            set(predicted_inchikeys).intersection(true_inchikeys)
        )
        unique_false_positives = len(
            set(predicted_inchikeys).difference(true_inchikeys)
        )
        unique_false_negatives = len(
            true_inchikeys.difference(set(predicted_inchikeys))
        )

        if precision_type == "pseudo":
            duplicate_true_positives = len(
                [item for item in predicted_inchikeys if item in true_inchikeys]
            )
            duplicate_false_positives = (
                len(predicted_inchikeys) - duplicate_true_positives
            )

        if precision_type == "regular":
            current_precision = (
                unique_true_positives / (unique_true_positives + unique_false_positives)
                if (unique_true_positives + unique_false_positives) > 0
                else 0
            )
        elif precision_type == "pseudo":
            current_precision = (
                unique_true_positives
                / (duplicate_true_positives + duplicate_false_positives)
                if (duplicate_true_positives + duplicate_false_positives) > 0
                else 0
            )

        if recall_type == "regular":
            current_recall = (
                unique_true_positives / (unique_true_positives + unique_false_negatives)
                if (unique_true_positives + unique_false_negatives) > 0
                else 0
            )

        precision_list.append(current_precision)
        recall_list.append(current_recall)

        current_f1 = (
            2
            * (current_precision * current_recall)
            / (current_precision + current_recall)
            if (current_precision + current_recall) > 0
            else 0
        )
        f1_list.append(current_f1)

    # Compute max metrics and their thresholds
    max_precision_idx = np.argmax(precision_list)
    max_recall_idx = np.argmax(recall_list)
    max_f1_idx = np.argmax(f1_list)

    # Compute metrics at fixed threshold
    fixed_predicted_inchikeys = (
        df[df[SCORE_COLUMN] >= fixed_threshold][INCHIKEY_COLUMN]
        .apply(lambda x: x.split("-")[0])
        .to_list()
    )

    fixed_unique_true_positives = len(
        set(fixed_predicted_inchikeys).intersection(true_inchikeys)
    )
    fixed_unique_false_positives = len(
        set(fixed_predicted_inchikeys).difference(true_inchikeys)
    )
    fixed_unique_false_negatives = len(
        true_inchikeys.difference(set(fixed_predicted_inchikeys))
    )

    if precision_type == "pseudo":
        fixed_duplicate_true_positives = len(
            [item for item in fixed_predicted_inchikeys if item in true_inchikeys]
        )
        fixed_duplicate_false_positives = (
            len(fixed_predicted_inchikeys) - fixed_duplicate_true_positives
        )

    if precision_type == "regular":
        fixed_precision = (
            fixed_unique_true_positives
            / (fixed_unique_true_positives + fixed_unique_false_positives)
            if (fixed_unique_true_positives + fixed_unique_false_positives) > 0
            else 0
        )
    elif precision_type == "pseudo":
        fixed_precision = (
            fixed_unique_true_positives
            / (fixed_duplicate_true_positives + fixed_duplicate_false_positives)
            if (fixed_duplicate_true_positives + fixed_duplicate_false_positives) > 0
            else 0
        )

    if recall_type == "regular":
        fixed_recall = (
            fixed_unique_true_positives
            / (fixed_unique_true_positives + fixed_unique_false_negatives)
            if (fixed_unique_true_positives + fixed_unique_false_negatives) > 0
            else 0
        )

    fixed_f1 = (
        2 * (fixed_precision * fixed_recall) / (fixed_precision + fixed_recall)
        if (fixed_precision + fixed_recall) > 0
        else 0
    )

    max_metrics = {
        "max_precision": precision_list[max_precision_idx],
        "max_recall": recall_list[max_recall_idx],
        "max_f1": f1_list[max_f1_idx],
        "max_precision_threshold": graph_range[max_precision_idx],
        "max_recall_threshold": graph_range[max_recall_idx],
        "max_f1_threshold": graph_range[max_f1_idx],
        "fixed_precision": fixed_precision,
        "fixed_recall": fixed_recall,
        "fixed_f1": fixed_f1,
    }

    return max_metrics


def compute_r2_values_for_plasma_spikein(
    merged_feature_tables: Dict[str, pd.DataFrame],
    groundtruth_data: pd.DataFrame,
    threshold: float = 0.7,
):
    r2_dict = {tool: [] for tool in merged_feature_tables.keys()}
    concentrations_names = ["LOW", "MEDIUM", "HIGH", "SUPER_HIGH"]

    for tool, df in merged_feature_tables.items():
        subset_df = df[df[SCORE_COLUMN] >= threshold]
        subset_df[INCHIKEY_COLUMN] = subset_df[INCHIKEY_COLUMN].apply(
            lambda x: x.split("-")[0]
        )

        for _, row in groundtruth_data.iterrows():
            curr_inchikey = row["inchikey_2d"].split("-")[0]

            skip_inchikey = False

            conc_vals = []
            intensities = []
            for conc, concentration_name in enumerate(concentrations_names):
                conc_vals.append(conc)
                curr_sample_group = row[concentration_name]

                # Match based on inchikey14
                curr_df = subset_df[subset_df[INCHIKEY_COLUMN] == curr_inchikey]

                if curr_df.empty:
                    skip_inchikey = True
                    break

                # Get row with max score
                curr_df = curr_df.loc[curr_df[SCORE_COLUMN].idxmax()]
                sample_columns = [
                    col
                    for col in curr_df.index
                    if isinstance(curr_sample_group, str) and curr_sample_group in col
                ]
                curr_avg = curr_df[sample_columns].mean()
                intensities.append(curr_avg)

            if skip_inchikey:
                continue
            else:
                linreg_result = stats.linregress(conc_vals, intensities)
                r2_value = linreg_result.rvalue**2
                p_value = linreg_result.pvalue
                r2_dict[tool].append((r2_value, p_value))

    return r2_dict


def compute_r2_values_for_plant_spikein(
    merged_feature_tables: Dict[str, pd.DataFrame],
    combinations: list[tuple[str, str]],
    groundtruth_data: pd.DataFrame,
    concentrations: set = (50, 100, 200),
    score_threshold: float = 0.7,
):
    r2_dict = {tool: [] for tool in merged_feature_tables.keys()}

    conc_list = sorted(list(concentrations))
    conc_array = np.array(conc_list, dtype=float)

    # Transform settings: (tic_norm, transform_type)
    transform_settings = {
        "rclr": (True, "rclr"),
        "clr": (True, "clr"),
        "raw": (False, None),
    }

    for tool_name, transform_type in combinations:
        if transform_type not in transform_settings:
            print(f"Warning: Transform '{transform_type}' not recognized. Skipping.")
            continue

        tic_norm, transform_arg = TRANSFORM_SETTINGS[transform_type]

        tool_df = merged_feature_tables[tool_name]

        column_names = [col for col in tool_df.columns if "SAMPLE" in col]

        intensities_df = tool_df[column_names]
        if tic_norm:
            intensities_df = intensities_df.div(intensities_df.sum(axis=0), axis=1)
        if transform_arg == "rclr":
            intensities_df = intensities_df.apply(apply_rclr_transform, axis=0)
        elif transform_arg == "clr":
            intensities_df = intensities_df.apply(apply_clr_transform, axis=0)

        formatted_samples = defaultdict(dict)

        for _, column in enumerate(column_names):
            compound_name = column.split("_")[1]
            conc = int(column.split("_")[2].replace("ng", "").strip())
            if conc not in concentrations:
                continue

            formatted_samples[compound_name][conc] = column

        for _, mapping in formatted_samples.items():
            candidates_per_conc = {}
            skip_sample = False

            for conc in conc_list:
                if conc not in mapping:
                    skip_sample = True
                    break

                compound_name = column.split("_")[1]

                inchikey_nonisomeric = list(
                    set(
                        groundtruth_data[
                            groundtruth_data["metabolite_name"] == compound_name
                        ]["inchikey_2d"].values
                    )
                )

                matching_rows = tool_df[
                    (tool_df[INCHIKEY_COLUMN].isin(inchikey_nonisomeric))
                    & (tool_df[SCORE_COLUMN] >= score_threshold)
                ]

                if matching_rows.empty:
                    skip_sample = True
                    break

                candidates_per_conc[conc] = matching_rows

            if skip_sample or len(candidates_per_conc) != len(conc_list):
                continue

            candidate_indices = [
                candidates_per_conc[c].index.tolist() for c in conc_list
            ]
            best_combo = None
            best_score_sum = -1

            for combo in itertools_product(*candidate_indices):
                combo_rows = [tool_df.loc[idx] for idx in combo]
                mzs = [row["M/Z"] for row in combo_rows]
                ccss = [row["CCS"] for row in combo_rows]
                rts = [row["RT"] for row in combo_rows]

                mz_ok = True
                for i in range(len(mzs)):
                    for j in range(i + 1, len(mzs)):
                        mean_mz = (mzs[i] + mzs[j]) / 2
                        if mean_mz == 0:
                            mz_ok = False
                            break
                        ppm_diff = abs(mzs[i] - mzs[j]) / mean_mz * 1e6
                        if ppm_diff > 20:
                            mz_ok = False
                            break
                    if not mz_ok:
                        break
                if not mz_ok:
                    continue

                ccs_ok = True
                for i in range(len(ccss)):
                    for j in range(i + 1, len(ccss)):
                        mean_ccs = (ccss[i] + ccss[j]) / 2
                        if mean_ccs == 0:
                            ccs_ok = False
                            break
                        ccs_pct_diff = abs(ccss[i] - ccss[j]) / mean_ccs * 100
                        if ccs_pct_diff > 1:
                            ccs_ok = False
                            break
                    if not ccs_ok:
                        break
                if not ccs_ok:
                    continue

                # Check pairwise RT within 0.1 min
                rt_ok = True
                for i in range(len(rts)):
                    for j in range(i + 1, len(rts)):
                        if abs(rts[i] - rts[j]) > 0.1:
                            rt_ok = False
                            break
                    if not rt_ok:
                        break
                if not rt_ok:
                    continue

                score_sum = sum(tool_df.loc[idx, SCORE_COLUMN] for idx in combo)
                if score_sum > best_score_sum:
                    best_score_sum = score_sum
                    best_combo = combo

            if best_combo is None:
                continue

            # Get intensities for each concentration from the best combination
            intensity_vals = []
            for k, conc in enumerate(conc_list):
                intensity_vals.append(intensities_df.loc[best_combo[k], mapping[conc]])
            intensity_array = np.array(intensity_vals, dtype=float)

            # Compute R² and p-value
            if np.std(intensity_array) == 0 or np.std(conc_array) == 0:
                continue

            corr, p_value = stats.pearsonr(conc_array, intensity_array)
            r2 = corr**2
            r2_dict[tool_name].append((r2, p_value))

    return r2_dict
