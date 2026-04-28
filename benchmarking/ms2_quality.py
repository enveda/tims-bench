# -*- coding: utf-8 -*-

"""Module for computing MS2 spectral quality metrics."""

from typing import List, Union

import numpy as np
import pandas as pd


def calculate_spectral_entropy(intensities: np.ndarray) -> float:
    """
    Calculate spectral entropy from a numpy array of intensities.

    Args:
        intensities (np.ndarray): Array of intensities.

    Returns:
        float: Spectral entropy value.
    """
    fractional_intensities = np.array(intensities) / sum(intensities)
    entropy = -fractional_intensities.dot(np.log(fractional_intensities))
    return entropy


def calculate_total_ms2_intensity(ms2_intensities: pd.Series) -> pd.Series:
    """
    Calculate total MS2 intensity from a series of numpy arrays.

    Args:
        ms2_intensities (pd.Series): Series of numpy arrays containing MS2 intensities.

    Returns:
        pd.Series: Series of total MS2 intensities.
    """
    return ms2_intensities.apply(
        lambda x: (sum(x) if isinstance(x, np.ndarray) or isinstance(x, list) else 0)
    )


def has_uniform_high_peaks(
    intensities: Union[np.ndarray, List[float]],
    threshold: float = 0.5,
    min_peaks: int = 10,
) -> bool:
    """
    Check if MS2 spectrum has uniformly high peaks.

    Args:
        intensities (Union[np.ndarray, List[float]]): Array of intensities.
        threshold (float): Normalized intensity threshold (0-1).
        min_peaks (int): Minimum number of peaks required.

    Returns:
        bool: True if all normalized intensities are above threshold and
        number of peaks is >= min_peaks.
    """
    if not isinstance(intensities, (np.ndarray, list)) or len(intensities) == 0:
        return False

    # Convert to array if it's a list
    if isinstance(intensities, list):
        intensities = np.array(intensities)

    # Check if we have enough peaks
    if len(intensities) < min_peaks:
        return False

    # Normalize intensities relative to max
    max_intensity = np.max(intensities)
    if max_intensity == 0:
        return False

    normalized_intensities = intensities / max_intensity

    # Check if all normalized intensities are above threshold
    return np.all(normalized_intensities >= threshold)


def has_any_intensity_above_threshold(
    intensities: Union[np.ndarray, List[float]],
    threshold: float = 200,
) -> bool:
    """
    Check if any MS2 intensity is above a given threshold.

    Args:
        intensities (Union[np.ndarray, List[float]]): Array of intensities.
        threshold (float): Intensity threshold.

    Returns:
        bool: True if any intensity is above the threshold.
    """
    if not isinstance(intensities, (np.ndarray, list)) or len(intensities) == 0:
        return False

    # Convert to array if it's a list
    if isinstance(intensities, list):
        intensities = np.array(intensities)

    # Check if any intensity is above threshold
    return np.any(intensities >= threshold)


def has_no_intensities_above_threshold(
    intensities: Union[np.ndarray, List[float]],
    threshold: float = 200,
) -> bool:
    """
    Check if no MS2 intensities are above a given threshold.

    Args:
        intensities (Union[np.ndarray, List[float]]): Array of intensities.
        threshold (float): Intensity threshold.

    Returns:
        bool: True if no intensities are above the threshold.
    """
    if not isinstance(intensities, (np.ndarray, list)) or len(intensities) == 0:
        return True

    # Convert to array if it's a list
    if isinstance(intensities, list):
        intensities = np.array(intensities)

    # Check if no intensity is above threshold
    return not np.any(intensities >= threshold)


def assess_ms2_quality(
    feature_table: pd.DataFrame,
    spectral_entropy_min_threshold: float = 1.5,
    peak_uniformity_threshold: float = 0.5,
    min_peaks_for_uniformity_check: int = 10,
    intensity_max_threshold: float = 200,
    intensities_column: str = "MS/MS_INTENSITIES",
    ms2_qc_column: str = "MS2_QC",
    total_ms2_intensity_column: str = "TOTAL_MS2_INTENSITY",
    spectral_entropy_column: str = "SPECTRAL_ENTROPY",
) -> pd.DataFrame:
    """
    Assign MS2 quality based on spectral info and peak distribution .

    Args:
        feature_table (pd.DataFrame): DataFrame containing feature data.
        spectral_entropy_min_threshold (float): Threshold for spectral entropy.
        peak_uniformity_threshold (float): Threshold for normalized intensity in uniformity check.
        min_peaks_for_uniformity_check (int): Minimum number of peaks required for uniformity check.
        intensity_max_threshold (float): Maximum threshold for all intensities.

    Returns:
        pd.DataFrame: DataFrame with calculated spectral information.
    """
    # Check spectral_entropy_min_threshold is a number and non-negative
    if (
        not isinstance(spectral_entropy_min_threshold, (int, float))
        or spectral_entropy_min_threshold < 0
    ):
        raise ValueError("spectral_entropy_min_threshold must be a non-negative number")

    # Step 1: Calculate total MS2 intensity
    total_ms2_intensity = calculate_total_ms2_intensity(
        feature_table[intensities_column]
    )
    # Step 2: Calculate spectral entropy for each row with ms2 intensities available
    spectral_entropy = feature_table[intensities_column].apply(
        lambda x: (
            calculate_spectral_entropy(x)
            if isinstance(x, np.ndarray) or isinstance(x, list)
            else np.nan
        )
    )
    # Step 3: Assign initial MS2 quality based on spectral entropy threshold
    ms2_qc = spectral_entropy.apply(
        lambda x: (
            "GOOD"
            if pd.notna(x)
            and pd.notna(spectral_entropy_min_threshold)
            and x > spectral_entropy_min_threshold
            else "BAD" if pd.notna(x) else "NO_MS2"
        )
    )

    # Step 4: Among the ones with good quality, evaluate whether the MS2s
    # are made of a bunch of peaks with similar intensities or if no intensity is above threshold

    ms2_qc_df = pd.DataFrame(
        {
            intensities_column: feature_table[intensities_column],
            ms2_qc_column: ms2_qc,
        }
    )

    def refine_ms2_qc(row):
        # Skip if already marked as bad or no MS2
        if row[ms2_qc_column] != "GOOD":
            return row[ms2_qc_column]

        # Check for uniform high peaks
        if has_uniform_high_peaks(
            intensities=row[intensities_column],
            threshold=peak_uniformity_threshold,
            min_peaks=min_peaks_for_uniformity_check,
        ):
            return "BAD"

        # Check if no intensity is above threshold
        if has_no_intensities_above_threshold(
            intensities=row[intensities_column],
            threshold=intensity_max_threshold,
        ):
            return "BAD"

        return "GOOD"

    ms2_qc_df[ms2_qc_column] = ms2_qc_df.apply(refine_ms2_qc, axis=1)

    spectral_info_df = pd.DataFrame(
        {
            total_ms2_intensity_column: total_ms2_intensity,
            spectral_entropy_column: spectral_entropy,
            ms2_qc_column: ms2_qc_df[ms2_qc_column],
        }
    )

    return spectral_info_df
