# -*- coding: utf-8 -*-

"""Util functions."""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from numba import njit, prange
from numba.typed import List as NumbaList

import pubchempy as pcp
import rdkit.Chem as Chem

from benchmarking.constants import NUMBER_FILTERED_PEAKS


def compliant_smiles_from_cid(cid):
    if cid is None:
        return None
    # Rate limit of 5 requests per second
    time.sleep(0.2)
    if pcp is None:
        raise ImportError(
            "pubchempy is not installed. Please install it to use this function."
        )
    try:
        compound = pcp.Compound.from_cid(cid)
    except Exception as e:
        print(f"Error fetching compound for CID {cid}: {e}")
        return None
    return compound.smiles


def smiles_to_inchikey_2d(smiles: str) -> str:
    """
    Compute the 2D InChIKey (inchikey14 with sterochemistry) from a SMILES string.

    :param smiles: SMILES representation of the molecule
    :return: 2D InChIKey (inchikey14 with stereochemistry)
    """
    if not smiles or not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    inchikey = Chem.MolToInchiKey(mol)
    inchikey_2d = inchikey.split("-")[0] + "-" + inchikey.split("-")[1]
    return inchikey_2d


def str_to_np_array(s) -> np.ndarray:
    """
    Convert a string representation of a numpy array back to a numpy array.

    Parameters
    ----------
    s : str
         String representation of a numpy array (e.g., "[1, 2, 3]").

    Returns
    -------
    np.ndarray
        The corresponding numpy array. If the input is not a valid string representation of a numpy array
    """
    if isinstance(s, str):
        return np.fromstring(s.strip("[]"), sep=",")

    return np.array([])


@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def reduced_spectral_matrix(input_size, output_size, indices_x, indices_y, scores):
    # Create a new matrix with the reduced size
    reduced_matrix = np.zeros((output_size, output_size), dtype=np.float32)

    block_size = input_size // output_size
    out_indices_x = indices_x // block_size
    out_indices_y = indices_y // block_size

    score_length = len(scores)

    for i in prange(score_length):
        x = min(out_indices_x[i], output_size - 1)
        y = min(out_indices_y[i], output_size - 1)
        reduced_matrix[x, y] += scores[i]

    return reduced_matrix


def _filter_spectrum_peaks(
    intensities: Union[List, np.ndarray],
    mzs: Union[List, np.ndarray],
    min_abs_intensity: Optional[float] = 0.0,
    min_rel_intensity: Optional[float] = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter MS2 spectrum peaks based on both absolute and relative intensity thresholds.

    Parameters
    ----------
    intensities : List or np.ndarray
        List or array of peak intensities.
    mzs : List or np.ndarray
        List or array of m/z values corresponding to the intensities.
    min_abs_intensity : float, optional
        Minimum absolute intensity threshold. Peaks with intensities below this value will be filtered out. Default is 0.0.
    min_rel_intensity : float, optional
        Minimum relative intensity threshold (between 0 and 1). Peaks with intensities below this percentage of the maximum intensity will be filtered out. Default is 0.01 (1% of
        the maximum intensity).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Filtered intensities and corresponding m/z values as numpy arrays.
    """
    # Convert inputs to numpy arrays if they aren't already
    if isinstance(intensities, list):
        intensities = np.array(intensities)

    if isinstance(mzs, list):
        mzs = np.array(mzs)

    # Check input lengths
    if len(intensities) != len(mzs):
        raise ValueError("intensities and mzs must have the same length")

    # Validate threshold types and values
    if not isinstance(min_abs_intensity, (int, float)):
        raise TypeError("min_abs_intensity must be numeric")
    if not isinstance(min_rel_intensity, (int, float)):
        raise TypeError("min_rel_intensity must be numeric")

    if min_abs_intensity is not None and min_abs_intensity < 0:
        raise ValueError("min_abs_intensity cannot be negative")
    if min_rel_intensity is not None and (
        min_rel_intensity < 0 or min_rel_intensity > 1
    ):
        raise ValueError("min_rel_intensity must be between 0 and 1")

    # Check if all intensities are below 1 (indicating normalized spectrum)
    if np.all(intensities <= 1.0):
        return intensities, mzs

    # First filter based on absolute intensity
    abs_mask = intensities >= min_abs_intensity
    intensities = intensities[abs_mask]
    mzs = mzs[abs_mask]

    # Then filter based on relative intensity
    if len(intensities) > 0:
        max_intensity = np.max(intensities)
        rel_mask = intensities / max_intensity >= min_rel_intensity
        intensities = intensities[rel_mask]
        mzs = mzs[rel_mask]

    return intensities, mzs


def filter_spectrum_peaks_in_df(
    df: pd.DataFrame,
    column_intensities: str,
    column_mzs: str,
    min_abs_intensity: float = 0.0,
    min_rel_intensity: float = 0.0,
    with_progress_bar: bool = False,
) -> pd.DataFrame:
    """
    Filter MS2 spectrum peaks in a DataFrame based on both absolute and relative intensity thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the spectrum data.
    column_intensities : str
        Name of the column in the DataFrame that contains the peak intensities.
    column_mzs : str
        Name of the column in the DataFrame that contains the m/z values corresponding to the intensities.
    min_abs_intensity : float, optional
        Minimum absolute intensity threshold. Peaks with intensities below this value will be filtered out.
        Default is 0.0.
    min_rel_intensity : float, optional
        Minimum relative intensity threshold (between 0 and 1). Peaks with intensities below this percentage
        of the maximum intensity will be filtered out. Default is 0.0 (no relative filtering).
    with_progress_bar : bool, optional
        Whether to display a progress bar during processing. Default is False.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the filtered intensities and m/z values, and an additional column indicating
        the number of peaks filtered for each spectrum.
    """

    df_copy = df.copy()

    # Validate column existence
    if column_intensities not in df_copy.columns:
        raise KeyError(f"Column {column_intensities} not found in DataFrame")
    if column_mzs not in df_copy.columns:
        raise KeyError(f"Column {column_mzs} not found in DataFrame")

    new_intensities = []
    new_mzs = []
    num_filtered_peaks = []

    iterator = df_copy.iterrows()
    if with_progress_bar:
        iterator = tqdm(iterator, total=len(df_copy), desc="Filtering spectrum peaks")

    for index, row in iterator:
        # If the intensities and mzs are numpy arrays or lists, process them
        if isinstance(row[column_intensities], np.ndarray) or isinstance(
            row[column_intensities], list
        ):
            intensities = row[column_intensities]
            mzs = row[column_mzs]
        # If the intensities and mzs are NaN, do nothing
        elif pd.isna(row[column_intensities]) or pd.isna(row[column_mzs]):
            new_intensities.append(row[column_intensities])
            new_mzs.append(row[column_mzs])
            num_filtered_peaks.append(np.nan)
            continue

        # If the intensities and mzs are not numpy arrays or lists, raise an error
        else:
            raise ValueError(
                f"Invalid data type for column in row {index}: {row[column_intensities]} / {row[column_mzs]}"  # noqa: E501
            )

        # Filter the intensities and mzs
        intensities, mzs = _filter_spectrum_peaks(
            intensities=intensities,
            mzs=mzs,
            min_abs_intensity=min_abs_intensity,
            min_rel_intensity=min_rel_intensity,
        )

        # Check if the filtered intensities is an empty list
        if len(intensities) == 0:
            # if so, set the intensities and mzs to NaN
            new_intensities.append(np.nan)
            new_mzs.append(np.nan)
            num_filtered_peaks.append(len(row[column_intensities]))
        else:
            new_intensities.append(intensities)
            new_mzs.append(mzs)
            num_filtered_peaks.append(len(row[column_intensities]) - len(intensities))

    # Replace the intensities and mzs in the DataFrame
    df_copy[column_intensities] = new_intensities
    df_copy[column_mzs] = new_mzs

    # Add a new col with the number of peaks filtered for each MS2
    df_copy[NUMBER_FILTERED_PEAKS] = num_filtered_peaks

    return df_copy


def to_typed_bank(specs):
    bank = NumbaList()
    for s in specs:
        a = np.asarray(s, dtype=np.float32, order="C").reshape(-1, 2)
        bank.append(a)
    return bank


def apply_rclr_transform(sample_column):
    """
    Apply the rclr transformation to a sample column.

    Parameters
    ----------
    sample_column : array-like
        A 1D array of values for a single sample (e.g., a column from a DataFrame).

    Returns
    -------
    np.ndarray
        The rclr-transformed values for the input sample column.
    """
    # Mask nonzero values
    mask = sample_column > 0
    filtered_values = sample_column[mask]

    # compute log(GM) of non-zero values
    log_gm = np.mean(np.log(filtered_values))

    # Apply rclr transformation to non-zero values
    rclr_values = np.where(mask, np.log(sample_column) - log_gm, 0)
    return rclr_values


def apply_clr_transform(sample_column, pseudocount="min"):
    """
    Apply the clr transformation to a sample column.

    Parameters
    ----------
    sample_column : array-like
        A 1D array of values for a single sample (e.g., a column from a DataFrame).
    pseudocount : float or str, optional
        A small value to add to the sample column to avoid log(0). If "min", the minimum non-zero value
        in the sample column will be used. Default is "min".

    Returns
    -------
    np.ndarray
        The clr-transformed values for the input sample column.
    """

    # Add pseudocount to avoid log(0)
    if pseudocount == "min":
        min_nonzero = sample_column[sample_column > 0].min()
        pseudocount = min_nonzero

    adjusted_values = sample_column + pseudocount

    # Compute log(GM) of all values
    log_gm = np.mean(np.log(adjusted_values))

    # Apply clr transformation
    clr_values = np.log(adjusted_values) - log_gm
    return clr_values
