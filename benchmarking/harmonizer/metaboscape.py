# -*- coding: utf-8 -*-

"""Code to harmonize Metaboscape files and prepare them for analysis."""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import collections
import numpy as np
import pandas as pd

from benchmarking.harmonizer.adducts import calculate_adduct_mz
from benchmarking.harmonizer.loader import mgf_to_feature_peaks_dict


def harmonize_metaboscape(
    metaboscape_mgf_path: str, metaboscape_data_path: str
) -> pd.DataFrame:
    """
    Harmonizes Metaboscape data by reading the CSV file.

    Args:
        metaboscape_mgf_path (str): Path to the Metaboscape MGF file.
        metaboscape_data_path (str): Path to the Metaboscape CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing harmonized Metaboscape data.
    """
    df = pd.read_csv(metaboscape_data_path, encoding="utf-8")
    remove_columns = [
        "SIGMA_SCORE",
        "KEGG",
        "NAME_METABOSCAPE",
        "MOLECULAR_FORMULA",
        "CAS",
        "MaxIntensity",
    ]
    df = df.drop(
        columns=[col for col in remove_columns if col in df.columns],
        errors="ignore",
    )
    df = df.rename({"PEPMASS": "M/Z"}, axis="columns")

    # get location of ADDUCT column
    adduct_col_loc = df.columns.get_loc("ADDUCT") if "ADDUCT" in df.columns else None

    if adduct_col_loc is None:
        raise ValueError("The 'ADDUCT' column is missing from the Metaboscape data.")

    before_and_including_adduct = df.iloc[:, : adduct_col_loc + 1]

    if "ION_MOBILITY" not in before_and_including_adduct.columns:
        before_and_including_adduct["ION_MOBILITY"] = "N/A"

    mass_spectra, precursors = mgf_to_feature_peaks_dict(
        metaboscape_mgf_path, with_precursors=True
    )
    if not mass_spectra:
        raise ValueError("No valid FEATURE_ID found in the MGF file.")

    ms_ms_spectra = before_and_including_adduct["FEATURE_ID"].map(mass_spectra)
    before_and_including_adduct["MS/MS_MZS"] = ms_ms_spectra.apply(
        lambda x: (
            [mz for mz, _ in x] if isinstance(x, collections.abc.Sequence) else []
        )
    )
    before_and_including_adduct["MS/MS_INTENSITIES"] = ms_ms_spectra.apply(
        lambda x: (
            [intensity for _, intensity in x]
            if isinstance(x, collections.abc.Sequence)
            else []
        )
    )

    before_and_including_adduct["MS/MS_ASSIGNED"] = ms_ms_spectra.notna()
    before_and_including_adduct = before_and_including_adduct[
        [
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
    ]

    molecule_masses = before_and_including_adduct["M/Z"]
    adducts = before_and_including_adduct["ADDUCT"]

    calculated_precursors = []
    for mass, adduct in zip(molecule_masses, adducts):
        try:
            calc_precursor = calculate_adduct_mz(mass, adduct)
            calculated_precursors.append(calc_precursor)
        except:
            calculated_precursors.append(None)

    calculated_precursors = np.array(calculated_precursors)

    # Use precursors to update M/Z values, if available, else fill with None

    before_and_including_adduct["M/Z"] = before_and_including_adduct["FEATURE_ID"].map(
        lambda x: precursors.get(x, None)
    )

    # Fill None values in M/Z with calculated_precursors
    before_and_including_adduct["M/Z"] = before_and_including_adduct[
        "M/Z"
    ].combine_first(pd.Series(calculated_precursors))

    before_and_including_adduct["RT"] = (
        before_and_including_adduct["RT"].astype(float) / 60.0
    )  # Convert RT from seconds to minutes
    after_adduct = df.iloc[:, adduct_col_loc + 1 :]
    sorted_after_adduct = after_adduct.reindex(sorted(after_adduct.columns), axis=1)

    df = pd.concat([before_and_including_adduct, sorted_after_adduct], axis=1)

    df["ADDUCT"] = df["ADDUCT"].apply(
        lambda x: x.replace("ION=", "").replace("[M+H+H]2+", "[M+2H]+2")
    )

    return df
