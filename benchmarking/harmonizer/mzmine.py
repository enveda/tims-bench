# -*- coding: utf-8 -*-

"""Code to harmonize raw MZmine files and prepare them for analysis."""

import collections
import pandas as pd
from benchmarking.harmonizer.loader import mgf_to_feature_peaks_dict


def harmonize_mzmine(mzmine_mgf_path: str, mzmine_data_path: str) -> pd.DataFrame:
    """
    Harmonizes MZmine data by reading the feature list CSV file.

    :param mzmine_mgf_path: Path to the MZmine MGF file.
    :param mzmine_data_path: Path to the MZmine feature list CSV file.
    :return: A DataFrame containing harmonized MZmine data.
    """
    df = pd.read_csv(mzmine_data_path, encoding="utf-8", low_memory=False)
    df = df.iloc[:, :-1]

    remove_cols = [
        "row ion mobility unit",
        "correlation group ID",
        "annotation network number",
        "auto MS2 verify",
        "partners",
        "identified by n=",
        "neutral M mass",
    ]

    df = df.drop(
        columns=[col for col in remove_cols if col in df.columns],
        errors="ignore",
    )

    df = df.rename(
        lambda x: x.replace(".d", "").replace("Peak area", "").strip(),
        axis="columns",
    )
    df = df.rename(lambda x: x.replace("row ", ""), axis="columns")

    best_ion_loc = df.columns.get_loc("best ion") if "best ion" in df.columns else None
    if best_ion_loc is None:
        raise ValueError("The 'best ion' column is missing from the MZmine data.")

    before_and_including_best_ion = df.iloc[:, : best_ion_loc + 1]
    after_best_ion = df.iloc[:, best_ion_loc + 1 :]
    sorted_after_best_ion = after_best_ion.reindex(
        sorted(after_best_ion.columns), axis=1
    )
    before_and_including_best_ion = before_and_including_best_ion.rename(
        {
            "best ion": "ADDUCT",
            "retention time": "RT",
            "ion mobility": "ION_MOBILITY",
            "m/z": "M/Z",
            "ID": "FEATURE_ID",
        },
        axis="columns",
    )
    mass_spectra = mgf_to_feature_peaks_dict(mzmine_mgf_path)
    if not mass_spectra:
        raise ValueError("No valid FEATURE_ID found in the MGF file.")
    ms_ms_spectra = before_and_including_best_ion["FEATURE_ID"].map(mass_spectra)
    before_and_including_best_ion["MS/MS_MZS"] = ms_ms_spectra.apply(
        lambda x: ([i[0] for i in x] if isinstance(x, collections.abc.Sequence) else [])
    )
    before_and_including_best_ion["MS/MS_INTENSITIES"] = ms_ms_spectra.apply(
        lambda x: ([i[1] for i in x] if isinstance(x, collections.abc.Sequence) else [])
    )
    before_and_including_best_ion["MS/MS_ASSIGNED"] = ms_ms_spectra.notna()
    before_and_including_best_ion = before_and_including_best_ion[
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

    df = pd.concat([before_and_including_best_ion, sorted_after_best_ion], axis=1)

    return df
