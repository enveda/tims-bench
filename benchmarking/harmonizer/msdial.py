# -*- coding: utf-8 -*-

"""Code to harmonize raw MS-DIAL files and prepare them for analysis."""

import collections
from typing import List
import pandas as pd


def read_msdial_master(msdial_data_path: str, sep=",") -> pd.DataFrame:
    """
    Reads a CSV file from MS-DIAL and returns a cleaned DataFrame.
    Args:
        msdial_data_path (str): Path to the MS-DIAL CSV file.
    Returns:
        pd.DataFrame: A DataFrame containing the cleaned data.
    """
    with open(msdial_data_path, "r", encoding="utf-8") as f:
        lines: List[str] = [line.rstrip("\n") for line in f if line.strip()]
    if not lines:
        return pd.DataFrame()

    comma_counts = [ln.count(sep) for ln in lines]
    expected_commas, _ = collections.Counter(comma_counts).most_common(1)[0]
    expected_cols = expected_commas + 1

    header = lines[0].split(sep)
    try:
        name_idx = header.index("Metabolite name")
    except ValueError:
        print(f"Headers in {msdial_data_path} do not contain 'Metabolite name'.")
        raise ValueError("The 'Metabolite name' column is missing from the headers.")

    cleaned_rows = []
    for ln in lines[1:]:
        parts = ln.split(sep)
        if len(parts) != expected_cols:
            extra = len(parts) - expected_cols
            before = parts[:name_idx]
            inside = parts[name_idx : name_idx + extra + 1]
            after = parts[name_idx + extra + 1 :]
            merged_name = sep.join(inside).strip()
            parts = before + [merged_name] + after
        cleaned_rows.append([p.strip() for p in parts])

    df = pd.DataFrame(cleaned_rows, columns=header)
    return df


def read_msdial_peak(msdial_data_path: str, sep=",") -> pd.DataFrame:
    """
    Reads a CSV file of peak data from MS-DIAL and returns a cleaned and aggregated DataFrame.

    Args:
        msdial_data_path (str): Path to the MS-DIAL peak CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned and aggregated peak data.
    """
    df = pd.read_csv(msdial_data_path, encoding="utf-8", sep=sep)
    required_cols = {"ID", "File", "Height"}

    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    try:
        height_df = df.pivot(index="ID", columns="File", values="Height")
    except:
        height_df = df.pivot_table(
            index="ID", columns="File", values="Height", aggfunc="max"
        )
    return height_df


def harmonize_msdial_multi_sample(
    msdial_master_path: str, msdial_peak_path: str
) -> pd.DataFrame:
    """
    Harmonizes MS-DIAL data by reading master and peak files, cleaning, and aggregating.

    Args:
        msdial_master_path (str): Path to the MS-DIAL master CSV file.
        msdial_peak_path (str): Path to the MS-DIAL peak CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing harmonized MS-DIAL data.
    """
    if ".txt" in msdial_master_path or ".tsv" in msdial_master_path:
        sep = "\t"
    elif ".csv" in msdial_master_path:
        sep = ","
    else:
        raise ValueError("Invalid file type for MS-DIAL master file")

    master_df = read_msdial_master(msdial_master_path, sep=sep)
    peak_df = read_msdial_peak(msdial_peak_path, sep=sep)

    master_df = master_df[
        [
            "Alignment ID",
            "Average Rt(min)",
            "Average Mz",
            "Average mobility",
            "Average CCS",
            "Adduct type",
            "MS/MS assigned",
            "MS/MS spectrum",
        ]
    ]
    master_df = master_df.rename(
        columns={
            "Alignment ID": "FEATURE_ID",
            "Average Rt(min)": "RT",
            "Average Mz": "M/Z",
            "Average mobility": "ION_MOBILITY",
            "Average CCS": "CCS",
            "Adduct type": "ADDUCT",
            "MS/MS assigned": "MS/MS_ASSIGNED",
            "MS/MS spectrum": "MS/MS_SPECTRUM",
        }
    )

    # convert dtypes
    master_df["FEATURE_ID"] = master_df["FEATURE_ID"].astype(int)
    master_df["RT"] = master_df["RT"].astype(float)
    master_df["M/Z"] = master_df["M/Z"].astype(float)
    master_df["ION_MOBILITY"] = master_df["ION_MOBILITY"].astype(float)
    master_df["CCS"] = master_df["CCS"].astype(float)
    master_df["ADDUCT"] = master_df["ADDUCT"].astype(str)
    master_df["ADDUCT"] = master_df["ADDUCT"].apply(
        lambda x: "[" + x + "]" if (not x.startswith("[") and x != "") else x
    )
    master_df["MS/MS_ASSIGNED"] = (
        master_df["MS/MS_ASSIGNED"]
        .astype(str)
        .apply(lambda x: True if x.lower() == "true" else False)
    )
    master_df["MS/MS_SPECTRUM"] = master_df["MS/MS_SPECTRUM"].astype(str)
    master_df["MS/MS_SPECTRUM"] = master_df["MS/MS_SPECTRUM"].apply(
        lambda x: x if x != "null" else None
    )

    ms_spectra = master_df["MS/MS_SPECTRUM"].apply(
        lambda x: (
            [[float(v) for v in i.split(":")] for i in x.split()]
            if x is not None
            else None
        )
    )
    master_df["MS/MS_MZS"] = ms_spectra.apply(
        lambda x: ([i[0] for i in x] if isinstance(x, collections.abc.Sequence) else [])
    )
    master_df["MS/MS_INTENSITIES"] = ms_spectra.apply(
        lambda x: ([i[1] for i in x] if isinstance(x, collections.abc.Sequence) else [])
    )
    master_df = master_df[
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

    peak_df = peak_df.drop(
        columns=[
            "IdentificationMethod",
            "MsmsIncluded",
            "Mobility",
            "CCS",
            "SN",
            "Rt",
            "Class",
        ],
        errors="ignore",
    )
    # reset index and keep 'ID' as a column
    peak_df = peak_df.reset_index()
    peak_df = peak_df.rename(columns={"ID": "FEATURE_ID"})
    peak_df_after_feature_id = peak_df.drop(columns=["FEATURE_ID"], errors="ignore")
    feature_id_col = peak_df["FEATURE_ID"].astype(int)
    peak_df_after_feature_id = peak_df_after_feature_id.reindex(
        sorted(peak_df_after_feature_id.columns), axis=1
    )

    peak_df = pd.concat([feature_id_col, peak_df_after_feature_id], axis=1)

    # Merge master and peak data on 'FEATURE_ID'
    harmonized_df = pd.merge(master_df, peak_df, on="FEATURE_ID", how="outer")
    return harmonized_df


def is_valid(value) -> bool:
    return value is not pd.NA and not pd.isna(value) and value is not None


def harmonize_msdial_multi_sample_combined(msdial_csv_path: str) -> pd.DataFrame:
    """
    Harmonizes MS-DIAL data from a combined CSV file.

    Args:
        msdial_csv_path (str): Path to the combined MS-DIAL CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing harmonized MS-DIAL data.
    """

    if ".txt" in msdial_csv_path or ".tsv" in msdial_csv_path:
        sep = "\t"
    elif ".csv" in msdial_csv_path:
        sep = ","
    else:
        raise ValueError("Invalid file type for MS-DIAL master file")

    msdial_df = pd.read_csv(msdial_csv_path, sep=sep)

    possible_spectra_columns = ["MSMS spectrum", "MS/MS spectrum"]
    spectrum_column = msdial_df.columns.intersection(possible_spectra_columns)

    if len(spectrum_column) == 0:
        raise ValueError(
            f"No valid MS/MS spectrum column found in {msdial_csv_path}. "
            "Expected one of: " + ", ".join(possible_spectra_columns)
        )

    msdial_df[spectrum_column[0]] = msdial_df[spectrum_column[0]].apply(
        lambda x: x if x != "null" and is_valid(x) else None
    )
    msdial_df["MS/MS_ASSIGNED"] = msdial_df[spectrum_column[0]].notna()

    ms_spectra = msdial_df[spectrum_column[0]].apply(
        lambda x: (
            [[float(v) for v in i.split(":")] for i in x.split()]
            if x is not None
            else None
        )
    )
    msdial_df["MS/MS_MZS"] = ms_spectra.apply(
        lambda x: ([i[0] for i in x] if isinstance(x, collections.abc.Sequence) else [])
    )
    msdial_df["MS/MS_INTENSITIES"] = ms_spectra.apply(
        lambda x: ([i[1] for i in x] if isinstance(x, collections.abc.Sequence) else [])
    )
    msdial_df_first_half = msdial_df[
        [
            "Alignment ID",
            "Average Rt(min)",
            "Average Mz",
            "Average mobility",
            "Average CCS",
            "Adduct type",
            "MS/MS_ASSIGNED",
            "MS/MS_MZS",
            "MS/MS_INTENSITIES",
        ]
    ]
    msdial_df_first_half.rename(
        columns={
            "Alignment ID": "FEATURE_ID",
            "Average Rt(min)": "RT",
            "Average Mz": "M/Z",
            "Average mobility": "ION_MOBILITY",
            "Average CCS": "CCS",
            "Adduct type": "ADDUCT",
            spectrum_column[0]: "MS/MS_SPECTRUM",
        },
        inplace=True,
    )
    spectrum_column_loc = msdial_df.columns.get_loc(spectrum_column[0])
    msdial_second_half = msdial_df.iloc[:, spectrum_column_loc + 1 :]
    msdial_second_half.drop(
        columns=msdial_second_half.columns.intersection(
            ["MS/MS_ASSIGNED", "MS/MS_MZS", "MS/MS_INTENSITIES"]
        ),
        inplace=True,
    )
    msdial_output = pd.concat([msdial_df_first_half, msdial_second_half], axis=1)
    msdial_output["ADDUCT"] = msdial_output["ADDUCT"].apply(
        lambda x: (
            "[" + x + "]" if is_valid(x) and (not x.startswith("[") and x != "") else x
        )
    )
    return msdial_output


def harmonize_msdial_single_sample(msdial_txt_path: str) -> pd.DataFrame:
    if msdial_txt_path.endswith("txt") or msdial_txt_path.endswith("tsv"):
        msdial_df = pd.read_csv(msdial_txt_path, sep="\t")
    elif msdial_txt_path.endswith("csv"):
        msdial_df = pd.read_csv(msdial_txt_path)
    else:
        raise ValueError("Invalid file type")

    msdial_df["MS/MS_ASSIGNED"] = msdial_df["MSMS spectrum"].notna()
    msdial_df["MSMS spectrum"] = msdial_df["MSMS spectrum"].apply(
        lambda x: x if x != "null" and is_valid(x) else None
    )

    ms_spectra = msdial_df["MSMS spectrum"].apply(
        lambda x: (
            [[float(v) for v in i.split(":")] for i in x.split()]
            if x is not None
            else None
        )
    )
    msdial_df["MS/MS_MZS"] = ms_spectra.apply(
        lambda x: ([i[0] for i in x] if isinstance(x, collections.abc.Sequence) else [])
    )
    msdial_df["MS/MS_INTENSITIES"] = ms_spectra.apply(
        lambda x: ([i[1] for i in x] if isinstance(x, collections.abc.Sequence) else [])
    )

    msdial_df = msdial_df[
        [
            "Peak ID",
            "RT (min)",
            "Precursor m/z",
            "Mobility",
            "CCS",
            "Adduct",
            "MS/MS_ASSIGNED",
            "MS/MS_MZS",
            "MS/MS_INTENSITIES",
            "Height",
        ]
    ]
    msdial_df.rename(
        columns={
            "Peak ID": "FEATURE_ID",
            "RT (min)": "RT",
            "Precursor m/z": "M/Z",
            "Mobility": "ION_MOBILITY",
            "CCS": "CCS",
            "Adduct": "ADDUCT",
            "MSMS spectrum": "MS/MS_SPECTRUM",
            "Height": msdial_txt_path.split("/")[-1].split(".")[0],
        },
        inplace=True,
    )
    msdial_df["ADDUCT"] = msdial_df["ADDUCT"].apply(
        lambda x: (
            "[" + x + "]" if is_valid(x) and (not x.startswith("[") and x != "") else x
        )
    )
    return msdial_df
