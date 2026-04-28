# -*- coding: utf-8 -*-

"""Code to harmonize raw files and prepare them for analysis."""

import os

from benchmarking.harmonizer.mzmine import harmonize_mzmine
from benchmarking.harmonizer.metaboscape import harmonize_metaboscape
from benchmarking.harmonizer.msdial import (
    harmonize_msdial_multi_sample,
    harmonize_msdial_multi_sample_combined,
    harmonize_msdial_single_sample,
)


HARMONIZATION_FUNCTIONS = {
    "mzmine": harmonize_mzmine,
    "metaboscape": harmonize_metaboscape,
    "msdial-multi": harmonize_msdial_multi_sample,
    "msdial-multi-combined": harmonize_msdial_multi_sample_combined,
    "msdial-single": harmonize_msdial_single_sample,
}


def harmonize(
    input_directory,
    input_tool_type,
    output_path,
    output_type="pq",
) -> None:
    """
    Main function to harmonize raw files from different tools and prepare them for analysis.

    :param input_directory: Path to the directory containing the raw files to be harmonized.
    :param input_tool_type: Type of the input tool (e.g., 'mzmine', 'metaboscape', 'msdial-single', 'msdial-multi', 'msdial-multi-combined').
    :param output_path: Path to save the harmonized output file.
    :param output_type: Format of the output file ('pq' for Parquet, 'csv' for CSV, 'tsv' for TSV). Default is 'pq'.
    :return: None
    """
    harmonization_function = HARMONIZATION_FUNCTIONS.get(input_tool_type, None)

    if harmonization_function is None:
        raise ValueError(
            f"Harmonization for tool type '{input_tool_type}' is not implemented."
        )

    if output_type not in ["pq", "csv", "tsv"]:
        raise ValueError(f"Output type must be one of: 'pq', 'csv', 'tsv'")

    match input_tool_type:
        case "mzmine" | "metaboscape":
            quant_table_file = [
                f for f in os.listdir(input_directory) if f.endswith(".csv")
            ][0]
            mgf_file = [f for f in os.listdir(input_directory) if f.endswith(".mgf")][0]

            if not quant_table_file:
                raise ValueError(
                    f"No CSV file found in {input_directory} for {input_tool_type} harmonization."
                )
            if not mgf_file:
                raise ValueError(
                    f"No MGF file found in {input_directory} for {input_tool_type} harmonization."
                )

            quant_table_path = os.path.join(input_directory, quant_table_file)
            mgf_path = os.path.join(input_directory, mgf_file)

            harmonized_df = harmonization_function(mgf_path, quant_table_path)

        case "msdial-multi":
            msdial_master_file = [
                f for f in os.listdir(input_directory) if "Master" in f
            ][0]
            msdial_peak_file = [
                f for f in os.listdir(input_directory) if "Values" in f
            ][0]

            if not msdial_master_file:
                raise ValueError(
                    f"Peak Master file not found in {input_directory} for {input_tool_type} harmonization."
                )
            if not msdial_peak_file:
                raise ValueError(
                    f"Peak Values file not found in {input_directory} for {input_tool_type} harmonization."
                )

            msdial_master_path = os.path.join(input_directory, msdial_master_file)
            msdial_peak_path = os.path.join(input_directory, msdial_peak_file)

            harmonized_df = harmonize_msdial_multi_sample(
                msdial_master_path, msdial_peak_path
            )

        case "msdial-multi-combined":
            msdial_csv_file = [
                f
                for f in os.listdir(input_directory)
                if f.endswith(".csv") and "Height" in f
            ][0]

            msdial_csv_path = os.path.join(input_directory, msdial_csv_file)

            harmonized_df = harmonize_msdial_multi_sample_combined(msdial_csv_path)

        case "msdial-single":
            msdial_txt_file = [
                f
                for f in os.listdir(input_directory)
                if f.endswith(".txt") or f.endswith(".tsv") or f.endswith(".csv")
            ][0]

            if not msdial_txt_file:
                raise ValueError(
                    f"No TXT/TSV/CSV file found in {input_directory} for {input_tool_type} harmonization."
                )

            msdial_txt_path = os.path.join(input_directory, msdial_txt_file)

            harmonized_df = harmonize_msdial_single_sample(msdial_txt_path)

    if output_type == "pq":
        harmonized_df.to_parquet(output_path, index=False, engine="pyarrow")
    elif output_type == "csv":
        harmonized_df.to_csv(output_path, index=False)
    elif output_type == "tsv":
        harmonized_df.to_csv(output_path, sep="\t", index=False)
