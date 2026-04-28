# -*- coding: utf-8 -*-

from typing import Dict, Union

"""Code to prepare feature data for different file formats."""


def mgf_to_feature_peaks_dict(
    filename, with_precursors=False
) -> Union[Dict[str, list], Dict[str, float]]:
    """
    Parses an MGF file to generate feature dictionaries.
    Only spectra with a valid FEATURE_ID are included.

    :param filename: Path to the MGF file to be parsed.
    :param with_precursors: Whether to include precursor m/z values in the output.
    :return: A dictionary mapping FEATURE_ID (int) to peaks (list of [mz, intensity]).
    """
    feature_dict = {}

    feature_id = None
    precursor_mz = None
    peaks = []
    precursors = {}

    file_lines = open(filename, "r").readlines()

    for line in file_lines:
        line = line.strip()
        if not line:
            continue

        if line == "BEGIN IONS":
            feature_id = None
            peaks = []

        elif line == "END IONS":
            if feature_id is not None:
                feature_dict[feature_id] = peaks
                precursors[feature_id] = precursor_mz

            # else: skip spectra without FEATURE_ID

        elif "=" in line:
            key, value = line.split("=", 1)
            if key == "FEATURE_ID":
                try:
                    feature_id = int(value)
                except ValueError:
                    feature_id = value.strip()  # Keep as string if not int

            if key == "PEPMASS":
                try:
                    precursor_mz = float(value.strip())
                except ValueError:
                    precursor_mz = value.strip()  # Keep as string if not float

        else:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz, intensity = map(float, parts[:2])
                    peaks.append([mz, intensity])
                except ValueError:
                    pass  # Ignore lines that can't be parsed as numbers

    if with_precursors:
        return feature_dict, precursors

    return feature_dict


def msp_to_feature_peaks_dict(filename) -> Dict[str, list]:
    """
    Parses an MSP file to generate feature dictionaries.
    Only spectra with a valid FEATURE_ID are included.

    :param filename: Path to the MSP file to be parsed.
    :return: A dictionary mapping FEATURE_ID (int) to peaks (list of [mz, intensity]).
    """
    feature_dict = {}

    with open(filename, "r") as f:
        feature_id = None
        peaks = []
        num_peaks = None

        while True:

            line = f.readline()

            if not line:
                break

            if "NAME:" in line:
                feature_id = line.split("NAME:")[1].strip()
                peaks = []

            elif "Num Peaks:" in line:
                num_peaks = int(line.split("Num Peaks:")[1].strip())
                for _ in range(num_peaks):
                    peak_line = f.readline().strip()
                    if peak_line:
                        parts = peak_line.split()
                        if len(parts) >= 2:
                            try:
                                mz, intensity = map(float, parts[:2])
                                peaks.append([mz, intensity])
                            except ValueError:
                                pass  # Ignore lines that can't be parsed as numbers

                if feature_id is not None:
                    try:
                        feature_id_int = int(feature_id)
                        feature_dict[feature_id_int] = peaks
                    except ValueError:
                        feature_dict[feature_id] = peaks

    return feature_dict
