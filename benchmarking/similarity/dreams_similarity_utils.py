# -*- coding: utf-8 -*-

"""Util functions to perform dreams similarity calculations."""


def df_to_feature_id_dict(
    df,
    feature_id_column="FEATURE_ID",
    ms2_column="MS/MS_ASSIGNED",
    mzs_column="MS/MS_MZS",
    intensities_column="MS/MS_INTENSITIES",
    precursor_mz_column="M/Z",
):
    if ms2_column is not None:
        df = df[df[ms2_column] == True]
    spectra = df.apply(
        lambda x: (x[mzs_column], x[intensities_column]), axis=1
    ).tolist()
    precursors = df[precursor_mz_column].tolist()
    feature_ids = df[feature_id_column].tolist()
    feature_id_dict = {
        feature_id: (precursor, spectrum)
        for feature_id, precursor, spectrum in zip(feature_ids, precursors, spectra)
    }
    return feature_id_dict


def feature_id_dict_to_mgf(feature_id_dict, filename):
    """
    Converts a dictionary of FEATURE_IDs to an MGF file format.

    Parameters:
    feature_id_dict (dict): Dictionary with keys as FEATURE_IDs and values as lists of spectra.
    filename (str): The name of the output MGF file.

    Returns:
    None
    """
    with open(filename, "w") as f:
        for feature_id, (precursor, spectrum) in feature_id_dict.items():
            f.write("BEGIN IONS\n")
            f.write(f"FEATURE_ID={feature_id}\n")
            f.write(f"MS2_LEVEL=2\n")
            f.write(f"PEPMASS={precursor}\n")
            if type(spectrum[0]) is not float and type(spectrum[0]) is not None:
                for i in range(len(spectrum[0])):
                    mz = spectrum[0][i]
                    intensity = spectrum[1][i]
                    f.write(f"{mz} {intensity}\n")
            f.write("END IONS\n")
            f.write("\n")


def mgf_to_feature_ids(filename) -> dict:
    """
    Parses an MGF file and returns a list of FEATURE_IDs
    """
    feature_ids = []
    with open(filename, "r") as f:
        feature_id = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                feature_id = None
            elif line == "END IONS":
                if feature_id is not None:
                    feature_ids.append(feature_id)
            elif "=" in line:
                key, value = line.split("=", 1)
                if key == "FEATURE_ID":
                    try:
                        feature_id = int(value)
                    except ValueError:
                        feature_id = None
    return feature_ids
