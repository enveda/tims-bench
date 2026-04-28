# -*- coding: utf-8 -*-

"""Constants used in the repo."""


NUMBER_FILTERED_PEAKS = "num_filtered_peaks"

FEATURE_ID_COLUMN = "FEATURE_ID"
LIBRARY_ID_COLUMN = "LIBRARY_ID"
LIBRARY_PRECURSOR_MZ_COLUMN = "LIBRARY_PRECURSOR_MZ"
LIBRARY_MZS_COLUMN = "LIBRARY_MZS"
LIBRARY_INTENSITIES_COLUMN = "LIBRARY_INTENSITIES"
METHOD_COLUMN = "METHOD"
INCHIKEY_COLUMN = "INCHIKEY"
SCORE_COLUMN = "SCORE"

TOOL_COLORS = {
    "metaboscape": "#D46428",
    "mzmine": "#0074B3",
    "msdial": "#009F75",
    "MetaboScape": "#D46428",
    "MZmine": "#0074B3",
    "MS-DIAL": "#009F75",
}

TOOL_NAMES = {
    "metaboscape": "MetaboScape",
    "mzmine": "MZmine",
    "msdial": "MS-DIAL",
}

DATASET_DESCRIPTORS = {
    "MSV000090327": "MSV000090327",
    "MSV000091642": "MSV000091642",
    "MSV000095813": "MSV000095813",
    "MSV000097967": "MSV000097967",
    "MSV000097015": "MSV000097015",
    "MSV000096291": "MSV000096291",
    "MSV000096189": "MSV000096189",
    "ST002402": "ST002402",
    "MSV000084402": "MSV000084402",
    "MTBLS12332": "MTBLS12332",
    "MSV000098263": "Reframe Drug Library",
    "plant_spikein": "Individual Plant Natural Products",
    "NIST_SRM": "NIST SRM 1950",
}

# Color dict for annotation methods
ANNOTATION_COLORS = {
    "Spectral Entropy": "#E87407",
    "Cosine Similarity": "#03888F",
    "DreaMS Similarity": "#BDBD00",
}

# Formatted names for annotation methods
ANNOTATION_NAMES = {
    "annotated_spectral_entropy": "Spectral Entropy",
    "annotated_cosine_similarity": "Cosine Similarity",
    "annotated_dreams_similarity": "DreaMS Similarity",
}


# Transform settings: (tic_norm, transform_type)
TRANSFORM_SETTINGS = {
    "rclr": (True, "rclr"),
    "clr": (True, "clr"),
    "raw": (False, None),
}
