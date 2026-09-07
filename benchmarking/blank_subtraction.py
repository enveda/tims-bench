# -*- coding: utf-8 -*-

"""Drop the features that a run's blanks can account for."""

import logging

import pandas as pd

from benchmarking.constants import (
    INCHIKEY_COLUMN,
    LIBRARY_ID_COLUMN,
    LIBRARY_INTENSITIES_COLUMN,
    LIBRARY_MZS_COLUMN,
    LIBRARY_PRECURSOR_MZ_COLUMN,
    METHOD_COLUMN,
    SCORE_COLUMN,
)

LOGGER = logging.getLogger(__name__)

# Every column that is not a per-run intensity measurement. These are carried through
# untouched; each remaining column is either a sample column or is dropped with the blanks.
FEATURE_METADATA_COLUMNS = (
    "FEATURE_ID",
    "M/Z",
    "RT",
    "ION_MOBILITY",
    "CCS",
    "ADDUCT",
    "MS/MS_ASSIGNED",
    "MS/MS_MZS",
    "MS/MS_INTENSITIES",
    # Present once load_all_feature_tables has merged the annotations in.
    LIBRARY_ID_COLUMN,
    METHOD_COLUMN,
    INCHIKEY_COLUMN,
    SCORE_COLUMN,
    LIBRARY_PRECURSOR_MZ_COLUMN,
    LIBRARY_MZS_COLUMN,
    LIBRARY_INTENSITIES_COLUMN,
)

SAMPLE_PREFIX = "SAMPLE_"
BLANK_PREFIX = "MSBLANK_"
BLANK_RATIO_THRESHOLD = 5.0
RATIO_EPSILON = 1e-9  # keeps a feature absent from the blanks from dividing by zero


def blank_subtract_feature_table(
    feature_table: pd.DataFrame,
    *,
    sample_prefix: str = SAMPLE_PREFIX,
    blank_prefix: str = BLANK_PREFIX,
    ratio_threshold: float = BLANK_RATIO_THRESHOLD,
    metadata_columns: tuple = FEATURE_METADATA_COLUMNS,
) -> tuple:
    """Keep only the features that stand above the blanks, and only the sample columns.

    A feature is kept when its mean sample intensity is at least ``ratio_threshold`` times its
    mean blank intensity, and when it carries some intensity in at least one sample. Intensity
    columns that are neither samples nor blanks (QC pools, water blanks, shutdown runs, ...)
    are dropped along with the blanks, so the result is the feature metadata plus the sample
    columns.

    Despite the name this filters features rather than editing intensities - no value is
    altered, features are either kept whole or dropped. Nothing here is dataset-specific: the
    split is driven entirely by the two column prefixes, so a table whose runs are named
    differently only needs different prefixes passed in.

    Parameters
    ----------
    feature_table: pd.DataFrame
        A harmonized feature table, with or without annotations merged in.
    sample_prefix: str
        Prefix identifying the sample intensity columns. Default ``"SAMPLE_"``.
    blank_prefix: str
        Prefix identifying the blank intensity columns. Default ``"MSBLANK_"``. If the table
        has no such columns the ratio filter is skipped.
    ratio_threshold: float
        Minimum mean-sample / mean-blank intensity ratio for a feature to be kept.
    metadata_columns: tuple
        Non-intensity columns to carry through. Missing ones are ignored.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        The filtered table, and a summary of how many features and columns were dropped.

    Raises
    ------
    ValueError
        If no column starts with ``sample_prefix``, which means this table names its runs
        differently - failing here rather than returning an empty table.
    """
    columns = list(feature_table.columns)
    # Only per-run intensity columns are candidates, so a prefix that happens to also match a
    # metadata column (e.g. "L" against LIBRARY_ID) cannot be pulled into the sample means.
    run_columns = [col for col in columns if col not in set(metadata_columns)]
    sample_columns = [col for col in run_columns if col.startswith(sample_prefix)]
    blank_columns = [col for col in run_columns if col.startswith(blank_prefix)]

    if not sample_columns:
        raise ValueError(
            f"No columns start with {sample_prefix!r}. This table names its runs differently - "
            f"pass a sample_prefix/blank_prefix that matches it."
        )

    feature_table[sample_columns] = feature_table[sample_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    feature_table[sample_columns] = feature_table[sample_columns].mask(
        feature_table[sample_columns] < 1000, 0
    )

    # A feature with no measured intensity in any sample cannot be judged against the blanks.
    has_sample_signal = feature_table[sample_columns].gt(0).any(axis=1)
    mean_samples = feature_table[sample_columns].mean(axis=1)

    if blank_columns:
        blanks = feature_table[blank_columns].apply(pd.to_numeric, errors="coerce")
        # A feature absent from every blank averages to NaN, which means "no blank signal".
        mean_blanks = blanks.mean(axis=1).fillna(0)
        above_blanks = (mean_samples / (mean_blanks + RATIO_EPSILON)) >= ratio_threshold
    else:
        LOGGER.warning(
            "No columns start with %r; keeping every feature that has sample signal",
            blank_prefix,
        )
        above_blanks = pd.Series(True, index=feature_table.index)

    keep = has_sample_signal & above_blanks
    kept_columns = [col for col in metadata_columns if col in columns] + sample_columns

    summary = pd.Series(
        {
            "n_features_in": len(feature_table),
            "n_sample_columns": len(sample_columns),
            "n_blank_columns": len(blank_columns),
            "n_columns_dropped": len(columns) - len(kept_columns),
            "dropped_no_sample_signal": int((~has_sample_signal).sum()),
            "dropped_at_or_below_blanks": int(
                (has_sample_signal & ~above_blanks).sum()
            ),
            "n_features_out": int(keep.sum()),
            "features_kept_%": (
                round(100 * keep.mean(), 1) if len(feature_table) else 0.0
            ),
        }
    )
    return feature_table.loc[keep, kept_columns], summary
