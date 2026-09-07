# -*- coding: utf-8 -*-

"""Resolve InChIKeys to SMILES using local resources with an optional PubChem fallback."""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd

from benchmarking.constants import (
    INCHIKEY_COLUMN,
    REFRAME_SMILES_PATH,
    SMILES_CACHE_PATH,
    SMILES_COLUMN,
)

LOGGER = logging.getLogger(__name__)


class PubChemUnavailableError(RuntimeError):
    """Raised when a PubChem request fails, as opposed to returning no match."""


INCHIKEY14_LENGTH = 14
PUBCHEM_BATCH_SIZE = 50
PUBCHEM_REQUEST_PAUSE_SECONDS = 0.25
PUBCHEM_REQUEST_TIMEOUT_SECONDS = 15.0
# Stop querying after this many batches fail in a row - the network is very likely unavailable
PUBCHEM_MAX_CONSECUTIVE_FAILURES = 3
PUBCHEM_PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{keys}/property/{properties}/CSV"
)
# PubChem renamed "CanonicalSMILES" to "SMILES"; try the current name first and fall back.
PUBCHEM_SMILES_PROPERTIES = ("SMILES", "IsomericSMILES", "CanonicalSMILES")


def load_smiles_cache(cache_path=SMILES_CACHE_PATH) -> Dict[str, str]:
    """Load the local InChIKey -> SMILES cache, returning an empty mapping when absent."""
    path = Path(cache_path)
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def save_smiles_cache(cache: Dict[str, str], cache_path=SMILES_CACHE_PATH) -> None:
    """Write the InChIKey -> SMILES cache back to disk, sorted for a stable diff."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)


def load_local_smiles_mapping(mapping_path=REFRAME_SMILES_PATH) -> Dict[str, str]:
    """Build an InChIKey -> SMILES mapping from the ReFRAME compound list.

    Returns an empty mapping (with a warning) when the file is not available, so callers
    can still fall back to the cache and PubChem.
    """
    path = Path(mapping_path)
    if not path.exists():
        LOGGER.warning("Local SMILES mapping not found at %s; skipping it", path)
        return {}

    mapping_df = pd.read_csv(path, usecols=["canonical_ikey", "smile"]).dropna(
        subset=["canonical_ikey", "smile"]
    )
    return dict(zip(mapping_df["canonical_ikey"], mapping_df["smile"], strict=True))


def _build_inchikey14_index(*mappings: Dict[str, str]) -> Dict[str, str]:
    """Index SMILES by the 14-character InChIKey skeleton block.

    Skeleton matches ignore stereochemistry, so a hit can be a different stereoisomer of the
    queried compound. Earlier mappings win over later ones.
    """
    index: Dict[str, str] = {}
    for mapping in mappings:
        for inchikey, smiles in mapping.items():
            prefix = inchikey[:INCHIKEY14_LENGTH]
            if prefix not in index and isinstance(smiles, str) and smiles:
                index[prefix] = smiles
    return index


def _parse_pubchem_csv(payload: str) -> Dict[str, str]:
    """Map the 14-character InChIKey block to SMILES from a PubChem property CSV response."""
    rows = list(csv.DictReader(payload.splitlines()))
    smiles_by_prefix: Dict[str, str] = {}
    for row in rows:
        inchikey = row.get("InChIKey") or ""
        smiles = next(
            (row[key] for key in PUBCHEM_SMILES_PROPERTIES if row.get(key)),
            "",
        )
        if inchikey and smiles:
            smiles_by_prefix[inchikey[:INCHIKEY14_LENGTH]] = smiles
    return smiles_by_prefix


def _query_pubchem_batch(
    inchikeys: list, timeout: float = PUBCHEM_REQUEST_TIMEOUT_SECONDS
) -> Dict[str, str]:
    """Fetch SMILES for one batch of InChIKeys from PubChem.

    Returns the SMILES found, keyed by the 14-character InChIKey block. An empty mapping means
    PubChem knows none of these keys; a failed request raises instead, so the caller can tell
    "no match" apart from "network down".

    Raises:
        PubChemUnavailableError: If the request fails or no SMILES property name is accepted.
    """
    for smiles_property in PUBCHEM_SMILES_PROPERTIES:
        url = PUBCHEM_PROPERTY_URL.format(
            keys=quote(",".join(inchikeys)),
            properties=f"InChIKey,{smiles_property}",
        )
        try:
            with urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed PubChem host
                return _parse_pubchem_csv(response.read().decode("utf-8"))
        except HTTPError as error:
            # 400 usually means this PubChem release does not know the property name; 404 means
            # none of the keys in the batch exist. Both are answers, not transport failures.
            if error.code == 400:
                LOGGER.debug("PubChem rejected property %s; trying the next name", smiles_property)
                continue
            if error.code == 404:
                return {}
            raise PubChemUnavailableError(
                f"PubChem returned {error.code} for a batch of {len(inchikeys)} InChIKeys"
            ) from error
        except (URLError, TimeoutError) as error:
            raise PubChemUnavailableError(
                f"PubChem unreachable for a batch of {len(inchikeys)} InChIKeys"
            ) from error

    raise PubChemUnavailableError(
        f"No PubChem SMILES property accepted for a batch of {len(inchikeys)} InChIKeys"
    )


def resolve_smiles_from_inchikeys(
    inchikeys: Iterable[str],
    *,
    cache_path=SMILES_CACHE_PATH,
    local_mapping_path=REFRAME_SMILES_PATH,
    use_pubchem: bool = True,
    match_inchikey14: bool = True,
    batch_size: int = PUBCHEM_BATCH_SIZE,
    request_timeout: float = PUBCHEM_REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Optional[str]]:
    """Resolve InChIKeys to SMILES, preferring local resources over the network.

    Lookup order per key: the on-disk cache, the ReFRAME compound list, an optional
    stereochemistry-insensitive match on the 14-character InChIKey block, and finally PubChem.
    Keys resolved from PubChem are written back to the cache so later runs stay offline.

    Args:
        inchikeys: InChIKeys to resolve. Duplicates and missing values are ignored.
        cache_path: JSON file used to persist resolved InChIKey -> SMILES pairs.
        local_mapping_path: ReFRAME CSV holding ``canonical_ikey`` and ``smile`` columns.
        use_pubchem: Query PubChem for keys that no local resource covers. Set ``False``
            for a fully offline run. Repeated request failures abort the PubChem pass (after
            ``PUBCHEM_MAX_CONSECUTIVE_FAILURES``) and leave the remaining keys unresolved.
        match_inchikey14: Allow matches on the 14-character skeleton block. These ignore
            stereochemistry, so the SMILES can be a different stereoisomer.
        batch_size: Number of InChIKeys per PubChem request.
        request_timeout: Per-request PubChem timeout, in seconds.

    Returns:
        A mapping of every input InChIKey to its SMILES, or to ``None`` when unresolved.
    """
    unique_keys = sorted({key for key in inchikeys if isinstance(key, str) and key})
    if not unique_keys:
        return {}

    cache = load_smiles_cache(cache_path)
    local_mapping = load_local_smiles_mapping(local_mapping_path)

    resolved: Dict[str, str] = {}
    for key in unique_keys:
        smiles = cache.get(key) or local_mapping.get(key)
        if isinstance(smiles, str) and smiles:
            resolved[key] = smiles
    n_exact = len(resolved)

    if match_inchikey14:
        inchikey14_index = _build_inchikey14_index(cache, local_mapping)
        for key in unique_keys:
            if key not in resolved:
                smiles = inchikey14_index.get(key[:INCHIKEY14_LENGTH])
                if smiles:
                    resolved[key] = smiles
    n_skeleton = len(resolved) - n_exact

    missing = [key for key in unique_keys if key not in resolved]
    n_from_pubchem = 0
    if missing and use_pubchem:
        LOGGER.info("Resolving %d InChIKeys via PubChem", len(missing))
        n_consecutive_failures = 0
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            try:
                smiles_by_prefix = _query_pubchem_batch(batch, timeout=request_timeout)
            except PubChemUnavailableError as error:
                n_consecutive_failures += 1
                LOGGER.warning("%s", error)
                if n_consecutive_failures >= PUBCHEM_MAX_CONSECUTIVE_FAILURES:
                    LOGGER.warning(
                        "Giving up on PubChem after %d failed requests; "
                        "%d InChIKeys were not queried",
                        n_consecutive_failures,
                        len(missing) - start - len(batch),
                    )
                    break
                continue
            n_consecutive_failures = 0

            for key in batch:
                smiles = smiles_by_prefix.get(key[:INCHIKEY14_LENGTH])
                if smiles:
                    resolved[key] = smiles
                    cache[key] = smiles
                    n_from_pubchem += 1
            if start + batch_size < len(missing):
                time.sleep(PUBCHEM_REQUEST_PAUSE_SECONDS)

        if n_from_pubchem:
            save_smiles_cache(cache, cache_path)

    n_unresolved = len(unique_keys) - len(resolved)
    LOGGER.info(
        "Resolved %d/%d InChIKeys (%d exact, %d skeleton-only, %d from PubChem); %d unresolved",
        len(resolved),
        len(unique_keys),
        n_exact,
        n_skeleton,
        n_from_pubchem,
        n_unresolved,
    )

    return {key: resolved.get(key) for key in unique_keys}


def add_smiles_column(
    feature_table: pd.DataFrame,
    *,
    inchikey_column: str = INCHIKEY_COLUMN,
    smiles_column: str = SMILES_COLUMN,
    smiles_map: Optional[Dict[str, Optional[str]]] = None,
    **resolver_kwargs,
) -> pd.DataFrame:
    """Return a copy of ``feature_table`` with a SMILES column derived from its InChIKeys.

    Args:
        feature_table: Table holding an InChIKey column.
        inchikey_column: Name of the InChIKey column to map from.
        smiles_column: Name of the SMILES column to add. An existing column is overwritten.
        smiles_map: Pre-resolved InChIKey -> SMILES mapping. Pass one when adding the column
            to several tables so the keys are resolved (and PubChem queried) only once.
        **resolver_kwargs: Forwarded to :func:`resolve_smiles_from_inchikeys` when
            ``smiles_map`` is not given.

    Raises:
        KeyError: If ``inchikey_column`` is not a column of ``feature_table``.
    """
    if inchikey_column not in feature_table.columns:
        raise KeyError(
            f"Column {inchikey_column!r} not found in the table; "
            f"available columns: {list(feature_table.columns)}"
        )

    if smiles_map is None:
        smiles_map = resolve_smiles_from_inchikeys(
            feature_table[inchikey_column].dropna(), **resolver_kwargs
        )

    annotated_table = feature_table.copy()
    annotated_table[smiles_column] = annotated_table[inchikey_column].map(smiles_map)
    return annotated_table


def add_smiles_to_feature_tables(
    feature_tables: Dict[str, pd.DataFrame],
    *,
    inchikey_column: str = INCHIKEY_COLUMN,
    smiles_column: str = SMILES_COLUMN,
    **resolver_kwargs,
) -> Dict[str, pd.DataFrame]:
    """Add a SMILES column to every table in a ``tool -> DataFrame`` mapping.

    All InChIKeys are resolved in a single pass so the shared cache (and any PubChem
    request) is used once for the whole set rather than once per tool.
    """
    all_inchikeys = set()
    for tool, feature_table in feature_tables.items():
        if inchikey_column not in feature_table.columns:
            raise KeyError(
                f"Column {inchikey_column!r} not found in the table for {tool!r}; "
                f"available columns: {list(feature_table.columns)}"
            )
        all_inchikeys.update(feature_table[inchikey_column].dropna())

    smiles_map = resolve_smiles_from_inchikeys(all_inchikeys, **resolver_kwargs)

    return {
        tool: add_smiles_column(
            feature_table,
            inchikey_column=inchikey_column,
            smiles_column=smiles_column,
            smiles_map=smiles_map,
        )
        for tool, feature_table in feature_tables.items()
    }
