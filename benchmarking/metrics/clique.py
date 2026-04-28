# -*- coding: utf-8 -*-

"""Module for computing clique related metrics."""

import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd

from ms_entropy import clean_spectrum

from benchmarking.similarity.modified_flashier_entropy import (
    modified_flashier_entropy_search,
)
from benchmarking.utils import filter_spectrum_peaks_in_df, to_typed_bank


class MinimumCliqueCover:
    def __init__(self, adj_list):
        # adj_list: list of sets, (self-loops will be removed)

        # adjacency lists as sets for original graph
        adj_indices = [i for i in range(len(adj_list))]
        for i in adj_indices:
            if i in adj_list[i]:
                adj_list[i].remove(i)

        updated_indices = []
        updated_adj_list = []
        for i in adj_indices:
            if len(adj_list[i]) > 0:
                updated_indices.append(i)
                updated_adj_list.append(adj_list[i])

        # Reindex adjacency lists
        index_map = {
            old_idx: new_idx for new_idx, old_idx in enumerate(updated_indices)
        }
        for i in range(len(updated_adj_list)):
            updated_adj_list[i] = {
                index_map[j] for j in updated_adj_list[i] if j in index_map
            }

        self.adj_ind = list(range(len(updated_adj_list)))
        self.adj = updated_adj_list
        self.singleton_count = len(adj_list) - len(updated_adj_list)

    def greedy_clique_cover(self) -> list:
        # Peel off large cliques in original graph
        G = self.adj
        remaining = set(self.adj_ind)
        cover = []

        while remaining:
            # start clique with highest-degree vertex
            v0 = max(remaining, key=lambda u: len(G[u] & remaining))
            C = {v0}
            # candidate set: common neighbors of clique in remaining
            Cand = G[v0] & remaining
            while True:

                if len(Cand) == 0:
                    break
                # pick u in Cand with max neighbors within Cand
                u = max(Cand, key=lambda x: len(G[x] & Cand))
                C.add(u)
                # update candidates
                Cand &= G[u]
            cover.append(C)
            # remove clique from remaining
            for u in C:
                remaining.remove(u)
            # optionally: remove edges (not needed for set logic)
        return cover


def compute_clique_metrics(sample_dataset_path):
    harmonized_path = os.path.join(sample_dataset_path, "harmonized")

    tool_clique_metrics = {}
    for file in os.listdir(harmonized_path):
        if "harmonized" in file:
            feature_table = pd.read_parquet(os.path.join(harmonized_path, file))
        else:
            continue

        if "mzmine" in file.lower():
            tool = "mzmine"
        elif "msdial" in file.lower():
            tool = "msdial"
        elif "metaboscape" in file.lower():
            tool = "metaboscape"
        else:
            tool = file.split("_")[-2]

        feature_table = feature_table[feature_table["MS/MS_ASSIGNED"] == True]

        feature_table_filtered = filter_spectrum_peaks_in_df(
            df=feature_table,
            column_intensities="MS/MS_INTENSITIES",
            column_mzs="MS/MS_MZS",
            min_abs_intensity=10.0,
            min_rel_intensity=0.01,
        )

        feature_table_filtered = feature_table_filtered.sort_values(by="M/Z")

        query_spectra = feature_table_filtered.apply(
            lambda x: np.column_stack((x["MS/MS_MZS"], x["MS/MS_INTENSITIES"])), axis=1
        ).to_numpy()

        query_spectra = [clean_spectrum(spectrum) for spectrum in query_spectra]
        query_spectra = np.array(query_spectra, dtype=object)
        query_precursors = feature_table_filtered["M/Z"].to_numpy()

        query_precursors_reshaped = np.asarray(
            query_precursors, dtype=np.float32, order="C"
        ).reshape(-1)
        query_bank = to_typed_bank(query_spectra)

        indices_x, indices_y, scores = modified_flashier_entropy_search(
            query_bank,
            query_precursors_reshaped,
            query_bank,
            query_precursors_reshaped,
            ms1_ppm_tolerance=20,  # 00000000,
            ms2_da_tolerance=0.05,
            min_matched_peaks=3,
        )

        ccs_x = feature_table_filtered["CCS"].to_numpy()[indices_x]
        ccs_y = feature_table_filtered["CCS"].to_numpy()[indices_y]

        rt_x = feature_table_filtered["RT"].to_numpy()[indices_x]
        rt_y = feature_table_filtered["RT"].to_numpy()[indices_y]

        ccs_mask = (
            np.abs(ccs_x - ccs_y) / np.maximum(np.minimum(ccs_x, ccs_y), 1e-8) <= 0.01
        )  # CCS difference <= 1%
        rt_mask = np.abs(rt_x - rt_y) <= 0.1  # RT difference <= 0.1 min
        scores_mask = scores >= 0.7  # score >= 0.7
        combined_mask = ccs_mask & rt_mask & scores_mask

        index_mask = indices_x != indices_y
        self_match_mask = index_mask & combined_mask
        self_match_count = np.sum(self_match_mask)

        matching_indices_x = indices_x[combined_mask]
        matching_indices_y = indices_y[combined_mask]

        # Create adjacency list
        adjacency_list = {}
        for i in range(len(feature_table_filtered)):
            adjacency_list[i] = set()

        for x, y in zip(matching_indices_x, matching_indices_y):
            adjacency_list[x].add(y)
            adjacency_list[y].add(x)

        vertices = list(adjacency_list.keys())
        corresponding_sets = [adjacency_list[v] for v in vertices]
        clique_clusterer = MinimumCliqueCover(corresponding_sets)
        cliques = clique_clusterer.greedy_clique_cover()
        singleton_count = clique_clusterer.singleton_count

        # Can alternatively plot boxplot of clique sizes
        clique_sizes = np.array([len(c) for c in cliques] + [1] * singleton_count)

        group_clique_count = len(clique_sizes[clique_sizes > 1])
        total_clique_count = len(clique_sizes)
        max_clique_size = np.max(clique_sizes)
        tool_clique_metrics[tool] = {
            "ms2_feature_count": len(feature_table),
            "self_match_count": self_match_count,
            "group_clique_count": group_clique_count,
            "total_clique_count": total_clique_count,
            "max_clique_size": max_clique_size,
            "clique_sizes": clique_sizes,
        }
    return tool_clique_metrics


def load_clique_metrics(cache_path: str, input_dataset_directories: list):
    """Load or compute clique metrics for all datasets.
    If cache_path is provided and file exists, loads metrics from cache.
    Otherwise, computes metrics for all datasets and saves to cache if path provided.
    """

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            all_dataset_metrics = json.load(f)
        print(f"Loaded clique metrics from cache: {cache_path}")
    else:
        all_dataset_metrics = {}

        # Compute metrics for datasets not in cache
        datasets_to_compute = []
        for dataset_path in input_dataset_directories:
            dataset_id = dataset_path.strip("/").split("/")[-1]
            datasets_to_compute.append(dataset_path)

        print(f"Computing clique metrics for {len(datasets_to_compute)} datasets...")
        for dataset_path in datasets_to_compute:
            dataset_id = dataset_path.strip("/").split("/")[-1]
            try:
                metrics = compute_clique_metrics(dataset_path)
                # Convert to serializable format (remove clique_sizes array)
                serializable_metrics = {}
                for tool, tool_metrics in metrics.items():
                    serializable_metrics[tool] = {
                        k: int(v) if isinstance(v, (np.integer, int)) else v
                        for k, v in tool_metrics.items()
                        if k != "clique_sizes"  # Skip numpy array
                    }
                all_dataset_metrics[dataset_id] = serializable_metrics
                print(f"  Computed: {dataset_id}")
            except Exception as e:
                print(f"  Error processing {dataset_id}: {e}")
                continue

        # Save to cache if path specified
        if cache_path:
            try:
                with open(cache_path, "w") as f:
                    json.dump(all_dataset_metrics, f, indent=2)
                print(f"Saved clique metrics to cache: {cache_path}")
            except Exception as e:
                print(f"Failed to save cache: {e}")

    # Aggregate metrics across all datasets
    combined_metrics = (
        {}
    )  # {tool: {self_match_count: X, group_clique_count: Y, total_clique_count: Z}}

    for dataset_path in input_dataset_directories:
        dataset_id = dataset_path.strip("/").split("/")[-1]

        if dataset_id not in all_dataset_metrics:
            continue

        for tool, tool_metrics in all_dataset_metrics[dataset_id].items():
            if tool not in combined_metrics:
                combined_metrics[tool] = defaultdict(int)

            combined_metrics[tool]["self_match_count"] += tool_metrics.get(
                "self_match_count", 0
            )
            combined_metrics[tool]["group_clique_count"] += tool_metrics.get(
                "group_clique_count", 0
            )
            combined_metrics[tool]["total_clique_count"] += tool_metrics.get(
                "total_clique_count", 0
            )

    if not combined_metrics:
        print("No data to plot")
        return None

    return combined_metrics
