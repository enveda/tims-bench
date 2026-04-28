# -*- coding: utf-8 -*-

"""Code to perform modified spectral entropy calculations."""

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _entropy_similarity_kernel(
    peaks_a,
    peaks_b,
    ms2_tolerance_in_da: float = 0.05,
    min_matched_peaks: int = 3,
):
    # Calculate the entropy similarity of the two spectra.
    peaks_a = peaks_a.copy()
    peaks_b = peaks_b.copy()
    # First apply intensity weighting

    entropy_a = -np.sum(peaks_a[:, 1] * np.log(peaks_a[:, 1]))
    entropy_b = -np.sum(peaks_b[:, 1] * np.log(peaks_b[:, 1]))

    if entropy_a < 3.0:
        w = 0.25 + 0.25 * entropy_a
        peaks_a[:, 1] = np.power(peaks_a[:, 1], w)
        intensity_sum = np.sum(peaks_a[:, 1])
        if intensity_sum > 0:
            peaks_a[:, 1] /= intensity_sum

    if entropy_b < 3.0:
        w = 0.25 + 0.25 * entropy_b
        peaks_b[:, 1] = np.power(peaks_b[:, 1], w)
        intensity_sum = np.sum(peaks_b[:, 1])
        if intensity_sum > 0:
            peaks_b[:, 1] /= intensity_sum

    a: int = 0
    b: int = 0
    peak_a_intensity: float = 0.0
    peak_b_intensity: float = 0.0
    peak_ab_intensity: float = 0.0
    entropy_similarity: float = 0.0

    max_allowed_mass_difference: float = ms2_tolerance_in_da

    matched_peaks = 0
    while a < peaks_a.shape[0] and b < peaks_b.shape[0]:
        mass_difference: float = peaks_a[a, 0] - peaks_b[b, 0]
        # if ms2_tolerance_in_ppm > 0:
        #     max_allowed_mass_difference = peaks_a[a, 0] * ms2_tolerance_in_ppm * 1e-6
        if mass_difference < -max_allowed_mass_difference:
            # This peak only exists in peaks_a.
            a += 1
        elif mass_difference > max_allowed_mass_difference:
            # This peak only exists in peaks_b.
            b += 1
        else:
            # This peak exists in both peaks_a and peaks_b.
            peak_a_intensity = peaks_a[a, 1]
            peak_b_intensity = peaks_b[b, 1]
            peak_ab_intensity = peak_a_intensity + peak_b_intensity
            entropy_similarity += (
                peak_ab_intensity * np.log2(peak_ab_intensity)
                - peak_a_intensity * np.log2(peak_a_intensity)
                - peak_b_intensity * np.log2(peak_b_intensity)
            )
            a += 1
            b += 1
            matched_peaks += 1

    entropy_similarity /= 2

    if matched_peaks < min_matched_peaks:
        return 0.0

    return entropy_similarity


@njit(cache=True)
def _lower_bound(a, x):
    lo = 0
    hi = a.size
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _upper_bound(a, x):
    lo = 0
    hi = a.size
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _ppm_window_bounds(Mq, B_prec_sorted, precursor_ppm):
    # window: |Mr - Mq| * 1e6 / Mq <= ppm -> Mr in [Mq - tol_abs, Mq + tol_abs]
    tol_abs = Mq * precursor_ppm * 1e-6
    low = Mq - tol_abs
    high = Mq + tol_abs
    left = _lower_bound(B_prec_sorted, low)
    right = _upper_bound(B_prec_sorted, high)
    return left, right


@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _modified_flashier_entropy_search(
    A_list,  # queries (typed List), original order
    B_list_sorted,  # refs sorted by precursor
    A_prec,  # (M,) float32, queries' precursors (original order)
    B_prec_sorted,  # (N,) float32, refs' precursors (sorted)
    ms1_ppm_tol: float,
    ms2_tol_da: float,
    min_matched_peaks: int = 3,
):
    m = len(A_list)
    window_sizes = np.zeros(m, dtype=np.int64)  # default 0.0 if gate empty

    for i in prange(m):  # parallel over queries
        ai = A_list[i]
        if ai.shape[0] == 0:
            continue

        Mq = A_prec[i]
        l, r = _ppm_window_bounds(Mq, B_prec_sorted, ms1_ppm_tol)
        if l >= r:
            continue  # no refs pass gate
        window_sizes[i] = r - l

    window_prefix_sums = np.zeros(m, dtype=np.int64)

    window_prefix_sums[1:] = window_sizes.cumsum()[:-1]
    window_prefix_sums[0] = 0

    num_nonzero_entries = window_sizes.sum()

    query_indices = np.zeros(num_nonzero_entries, dtype=np.int64)
    reference_indices = np.zeros(num_nonzero_entries, dtype=np.int64)
    scores = np.zeros(num_nonzero_entries, dtype=np.float32)

    for i in prange(m):  # parallel over queries
        ai = A_list[i]
        if ai.shape[0] == 0:
            continue

        Mq = A_prec[i]
        l, r = _ppm_window_bounds(Mq, B_prec_sorted, ms1_ppm_tol)
        if l >= r:
            continue  # no refs pass gate

        for j in range(l, r):
            bj = B_list_sorted[j]
            if bj.shape[0] == 0:
                continue
            entropy_score = _entropy_similarity_kernel(
                ai, bj, ms2_tol_da, min_matched_peaks
            )
            idx = window_prefix_sums[i] + (j - l)
            query_indices[idx] = i
            reference_indices[idx] = j
            scores[idx] = entropy_score

    return query_indices, reference_indices, scores


def modified_flashier_entropy_search(
    weighted_query_spectra,
    query_precursors,
    weighted_reference_spectra,
    reference_precursors,
    ms1_ppm_tolerance: float = 20,
    ms2_da_tolerance: float = 0.05,
    min_matched_peaks: int = 3,
):
    """
    Returns:
      best_per_query: (M,) float32 with max over references for each query
      (optional) argmax_ref_idx: (M,) int32 with index into the ORIGINAL reference order (or -1)
    """
    # Precursors as contiguous float32
    A_prec = query_precursors
    B_prec = reference_precursors

    # Build typed lists and apply intensity weighting in-place
    A_list = weighted_query_spectra
    B_list = weighted_reference_spectra

    # Warmup (JIT compile)
    _ = _modified_flashier_entropy_search(
        A_list[:3],
        B_list[:3],
        A_prec[:3],
        B_prec[:3],
        ms1_ppm_tolerance,
        ms2_da_tolerance,
    )

    # Run
    query_indices, reference_indices, scores = _modified_flashier_entropy_search(
        A_list,
        B_list,
        A_prec,
        B_prec,
        ms1_ppm_tolerance,
        ms2_da_tolerance,
        min_matched_peaks,
    )

    return query_indices, reference_indices, scores
