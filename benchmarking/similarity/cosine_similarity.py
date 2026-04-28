# -*- coding: utf-8 -*-

"""Code to perform cosine similarity calculations."""

import numpy as np
from numba import njit, prange
from numba.typed import List


@njit(cache=True, fastmath=True)
def _cosine_similarity_kernel(
    peaks_a,
    peaks_b,
    ms2_tolerance_in_da: float = 0.1,
    min_matched_peaks: int = 3,
):
    # Calculate the cosine similarity of the two spectra.
    max_allowed_mass_difference: float = ms2_tolerance_in_da

    lowest_idx = 0
    idx1 = List()
    idx2 = List()

    for peaka_idx in range(peaks_a.shape[0]):
        mz = peaks_a[peaka_idx, 0]
        low_bound = mz - max_allowed_mass_difference
        high_bound = mz + max_allowed_mass_difference
        for peakb_idx in range(lowest_idx, peaks_b.shape[0]):
            mz2 = peaks_b[peakb_idx, 0]
            if mz2 > high_bound:
                break
            if mz2 < low_bound:
                lowest_idx = peakb_idx + 1
            else:
                idx1.append(peaka_idx)
                idx2.append(peakb_idx)

    n_pairs = len(idx1)
    if n_pairs < min_matched_peaks:
        return 0.0

    # Convert typed lists -> numpy arrays + compute products (avoid peaks_a[idx1] fancy indexing)
    ia = np.empty(n_pairs, dtype=np.int64)
    ib = np.empty(n_pairs, dtype=np.int64)
    prod = np.empty(n_pairs, dtype=peaks_a.dtype)  # float32 if peaks are float32

    for k in range(n_pairs):
        a = idx1[k]
        b = idx2[k]
        ia[k] = a
        ib[k] = b
        prod[k] = peaks_a[a, 1] * peaks_b[b, 1]

    # sort candidate pairs by product (ascending), then traverse from largest -> smallest
    order = np.argsort(prod)

    used_a = np.zeros(peaks_a.shape[0], dtype=np.uint8)
    used_b = np.zeros(peaks_b.shape[0], dtype=np.uint8)

    total_product = 0.0
    true_matched_peaks = 0

    for t in range(n_pairs - 1, -1, -1):
        k = order[t]
        a = ia[k]
        b = ib[k]
        if used_a[a] == 0 and used_b[b] == 0:
            used_a[a] = 1
            used_b[b] = 1
            total_product += peaks_a[a, 1] * peaks_b[b, 1]
            true_matched_peaks += 1

    if true_matched_peaks < min_matched_peaks:
        return 0.0

    # norm computation (avoid np.linalg.norm if you want maximum compatibility)
    sa = 0.0
    for i in range(peaks_a.shape[0]):
        x = peaks_a[i, 1]
        sa += x * x

    sb = 0.0
    for j in range(peaks_b.shape[0]):
        x = peaks_b[j, 1]
        sb += x * x

    norm = np.sqrt(sa) * np.sqrt(sb)
    if norm == 0.0:
        return 0.0

    return total_product / norm


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
def _flash_cosine_search(
    A_list,  # queries (typed List), original order
    B_list_sorted,  # refs sorted by precursor
    A_prec,  # (M,) float32, queries' precursors (original order)
    B_prec_sorted,  # (N,) float32, refs' precursors (sorted)
    ms1_da_tol: float,
    ms1_ppm_tol: float,
    ms2_tol_da: float,
    min_matched_peaks,
):
    m = len(A_list)
    best = np.zeros(m, dtype=np.float32)  # default 0.0 if gate empty
    argj_sorted = np.full(m, -1, dtype=np.int32)  # argmax in *sorted* ref index space

    for i in prange(m):  # parallel over queries
        ai = A_list[i]
        if ai.shape[0] == 0:
            continue

        Mq = A_prec[i]
        if ms1_da_tol is None or ms1_da_tol <= 0:
            l, r = _ppm_window_bounds(Mq, B_prec_sorted, ms1_ppm_tol)
        else:
            low = Mq - ms1_da_tol
            high = Mq + ms1_da_tol
            l = _lower_bound(B_prec_sorted, low)
            r = _upper_bound(B_prec_sorted, high)

        if l >= r:
            continue  # no refs pass gate

        max_entropy_score = 0.0
        argmax_idx = -1
        for j in range(l, r):
            bj = B_list_sorted[j]
            if bj.shape[0] == 0:
                continue
            curr_entropy_score = _cosine_similarity_kernel(
                ai, bj, ms2_tol_da, min_matched_peaks
            )
            if curr_entropy_score > max_entropy_score:
                max_entropy_score = curr_entropy_score
                argmax_idx = j
        best[i] = max_entropy_score
        argj_sorted[i] = argmax_idx

    return best, argj_sorted


def flash_cosine_search(
    query_spectra,
    query_precursors,
    reference_spectra,
    reference_precursors,
    ms1_da_tolerance: float = None,
    ms1_ppm_tolerance: float = 20,
    ms2_da_tolerance: float = 0.1,
    min_matched_peaks: int = 3,
    return_argmax: bool = False,
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
    A_list = query_spectra
    B_list = reference_spectra

    # Warmup (JIT compile)
    _ = _flash_cosine_search(
        A_list[:3],
        B_list[:3],
        A_prec[:3],
        B_prec[:3],
        ms1_da_tolerance,
        ms1_ppm_tolerance,
        ms2_da_tolerance,
        min_matched_peaks,
    )

    # Run
    best, argj_sorted = _flash_cosine_search(
        A_list,
        B_list,
        A_prec,
        B_prec,
        ms1_da_tolerance,
        ms1_ppm_tolerance,
        ms2_da_tolerance,
        min_matched_peaks,
    )

    if return_argmax:
        return best, argj_sorted

    return best
