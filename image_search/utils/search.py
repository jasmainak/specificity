from sklearn.linear_model import LogisticRegression
import numpy as np
from progressbar import ProgressBar, SimpleProgress


def search_specificity(s, y, z, verbose=None):
    """Find specificity search ranking."""

    logit = LogisticRegression()

    rank_s = np.zeros(len(s), dtype=np.int)
    r_s = np.zeros((len(s), len(s)))
    if verbose is not None:
        pbar = ProgressBar(widgets=['Specificity search: ', SimpleProgress()],
                           maxval=len(s)).start()
    for query_idx in range(len(s)):
        if verbose is not None:
            pbar.update(query_idx + 1)

        for ref_idx, (this_y, this_z) in enumerate(zip(y, z)):
            logit.intercept_ = np.array([this_y])
            logit.coef_ = np.array([[this_z]])
            r_s[query_idx, ref_idx] = logit.predict_proba(s[query_idx,
                                                            ref_idx])[0][1]

        r_s[np.isnan(r_s)] = -np.inf
        idx_s = _matlab_sort(r_s[query_idx, :])

        # make matlab equiv. by adding 1
        rank_s[query_idx] = np.where(idx_s == query_idx)[0][0] + 1

    if verbose is not None:
        pbar.finish()

    return rank_s


def search_baseline(s, verbose=None):
    """Find baseline search ranking."""
    if verbose is not None:
        pbar = ProgressBar(widgets=['Baseline search: ', SimpleProgress()],
                           maxval=len(s)).start()

    rank_b = np.zeros(len(s), dtype=np.int)
    for query_idx in range(len(s)):
        if verbose is not None:
            pbar.update(query_idx + 1)

        idx_b = _matlab_sort(s[query_idx, :])
        rank_b[query_idx] = np.where(idx_b == query_idx)[0][0] + 1

    if verbose is not None:
        pbar.finish()

    return rank_b


def _matlab_sort(arr, method='descend'):
    """Matlab-like sort in python."""

    arr = -arr if method == 'descend' else arr
    sorted_idx = np.argsort(arr)

    # make sorting matlab like -- i.e. smaller idx always comes
    # first if two elements are equal.
    sorted_arr = np.sort(arr)
    for idx in xrange(len(sorted_arr) - 1):
        if sorted_arr[idx] == sorted_arr[idx + 1]:
            first_idx = min(sorted_idx[idx], sorted_idx[idx + 1])
            next_idx = max(sorted_idx[idx], sorted_idx[idx + 1])
            sorted_idx[idx], sorted_idx[idx + 1] = first_idx, next_idx

    return sorted_idx
