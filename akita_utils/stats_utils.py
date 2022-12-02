
import numpy as np

def _insul_diamond_central(mat, window=10):
    """calculate insulation in a diamond around the central pixel"""
    N = mat.shape[0]
    if window > N // 2:
        raise ValueError("window cannot be larger than matrix")
    mid = N // 2
    lo = max(0, mid + 1 - window)
    hi = min(mid + window, N)
    score = np.nanmean(mat[lo : (mid + 1), mid:hi])
    return score


def insul_diamonds_scores(mats, window=10):
    num_targets = mats.shape[-1]
    scores = np.zeros((num_targets,))
    for ti in range(num_targets):
        scores[ti] = _insul_diamond_central(mats[:, :, ti], window=window)
    return scores