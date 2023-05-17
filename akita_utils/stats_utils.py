
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


def scd_target_matrix(prediction_matrix, reference_matrix=None):
    """
    Calculates SCD score for a multiple-target prediction matrix.
    If reference_matrix is not given, it is assumed that an insertion-into-background
    experiment has been performed so reference values are close to 0.
    
    Parameters
    ------------
    prediction_matrix : numpy array
        Array with Akita predictions; size = (130305, num_targets)
    reference_matrix : numpy array
        Array with Akita predictions for a reference sequence; size = (130305, num_targets)
        
    Returns
    ---------
    num_targets-long vector with SCD score calculated for eacg target.
    """
    if reference_matrix == None:
        return np.sqrt((prediction_matrix**2).sum(axis=0))
    else:
        return np.sqrt(((prediction_matrix-reference_matrix)**2).sum(axis=0))


def scd_single_pred(prediction_vector, reference_vector=None):
    """
    Calculates SCD score for a single-target prediction vector.
    If reference_vector is not given, it is assumed that an insertion-into-background
    experiment has been performed so reference values are close to 0.
    
    Parameters
    ------------
    prediction_vector : numpy array
        Vector with Akita predictions; size = (130305,)
    reference_vector : numpy array
        Vector with Akita predictions for a reference sequence; size = (130305,)
        
    Returns
    ---------
    A float-type SCD score.
    """
    if reference_vector == None:
        return np.sqrt(prediction_vector**2)
    else:
        return np.sqrt((prediction_vector-reference_vector)**2)

                   