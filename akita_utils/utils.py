import numpy as np

# numeric utilites
def absmaxND(a, axis=None):
    """
    Compute the element-wise maximum absolute value along the specified axis.

    Parameters
    ----------
    a : numpy.ndarray
        Input array.
    axis : int or tuple of ints, optional
        Axis or axes along which to compute the maximum absolute values.
        If None, compute over all elements of the input array (default).

    Returns
    -------
    numpy.ndarray
        Array containing the maximum absolute values along the specified axis.
    """
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)


def smooth(y, box_pts):
    """
    Smooth the input array `y` using a moving average window defined by `box_pts`.

    Parameters
    ----------
    y : numpy.ndarray
        Input array to be smoothed.
    box_pts : int
        Size of the moving average window (number of points).

    Returns
    -------
    numpy.ndarray
        Smoothed array with the same shape as `y`.
    """
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def ut_dense(preds_ut, diagonal_offset=2):
    """Construct symmetric dense prediction matrices from upper triangular vectors.

    Parameters
    -----------
    preds_ut : ( M x O) numpy array
        Upper triangular matrix to convert. M is the number of upper triangular entries,
        and O corresponds to the number of different targets.
    diagonal_offset : int
        Number of diagonals that are added as zeros in the conversion.
        Typically 2 diagonals are ignored in Hi-C data processing.

    Returns
    --------
    preds_dense : (D x D x O) numpy array
        Each output upper-triangular vector is converted to a symmetric D x D matrix.
        Output matrices have zeros at the diagonal for `diagonal_offset` number of diagonals.

    """
    ut_len, num_targets = preds_ut.shape

    # infer original sequence length
    seq_len = int(np.sqrt(2 * ut_len + 0.25) - 0.5)
    seq_len += diagonal_offset

    # get triu indexes
    ut_indexes = np.triu_indices(seq_len, diagonal_offset)
    assert len(ut_indexes[0]) == ut_len

    # assign to dense matrix
    preds_dense = np.zeros(
        shape=(seq_len, seq_len, num_targets), dtype=preds_ut.dtype
    )
    preds_dense[ut_indexes] = preds_ut

    # symmetrize
    preds_dense += np.transpose(preds_dense, axes=[1, 0, 2])

    return preds_dense
