import pandas as pd
import numpy as np


import akita_utils


def test_ut_dense():

    ut_vecs = np.vstack(([4, 5, 6], [-2, -3, -4])).T

    # (3 entries) x (2 targets) input, two empty diagonals --> 4x4x2 output
    assert np.shape(akita_utils.ut_dense(ut_vecs, 2)) == (4, 4, 2)

    # (3 entries) x (2 targets) input, one empty diagonals --> 3x3x2 output
    dense_mats = akita_utils.ut_dense(ut_vecs, 1)

    assert np.shape(dense_mats) == (3, 3, 2)

    # outputs are symmetric dense matrices with the 3 original entries
    # and zeros at the diagonal
    target_0 = np.array([[0, 4, 5], [4, 0, 6], [5, 6, 0]])
    target_1 = np.array([[0, -2, -3], [-2, 0, -4], [-3, -4, 0]])

    assert (dense_mats[:, :, 0] == target_0).all()
    assert (dense_mats[:, :, 1] == target_1).all()
