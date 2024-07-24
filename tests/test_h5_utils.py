import pandas as pd
import h5py
import numpy as np
import os

from akita_utils.h5_utils import initialize_stat_output_h5


def _check_unique_values(df, h5_file):
    """
    Checks if the number of unique values for all columns in the dataframe is the same as the number of unique values
    for each key in the h5 file.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to compare.
    h5_file : h5py.File or str
        The h5 file or the path to the h5 file to compare.

    Returns
    -------
    bool
        True if the number of unique values match for all columns and keys, False otherwise.
    """
    if isinstance(h5_file, str):
        h5_file = h5py.File(h5_file, "r")

    for column in df.columns:
        if column in h5_file:
            df_unique_count = df[column].nunique()
            h5_unique_count = np.unique(h5_file[column][:]).size

            if df_unique_count != h5_unique_count:
                return False
        else:
            # If the column does not exist in the h5 file
            return False

    return True


def test_initialize_stat_output_h5():
    """
    Tests the `initialize_stat_output_h5` function for correct initialization and data integrity.

    This test function performs the following steps:
    1. Specifies an output directory and a list of statistical metrics.
    2. Reads a toy dataset of genomic coordinates.
    3. Specifies a model file path.
    4. Calls the `initialize_stat_output_h5` function to create an h5 file with these inputs.
    5. Asserts that the number of unique values in each column of the input DataFrame matches
       the number of unique values in each corresponding key in the created h5 file.
    6. Closes the h5 file and removes it from the filesystem.

    The test passes if the `initialize_stat_output_h5` function correctly initializes an h5 file
    and the number of unique values in the DataFrame columns matches those in the h5 file.

    Asserts
    -------
    The function asserts that the `_check_unique_values` function returns `True`, indicating
    that the number of unique values in the DataFrame columns and the h5 file keys are equal.

    Side Effects
    ------------
    - Creates a temporary h5 file in the specified output directory.
    - Deletes the temporary h5 file after the test is completed.
    """

    out_dir = "./OUT"
    stats = ["SCD", "INS-16", "INS-64"]
    toy_gen_coordinates_df = pd.read_csv(
        "./test_data/genomic_coordinates_10rows.tsv", sep="\t"
    )
    model_file = "/project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5"
    filename = "./OUT/STATS_OUT.h5"

    tmp_h5 = initialize_stat_output_h5(
        out_dir, model_file, stats, toy_gen_coordinates_df
    )

    assert _check_unique_values(toy_gen_coordinates_df, filename) is True

    tmp_h5.close()
    os.remove(filename)
