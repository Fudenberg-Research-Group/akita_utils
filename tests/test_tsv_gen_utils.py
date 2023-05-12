import pandas as pd
from akita_utils.tsv_gen_utils import (
    filter_by_chrmlen,
    filter_sites_by_score,
    generate_all_orientation_strings,
)


def test_filter_by_chrmlen():

    df1 = pd.DataFrame(
        [["chrX", 3, 8], ["chr1", 4, 5], ["chrX", 1, 5]],
        columns=["chrom", "start", "end"],
    )

    # get the same result with chrmsizes provided as dict or via StringIO

    # one interval is dropped for exceeding chrX len of 7
    assert filter_by_chrmlen(df1, {"chr1": 10, "chrX": 7}, 0).shape == (2, 3)

    # both chrX intervals are dropped if the buffer_bp are increased
    assert filter_by_chrmlen(df1, {"chr1": 10, "chrX": 7}, 3).shape == (1, 3)

    # no intervals remain if all of chr1 is excluded as well
    assert filter_by_chrmlen(df1, {"chr1": 10, "chrX": 7}, 5).shape == (0, 3)


def test_filter_dataframe_by_column():
    simple_df = pd.DataFrame([i for i in range(10)], columns=["stat"])

    two_and_one = filter_dataframe_by_column(
        simple_df,
        column_name="stat",
        upper_threshold=90,
        lower_threshold=10,
        filter_mode="tail",
        num_rows=2,
    )

    assert list(two_and_one.stat) == [2, 1]


def test_generate_all_orientation_strings():
    orientation_list = [">>", "<<", "<>", "><"]

    for orientation in generate_all_orientation_strings(2):
        assert orientation in orientation_list
