import pandas as pd
from akita_utils.df_utils import (
    filter_by_chrmlen,
    filter_dataframe_by_column,
    generate_all_orientation_strings,
    filter_by_overlap_num,
    split_df_equally
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


def test_filter_by_overlap_num():

    # sub-test 1 : testing if counting works

    df1 = pd.DataFrame(
        [
            ["chr1", 50, 60],
            ["chrX", 400, 410],
            ["chr1", 500, 510],
            ["chr1", 520, 530],
            ["chr1", 560, 570],
        ],
        columns=["chrom", "start", "end"],
    )
    df2 = pd.DataFrame(
        [
            ["chrX", 400, 410],
            ["chr1", 500, 510],
            ["chr1", 500, 510],
            ["chr1", 520, 530],
            ["chr1", 560, 570],
            ["chr1", 1500, 1510],
        ],
        columns=["chrom", "start", "end"],
    )

    test1_out = filter_by_overlap_num(
        df1, df2, expand_window=0, max_overlap_num=1
    )

    assert len(test1_out) == 4, "sub-test 1 failed"
    # the only row that is filtered out is: ["chr1", 500, 510]
    # since it repeats twice in the df2

    # sub-test 2 : checking if expanding the window by 30 bp works

    df1 = pd.DataFrame(
        [
            ["chr1", 50, 60],
            ["chr1", 500, 510],
            ["chr1", 560, 570],
        ],
        columns=["chrom", "start", "end"],
    )
    df2 = pd.DataFrame(
        [
            ["chr1", 520, 530],
        ],
        columns=["chrom", "start", "end"],
    )

    test2_out = filter_by_overlap_num(df1, df2, expand_window=30)

    assert len(test2_out) == 2, "sub-test 2 failed"
    # only the middle, ["chr1", 500, 510], row should be filtered out

    # sub-test 3 : checking if expanding the window by 40 bp works

    df1 = pd.DataFrame(
        [
            ["chrX", 400, 410],
            ["chr1", 459, 469],
            ["chr1", 520, 530],
            ["chr1", 560, 570],
            ["chr1", 1500, 1510],
        ],
        columns=["chrom", "start", "end"],
    )
    df2 = pd.DataFrame(
        [
            ["chr1", 500, 510],
        ],
        columns=["chrom", "start", "end"],
    )

    test3_out = filter_by_overlap_num(df1, df2, expand_window=60)

    assert len(test3_out) == 2, "sub-test 3 failed"
    # the only left df1 rows are the first and the last
    # since they do not overlap expanded window od df2

    # sub-test 4 : as sub-test 3, but we're checking the col names

    df1 = pd.DataFrame(
        [
            ["chrX", 400, 410],
            ["chr1", 459, 469],
            ["chr1", 520, 530],
            ["chr1", 560, 570],
            ["chr1", 1500, 1510],
        ],
        columns=["alt_chrom", "alt_start", "alt_end"],
    )
    df2 = pd.DataFrame(
        [
            ["chr1", 500, 510],
        ],
        columns=["chrom", "start", "end"],
    )

    test4_out = filter_by_overlap_num(
        df1, df2, working_df_cols=["alt_chrom", "alt_start", "alt_end"]
    )

    assert len(test4_out) == 2, "sub-test 4 failed"
    # same as in sub-test 3;
    # checking if column names' renaming works as expected

def test_split_df_equally():

    df = pd.DataFrame(np.linspace(0, 99, 100), columns=["col1"])
    fifth_chunk = split_df_equally(df, 20, 5)
    assert (
        fifth_chunk["col1"].to_numpy() == np.linspace(25, 29, 5)
    ).all() == True
