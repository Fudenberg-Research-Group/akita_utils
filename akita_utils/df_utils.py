import numpy as np
import pandas as pd
import bioframe as bf
import itertools


def average_stat_over_targets(df, model_index=0, head_index=1, stat="SCD"):
    """
    Calculate the average of a specified statistical metric (stat) over multiple targets for a given model and head.

    Parameters:
    df (DataFrame): The input DataFrame containing the data.
    model_index (int): The index of the model for which the metric is calculated.
    head_index (int): The index of the head for which the metric is calculated.
    stat (str, optional): The statistical metric to calculate the average for (default is "SCD").

    Returns:
    DataFrame: A DataFrame with a new column containing the average of the specified metric for the specified model and head.
    """
    if head_index == 1:
        target_indices = 6
    else:
        target_indices = 5

    df[f"{stat}_m{model_index}"] = df[
        [
            f"{stat}_h{head_index}_m{model_index}_t{target_index}"
            for target_index in range(target_indices)
        ]
    ].mean(axis=1)
    return df


def average_stat_over_backgrounds(
    df,
    model_index=0,
    head_index=1,
    num_backgrounds=10,
    stat="SCD",
    columns_to_keep=["chrom", "end", "start", "strand", "seq_id"],
    keep_background_columns=True,
):
    """
    Calculate the average of a specified statistical metric (stat) over multiple background samples for a given model and head.

    Parameters:
    df (DataFrame): The input DataFrame containing the data, including background information.
    model_index (int, optional): The index of the model for which the metric is calculated (default is 0).
    head_index (int, optional): The index of the head for which the metric is calculated (default is 1).
    num_backgrounds (int, optional): The number of background samples to consider (default is 10).
    stat (str, optional): The statistical metric to calculate the average for (default is "SCD").
    columns_to_keep (list, optional): A list of columns to keep in the output DataFrame (default is ["chrom", "end", "start", "strand", "seq_id"]).
    keep_background_columns (bool, optional): Whether to keep individual background columns in the output DataFrame (default is True).

    Returns:
    DataFrame: A DataFrame with the specified statistical metric's average for the specified model and head, along with optional columns.
    """
    num_sites = len(df) // num_backgrounds
    output_df = df[columns_to_keep][:num_sites]

    for bg_index in range(num_backgrounds):
        output_df[f"{stat}_bg{bg_index}"] = df[
            df["background_index"] == bg_index
        ][f"{stat}_m{model_index}"].values

    output_df[f"{stat}_m{model_index}"] = output_df[
        [f"{stat}_bg{bg_index}" for bg_index in range(num_backgrounds)]
    ].mean(axis=1)

    if keep_background_columns == False:
        output_df = output_df.drop(
            columns=[
                f"{stat}_bg{bg_index}" for bg_index in range(num_backgrounds)
            ]
        )

    return output_df


def average_stat_for_shift(df, shift, model_index=0, head_index=1, stat="SCD"):
    """
    Compute the average of specified statistics for a given shift and add it as a new column to the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the statistics.
    shift (str): The shift for which the average statistic is calculated (used as a suffix in the new column name).
    model_index (int): The index of the model from which the statistics are taken.
    head_index (int): The index of the head from which the statistics are taken. Determines the range of target indices.
    stat (str): The type of statistic to average (default is "SCD").

    Returns:
    pd.DataFrame: The DataFrame with a new column containing the average of the specified statistics for the given shift.
    """
    if head_index == 1:
        target_indices = 6
    else:
        target_indices = 5

    df[f"{stat}_{shift}"] = df[
        [
            f"{stat}_h{head_index}_m{model_index}_t{target_index}"
            for target_index in range(target_indices)
        ]
    ].mean(axis=1)
    return df


def split_by_percentile_groups(
    df,
    column_to_split,
    num_classes,
    upper_percentile=100,
    lower_percentile=0,
    category_colname="category",
):
    """
    Splits a dataframe into distinct groups based on the percentile ranges of a specified column.
    Each group represents a percentile range based on the number of classes specified.

    Parameters
    ----------
    df : DataFrame
        The input pandas dataframe.
    column_to_split : str
        The column based on which the dataframe is split into percentile groups.
    num_classes : int
        The number of classes to split the dataframe into.
    upper_percentile : int, default 100
        The upper limit of the percentile range. Typically set to 100.
    lower_percentile : int, default 0
        The lower limit of the percentile range. Typically set to 0.
    category_colname : str, default "category"
        The name of the new column to be added to the dataframe, indicating the category of each row based on percentile range.

    Returns
    -------
    DataFrame
        A new dataframe with an additional column named as specified by 'category_colname'.
        This column contains categorical labels corresponding to the specified percentile ranges.
    """
    bounds = np.linspace(
        lower_percentile,
        (upper_percentile - lower_percentile),
        num_classes + 1,
        dtype="int",
    )
    df_out = pd.DataFrame()

    for i in range(num_classes):
        group_df = filter_dataframe_by_column(
            df,
            column_name=column_to_split,
            upper_threshold=bounds[i + 1],
            lower_threshold=bounds[i],
            drop_duplicates=False,
        )
        group_df[category_colname] = f"Group_{i}"
        df_out = pd.concat([df_out, group_df])

    return df_out


def split_df_equally(df, num_chunks, chunk_idx):
    """
    Split a DataFrame into equal chunks and return the chunk specified by chunk_idx.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be split.
    num_chunks : int
        Total number of chunks to split the DataFrame into.
    chunk_idx : int
        Index of the chunk to retrieve (0-indexed).

    Returns
    -------
    pandas.DataFrame
        The chunk of the DataFrame corresponding to chunk_idx.
    """
    df_len = len(df)
    chunks_bounds = np.linspace(0, df_len, num_chunks + 1, dtype="int")
    df_chunk = df.loc[
        chunks_bounds[chunk_idx] : (chunks_bounds[chunk_idx + 1] - 1), :
    ]
    return df_chunk


def filter_by_chrmlen(df, chrmsizes, buffer_bp=0):
    """
    filter a dataFrame of intervals by a such than none exceed supplied chromosome
    sizes.

    Parameters
    ------------
    df : dataFrame
        Input dataframe
    chrmsizes : chrmsizes file or dictionary that can be converted to a view
        Input chromosome sizes for filtering
    buffer_bp : int
        Size of zone to exclude intervals at chrom starts or ends.

    Returns
    ---------
    df_filtered : dataFrame
        Subset of intervals that do not exceed chromosome size when extended
    """
    assert type(buffer_bp) is int
    if (type(chrmsizes) is not dict) and (
        type(chrmsizes) is not pd.core.frame.DataFrame
    ):
        chrmsizes = bf.read_chromsizes(chrmsizes)
    view_df = bf.from_any(chrmsizes)
    chromend_zones = view_df.copy()
    chromend_zones["start"] = chromend_zones["end"] - buffer_bp
    chromstart_zones = view_df.copy()
    chromstart_zones["end"] = chromstart_zones["start"] + buffer_bp
    filter_zones = pd.concat([chromend_zones, chromstart_zones]).reset_index(
        drop=True
    )
    df_filtered = bf.setdiff(df, filter_zones)
    return df_filtered


def filter_by_chromID(
    df, chrom_column="chrom", chrID_to_drop=["chrX", "chrY", "chrM"]
):
    """
    Filter a DataFrame based on chromosome IDs.

    This function takes a pandas DataFrame and a list of chromosome IDs
    to drop from the DataFrame. It filters out rows where the 'chrom' column
    matches any of the provided chromosome IDs.

    Parameters:
    - df (pandas.DataFrame): Input DataFrame containing a 'chrom' column
                            representing chromosome IDs.
    - chrID_to_drop (list): List of chromosome IDs to be filtered out from the DataFrame.

    Returns:
    pandas.DataFrame: A new DataFrame with rows removed where the 'chrom' column
                      matches any of the chromosome IDs in chrID_to_drop.
    """
    filtered_df = df[~df[chrom_column].isin(chrID_to_drop)]
    return filtered_df


def filter_dataframe_by_column(
    df,
    column_name="SCD",
    upper_threshold=100,
    lower_threshold=0,
    filter_mode="uniform",
    num_rows=None,
    drop_duplicates=True,
):
    """
    Filters a pandas dataframe by a specified column, considering given thresholds, filter mode, and other parameters. Optionally, removes duplicate rows based on the specified column.

    Parameters
    -----------
    df : DataFrame
        An input pandas dataframe with a column specified by column_name.
    column_name : str, default "SCD"
        Column that filtering is based on.
    upper_threshold : float, default 100
        Float in a range (0,100); rows with column_name's values above this percentile are removed.
    lower_threshold : float, default 0
        Float in a range (0,100); rows with column_name's values below this percentile are removed.
    filter_mode : str, default "uniform"
        Specifies the subset of interest in the filtered dataframe.
        "head" returns the first rows, "tail" returns the last rows, "random" returns a random set of rows.
        Otherwise, rows are sampled uniformly with respect to their column_name's values.
    num_rows : int, optional
        The number of rows to return. If None, the function returns the full filtered dataframe.
    drop_duplicates : bool, default True
        If True, duplicate rows based on the column_name are dropped.

    Returns
    --------
    filtered_df : DataFrame
        The resulting dataframe after applying filters and removing duplicates if specified.

    Raises
    ------
    ValueError
        If filter_mode is not one of "head", "tail", "uniform", "random".
    AssertionError
        If num_rows is more than the length of the filtered dataframe.
    """

    if filter_mode not in ("head", "tail", "uniform", "random"):
        raise ValueError(
            "a filter_mode has to be one from: head, tail, uniform, random"
        )

    upper_thresh = np.percentile(df[column_name].values, upper_threshold)
    lower_thresh = np.percentile(df[column_name].values, lower_threshold)

    filtered_df = df[
        (df[column_name] >= lower_thresh) & (df[column_name] <= upper_thresh)
    ].copy()

    if drop_duplicates is True:
        filtered_df = filtered_df.drop_duplicates(subset=[column_name])
    filtered_df = filtered_df.sort_values(column_name, ascending=False)

    if num_rows is not None:
        assert num_rows <= len(
            filtered_df
        ), "length of dataframe is smaller than requested number of sites, change contraints"

        if filter_mode == "head":
            filtered_df = filtered_df[:num_rows]
        elif filter_mode == "tail":
            filtered_df = filtered_df[-num_rows:]
        elif filter_mode == "uniform":
            filtered_df["binned"] = pd.cut(
                filtered_df[column_name], bins=num_rows
            )
            filtered_df = filtered_df.groupby("binned", observed=False).apply(
                lambda x: x.head(1)
            )
        else:
            filtered_df = filtered_df.sample(n=num_rows)

    return filtered_df


def unpack_range(int_range):
    """
    Given start and end of a range of integer numbers as a string converts it to a tuple of integers (int type).

    Parameters
    -----------
    int_range : string
        String in a form: "range_start,range_end"

    Returns
    --------
    (range_start,range_end) : tuple
        A tuple of integer-type numbers.

    """
    range_start, range_end = [int(num) for num in int_range.split(",")]
    return (range_start, range_end)


def filter_by_overlap_num(
    working_df,
    filter_df,
    expand_window=60,
    working_df_cols=["chrom", "start", "end"],
    filter_df_cols=["chrom", "start", "end"],
    max_overlap_num=0,
):
    """
    Filter out rows from working_df that overlap entries in filter_df above given threshold.

    Parameters
    -----------
    working_df : dataFrame
        First set of genomic intervals.
    filter_df : dataFrame
        Second set of genomic intervals.
    expand_window : int
        Indicates how big window around the given genomic intervals should be taken into account.
    working_df_cols : list
        Columns specifying genomic intervals in the working_df.
    filter_df_cols : list
        Columns specifying genomic intervals in the filter_df.
    max_overlap_num : int
        All the rows with number of overlaps above this threshold will be filtered out.

    Returns
    --------

    working_df : dataFrame
        Subset of working_df that do not have overlaps with filter_df above given threshold.
    """

    filter_df = bf.expand(filter_df, pad=expand_window)

    working_df = bf.count_overlaps(
        working_df, filter_df[filter_df_cols], cols1=working_df_cols
    )

    working_df = working_df.iloc[working_df["count"].values <= max_overlap_num]
    working_df.reset_index(inplace=True, drop=True)

    working_df = working_df.drop(columns=["count"])

    return working_df


def generate_all_orientation_strings(N):
    """
    Function generates all possible orientations of N-long string consisting of binary characters (> and <) only.
    Example: for N=2 the result is ['>>', '><', '<>', '<<'].

    Parameters
    -----------
    N : int
        A desired length of each orientation string.

    Returns
    --------
    list
        A list of all possible N-long orientation strings.
    """

    def _binary_to_orientation_string_map(binary_list):
        binary_to_orientation_dict = {0: ">", 1: "<"}
        orientation_list = [
            binary_to_orientation_dict[number] for number in binary_list
        ]

        return "".join(orientation_list)

    binary_list = [list(i) for i in itertools.product([0, 1], repeat=N)]

    return [
        _binary_to_orientation_string_map(binary_element)
        for binary_element in binary_list
    ]


def add_orientation(seq_coords_df, orientation_strings, all_permutations):
    """
    Function adds an additional column named 'orientation', to the given dataframe where each row corresponds to a set of CTCF-binding sites.

    Parameters
    -----------
    seq_coords_df : dataFrame
        Set of experiments where each row specifies a set of CTCF-binding sites.
    orientation_strings : list
        List of orientation strings encoding directionality of CTCF-binding sites.
    all_permutations : boolean
        True if all possible orientation strings of a given length should be generated.

    Returns
    --------
    seq_coords_df : dataFrame
        An input dataframe with the "orientation" column.
    """

    df_len = len(seq_coords_df)

    if len(orientation_strings) > 1:
        orientation_ls = []
        rep_unit = seq_coords_df

        for ind in range(len(orientation_strings)):
            orientation = orientation_strings[ind]
            orientation_ls = orientation_ls + [orientation] * df_len
            if len(seq_coords_df) != len(orientation_ls):
                seq_coords_df = pd.concat(
                    [seq_coords_df, rep_unit], ignore_index=True
                )

        seq_coords_df["orientation"] = orientation_ls

    else:
        if all_permutations:
            N = len(orientation_strings[0])

            orientation_strings = generate_all_orientation_strings(N)

            orientation_ls = []
            rep_unit = seq_coords_df

            for ind in range(len(orientation_strings)):
                orientation = orientation_strings[ind]
                orientation_ls = orientation_ls + [orientation] * df_len
                if len(seq_coords_df) != len(orientation_ls):
                    seq_coords_df = pd.concat(
                        [seq_coords_df, rep_unit], ignore_index=True
                    )

            seq_coords_df["orientation"] = orientation_ls

        else:
            orientation_ls = []
            orientation_ls = orientation_strings * df_len

            seq_coords_df["orientation"] = orientation_ls

    return seq_coords_df


def add_diff_flanks_and_const_spacer(
    seq_coords_df, flank_start, flank_end, flank_spacer_sum
):
    """
    Function adds two additional columns named "flank_bp" and "spacer_bp" to the given dataframe where each row corresponds to a set of CTCF-binding sites. Here, spacing stays constant while flank changes.

    Parameters
    -----------
    seq_coords_df : dataFrame
        Set of experiments where each row specifies a set of CTCF-binding sites.
    flank_start : int
        Integer specifying start of the tested flanks range.
    flank_end : int
        Integer specifying end of the tested flanks range.
    flank_spacer_sum : int
        Sum of flank and spacer lengths.
        In other words, one half of a tail-to-head distance between two CTCFs.

    Returns
    --------
    seq_coords_df : dataFrame
        An input dataframe with two columns added: "flank_bp" and "spacer_bp".
    """

    rep_unit = seq_coords_df
    df_len = len(rep_unit)

    flank_ls = []
    spacer_ls = []

    seq_coords_df = pd.concat(
        [rep_unit for i in range(flank_end - flank_start + 1)],
        ignore_index=True,
    )

    for flank in range(flank_start, flank_end + 1):
        spacer = flank_spacer_sum - flank
        flank_ls = flank_ls + [flank] * df_len
        spacer_ls = spacer_ls + [spacer] * df_len

    seq_coords_df["flank_bp"] = flank_ls
    seq_coords_df["spacer_bp"] = spacer_ls

    return seq_coords_df


def add_const_flank_and_diff_spacer(seq_coords_df, flank, spacing_list):
    """
    Function adds two additional columns named "flank_bp" and "spacer_bp" to the given dataframe where each row corresponds to a set of CTCF-binding sites. Here flank is constant, while spacing is changing.

    Parameters
    -----------
    seq_coords_df : dataFrame
        Set of experiments where each row specifies a set of CTCF-binding sites.
    flank : int
        Flank length that will stay constant in all experiments.
    spacing_list : list
        List of sums of spacer lengths.

    Returns
    --------
    seq_coords_df : dataFrame
        An input dataframe with two columns added: "flank_bp" and "spacer_bp".
    """

    rep_unit = seq_coords_df
    df_len = len(rep_unit)

    flank_ls = []
    spacer_ls = []

    seq_coords_df = pd.concat(
        [rep_unit for i in range(len(spacing_list))], ignore_index=True
    )

    for spacer in spacing_list:
        flank_ls = flank_ls + [flank] * df_len
        spacer_ls = spacer_ls + [spacer] * df_len

    seq_coords_df["flank_bp"] = flank_ls
    seq_coords_df["spacer_bp"] = spacer_ls

    return seq_coords_df


def add_background(seq_coords_df, background_indices_list):
    """
    Function adds an additional column named 'orientation', to the given dataframe where each row corresponds to a set of CTCF-binding sites.

    Parameters
    -----------
    seq_coords_df : dataFrame
        Set of experiments where each row specifies a set of CTCF-binding sites.
    background_indices_list: list
        List of background ids encoded as integers.

    Returns
    --------
    seq_coords_df : dataFrame
        An input dataframe with the "background_index" column.
    """

    rep_unit = seq_coords_df
    df_len = len(rep_unit)

    background_ls = []

    for background_ind in background_indices_list:
        background_ls = background_ls + [background_ind] * df_len

        if len(seq_coords_df) != len(background_ls):
            seq_coords_df = pd.concat(
                [seq_coords_df, rep_unit], ignore_index=True
            )

    seq_coords_df["background_index"] = background_ls

    return seq_coords_df
