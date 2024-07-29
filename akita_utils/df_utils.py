import numpy as np
import pandas as pd
from .tsv_utils import filter_dataframe_by_column

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
