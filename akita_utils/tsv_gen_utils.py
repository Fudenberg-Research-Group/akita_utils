import pandas as pd
import numpy as np
import akita_utils
import glob
import bioframe
import itertools
from io import StringIO
import akita_utils.format_io


def _split_spans(sites, concat=False, span_cols=["start_2", "end_2"]):
    """Helper function to split a span 'start-end' into two integer series, and either
    return as a dataFrame or concatenate to the input dataFrame"""

    sites_spans_split = (
        sites["span"]
        .str.split("-", expand=True)
        .astype(int)
        .rename(columns={0: span_cols[0], 1: span_cols[1]})
        .copy()
    )
    if concat:
        return pd.concat([sites, sites_spans_split], axis=1,)

    else:
        return sites_spans_split


def filter_boundary_ctcfs_from_h5(
    h5_dirs="/project/fudenber_735/tensorflow_models/akita/v2/analysis/permute_boundaries_motifs_ctcf_mm10_model*/scd.h5",
    score_key="SCD",
    threshold_all_ctcf=5,
):
    """Takes a set of boundary mutagenesis dataframes as input, where individual sites are saved in the 'span' column,
    extracts sites greater than a threshold, and filters out sites that overlap with repeatmasker elements.
    """
    ## load scores from boundary mutagenesis, average chosen score across models
    dfs = []
    for h5_file in glob.glob(h5_dirs):
        dfs.append(akita_utils.format_io.h5_to_df(h5_file))
    df = dfs[0].copy()
    df[score_key] = np.mean([df[score_key] for df in dfs], axis=0)

    # append scores for full mut and all ctcf mut to table
    print("annotating each site with boundary-wide scores")
    score_10k = np.zeros((len(df),))
    score_all_ctcf = np.zeros((len(df),))
    for i in np.unique(df["boundary_index"].values):
        inds = df["boundary_index"].values == i
        df_boundary = df.iloc[inds]
        score_10k[inds] = df_boundary.iloc[-1][score_key]
        if len(df_boundary) > 2:
            score_all_ctcf[inds] = df_boundary.iloc[-2][score_key]
    df["score_all_ctcf"] = score_all_ctcf
    df["score_10k"] = score_10k

    # considering only single ctcf mutations
    # require that they fall in an overall boundary that has some saliency
    # TODO: maybe also require that the neighboring bins don't have a more salient boundary?
    # suffix _2 means _motif
    sites = df.iloc[
        (df["strand_2"].values != "nan")
        * (df["score_all_ctcf"].values > threshold_all_ctcf)
    ].copy()

    # extracting start/end of motif from span
    sites = _split_spans(sites, concat=True)
    sites.reset_index(inplace=True, drop=True)
    if sites.duplicated().sum() > 0:
        raise ValueError("no duplicates allowed")
    return sites


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
        chrmsizes = bioframe.read_chromsizes(chrmsizes)
    view_df = bioframe.from_any(chrmsizes)
    chromend_zones = view_df.copy()
    chromend_zones["start"] = chromend_zones["end"] - buffer_bp
    chromstart_zones = view_df.copy()
    chromstart_zones["end"] = chromstart_zones["start"] + buffer_bp
    filter_zones = pd.concat([chromend_zones, chromstart_zones]).reset_index(drop=True)
    df_filtered = bioframe.setdiff(df, filter_zones)
    return df_filtered


def filter_sites_by_score(
    sites,
    score_key="SCD",
    upper_threshold=100,
    lower_threshold=0,
    mode="head",
    num_sites=None,
):
    """
    Given a dataframe of CTCF-binding sites returns a subset of them.

    Parameters
    -----------
    sites : dataframe
        Dataframe of a given CTCF-binding sites.
    score_key : str
        Column that filtering and sorting is done by.
    upper_threshold : float
        Float in a range (0,100); all the sites above this percentile will be removed from further analysis.
    lower_threshold : float
        Float in a range (0,100); all the sites bellow this percentile will be removed from further analysis.
    mode : str
        Specification which part of a filtered dataframe is of a user's interest: if "head" - first rows are returned, if "tail" - last rows are returned, if "random" - a set of random rows is returned.
    num_sites : int
        A requested number of sites. If type of num_sites is None, the function returns all filtered sites.

    Returns
    --------
    filtered_sites : dataframe
        An output dataframe with filtered CTCF-binding sites.

    """

    if mode not in ("head", "tail", "uniform", "random"):
        raise ValueError("a mode has to be one from: head, tail, uniform, random")

    upper_thresh = np.percentile(sites[score_key].values, upper_threshold)
    lower_thresh = np.percentile(sites[score_key].values, lower_threshold)

    filtered_sites = (
        sites[(sites[score_key] >= lower_thresh) & (sites[score_key] <= upper_thresh)]
        .copy()
        .drop_duplicates(subset=[score_key])
        .sort_values(score_key, ascending=False)
    )

    if num_sites != None:
        assert num_sites <= len(
            filtered_sites
        ), "length of dataframe is smaller than requested number of sites, change contraints"

        if mode == "head":
            filtered_sites = filtered_sites[:num_sites]
        elif mode == "tail":
            filtered_sites = filtered_sites[-num_sites:]
        elif mode == "uniform":
            filtered_sites["binned"] = pd.cut(filtered_sites[score_key], bins=num_sites)
            filtered_sites = filtered_sites.groupby("binned").apply(lambda x: x.head(1))
        else:
            filtered_sites = filtered_sites.sample(n=num_sites)

    return filtered_sites


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


def filter_by_rmsk(
    sites,
    rmsk_file="/project/fudenber_735/genomes/mm10/database/rmsk.txt.gz",
    exclude_window=60,
    site_cols=["chrom", "start", "end"],
    verbose=True,
):
    """
    Filter out sites that overlap any entry in rmsk.
    This is important for sineB2 in mice, and perhaps for other repetitive elements as well.

    Parameters
    -----------
    sites : dataFrame
        Set of genomic intervals, currently with columns "chrom","start_2","end_2"
        TODO: update this and allow columns to be passed
    rmsk_file : str
        File in repeatmasker format used for filtering sites.

    Returns
    --------
    sites : dataFrame
        Subset of sites that do not have overlaps with repeats in the rmsk_file.

    """

    if verbose:
        print("filtering sites by overlap with rmsk")

    rmsk_cols = list(
        pd.read_csv(
            StringIO(
                """bin swScore milliDiv milliDel milliIns genoName genoStart genoEnd genoLeft strand repName repClass repFamily repStart repEnd repLeft id"""
            ),
            sep=" ",
        )
    )

    rmsk = pd.read_table(rmsk_file, names=rmsk_cols,)
    rmsk.rename(
        columns={"genoName": "chrom", "genoStart": "start", "genoEnd": "end"},
        inplace=True,
    )

    rmsk = bioframe.expand(rmsk, pad=exclude_window)

    sites = bioframe.count_overlaps(
        sites, rmsk[site_cols], cols1=["chrom", "start_2", "end_2"]
    )

    sites = sites.iloc[sites["count"].values == 0]
    sites.reset_index(inplace=True, drop=True)

    return sites


def filter_by_ctcf(
    sites,
    ctcf_file="/project/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz",
    exclude_window=60,
    cols1=["chrom", "start_2", "end_2"],
    cols2=["chrom", "start", "end"],
    verbose=True,
):
    """
    Filter out sites that overlap any entry in ctcf within a window of 60bp up- and downstream.

    Parameters
    -----------
    sites : dataFrame
        Set of genomic intervals, currently with columns "chrom","start_2","end_2"
    ctcf_file : str
        File in tsv format used for filtering ctcf binding sites.

    Returns
    --------
    sites : dataFrame
        Subset of sites that do not have overlaps with ctcf binding sites in the ctcf_file.
    """

    if verbose:
        print("filtering sites by overlap with ctcfs")

    ctcf_cols = list(
        pd.read_csv(StringIO("""chrom start end name score pval strand"""), sep=" ",)
    )

    ctcf_motifs = pd.read_table(ctcf_file, names=ctcf_cols,)

    ctcf_motifs = bioframe.expand(ctcf_motifs, pad=exclude_window)

    sites = bioframe.count_overlaps(sites, ctcf_motifs, cols1=cols1, cols2=cols2)
    sites = sites.iloc[sites["count"].values == 0]
    sites.reset_index(inplace=True, drop=True)

    return sites


def validate_df_lenght(
    num_strong,
    num_weak,
    num_orientations,
    num_backgrounds,
    number_of_flanks_or_spacers,
    df,
):
    """
    validates if a created dataframe has an expected length (if number of experiments, so number of rows agrees)
    sizes.

    Parameters
    ------------
    num_strong : int
        Number of strong CTCF-binding sites to be tested.
    num_weak : int
        Number of weak CTCF-binding sites to be tested.
    num_backgrounds : int
        Number of different backgrounds to be used.
    flank_range : str
        String in a form: "range_start, range_end".
    df : dataFrame
        Input dataframe

    Returns
    ---------
    (expected_df_len, observed_df_len) : tuple
        Tuple of two integers: expected and observed number of rows.
        There is an assertation error if those values are not the same.
    """

    expected_df_len = (
        (num_strong + num_weak)
        * num_orientations
        * num_backgrounds
        * number_of_flanks_or_spacers
    )

    observed_df_len = len(df)

    assert expected_df_len == observed_df_len

    return (expected_df_len, observed_df_len)


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
                seq_coords_df = pd.concat([seq_coords_df, rep_unit], ignore_index=True)

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
        [rep_unit for i in range(flank_end - flank_start + 1)], ignore_index=True,
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
            seq_coords_df = pd.concat([seq_coords_df, rep_unit], ignore_index=True)

    seq_coords_df["background_index"] = background_ls

    return seq_coords_df


# -------------------------------------------------------------------------------------------------
# functions below under review


def generate_ctcf_positons(
    h5_dirs,
    rmsk_file,
    jaspar_file,
    score_key,
    mode,
    num_sites,
    weak_thresh_pct=1,
    strong_thresh_pct=99,
    unique_identifier="",
):
    """
    This function generates a list of genomic coordinates for potential CTCF binding sites in DNA sequences.

    Arguments:

    h5_dirs: a list of directories containing input .h5 files
    rmsk_file: path to a .bed file containing repeat-masker annotations
    jaspar_file: path to a JASPAR-formatted CTCF motif database
    score_key: key to extract score values of putative binding sites from input .h5 files
    mode: one of "head", "tail", "uniform", "random"; specifies which percentiles of CTCF sites ranked by score should be selected
    num_sites: number of sites to select per mode
    weak_thresh_pct: percentile below which putative binding sites are considered weak (default 1)
    strong_thresh_pct: percentile above which putative binding sites are considered strong (default 99)
    Returns:

    A list of strings representing genomic coordinates of putative CTCF binding sites in format "chrom,start,end,strand#score_key=score_value".
    """
    sites = akita_utils.tsv_gen_utils.filter_boundary_ctcfs_from_h5(
        h5_dirs=h5_dirs, score_key=score_key, threshold_all_ctcf=5,
    )

    sites = akita_utils.tsv_gen_utils.filter_by_rmsk(
        sites, rmsk_file=rmsk_file, verbose=True
    )

    sites = akita_utils.tsv_gen_utils.filter_by_ctcf(
        sites, ctcf_file=jaspar_file, verbose=True
    )

    site_df = akita_utils.tsv_gen_utils.filter_sites_by_score(
        sites,
        score_key=score_key,
        lower_threshold=weak_thresh_pct,
        upper_threshold=strong_thresh_pct,
        mode=mode,
        num_sites=num_sites,
    )

    seq_coords_df = (
        site_df[["chrom", "start_2", "end_2", "strand_2", score_key]]
        .copy()
        .rename(
            columns={
                "start_2": "start",
                "end_2": "end",
                "strand_2": "strand",
                score_key: "genomic_" + score_key,
            }
        )
    )

    seq_coords_df.reset_index(drop=True, inplace=True)
    seq_coords_df["locus_specification"] = (
        seq_coords_df["chrom"].astype(str)
        + ","
        + seq_coords_df["start"].astype(str)
        + ","
        + seq_coords_df["end"].astype(str)
        + ","
        + seq_coords_df["strand"].astype(str)
    )

    extra_cols = set(seq_coords_df.columns) - set(
        ["chrom", "start", "end", "strand", "locus_specification"]
    )
    for col in extra_cols:
        seq_coords_df["locus_specification"] += (
            "#" + f"{unique_identifier}_" + col + "=" + seq_coords_df[col].astype(str)
        )

    return seq_coords_df["locus_specification"].tolist()


# ---------------------------new implementation-------------------


def generate_locus_specification_list(
    dataframe, motif_threshold=1, specification_list=None, unique_identifier="dummy",
):
    """
    Generate a list of locus specifications from a dataframe of genomic features.

    Args:
        dataframe (pandas.DataFrame): A pandas dataframe containing genomic features with columns
            'chrom', 'start', 'end', 'strand' and additional columns to be included in the output.
        motif_threshold (int, optional): The maximum number of motifs allowed in a feature. Defaults to 1.
        specification_list (list, optional): A list of indices to include in the output. Defaults to None.
        unique_identifier (str, optional): A string to identify the unique identifier of additional columns. Defaults to "dummy".

    Returns:
        list: A list of locus specifications generated from the dataframe, where each specification is
            of the format "chrom,start,end,strand#unique_identifier_col_name=value#unique_identifier_col_name=value..."

    """

    # -----------different method of scanning for motifs-------------------
    # feature_dataframe["enhancer_num_of_motifs"] = None # initialisation
    # for row in feature_dataframe.itertuples():
    #     seq_1hot = akita_utils.dna_utils.dna_1hot(genome_open.fetch(row.chrom,row.start,row.end))
    #     motif = akita_utils.format_io.read_jaspar_to_numpy()
    #     num_of_motifs = len(akita_utils.seq_gens.generate_spans_start_positions(seq_1hot, motif, 8))
    #     feature_dataframe["enhancer_num_of_motifs"].at[row.Index] = num_of_motifs

    # -----------different method of scanning for motifs-------------------
    dataframe = filter_by_ctcf(dataframe, cols1=None)
    dataframe = dataframe.rename(columns={"count": "num_of_motifs"})
    # -----------------end-------------------

    dataframe = dataframe[dataframe["num_of_motifs"] <= motif_threshold]
    if "strand" not in dataframe.columns:  # some inserts dont have this column
        dataframe["strand"] = "+"

    # Generate locus specification column
    dataframe["locus_specification"] = (
        dataframe["chrom"].astype(str)
        + ","
        + dataframe["start"].astype(str)
        + ","
        + dataframe["end"].astype(str)
        + ","
        + dataframe["strand"].astype(str)
    )

    # Add any other arbitrary columns from insert dataframe
    extra_cols = set(dataframe.columns) - set(
        ["chrom", "start", "end", "strand", "locus_specification"]
    )
    for col in extra_cols:
        dataframe["locus_specification"] += (
            "#" + f"{unique_identifier}_" + col + "=" + dataframe[col].astype(str)
        )

    # Generate list of locus specifications
    if specification_list is not None:
        locus_specifications = dataframe.loc[
            specification_list, "locus_specification"
        ].tolist()
    else:
        locus_specifications = dataframe["locus_specification"].tolist()

    return locus_specifications


def parameter_dataframe_reorganisation(parameters_combo_dataframe, insert_names_list):
    """
    Reorganizes a parameter combination dataframe to have separate columns for each insert and its
    associated parameters.

    Args:
        parameters_combo_dataframe (pandas.DataFrame): A dataframe with the parameter combinations
            to test, where each row represents a unique combination of parameters and each column
            represents a different parameter. The dataframe must have columns with the locus
            specification for each insert, as well as columns with the flank size, offset, and
            orientation for each insert.
        insert_names_list (list): A list of the names of the inserts to be included in the final
            output dataframe.

    Returns:
        pandas.DataFrame: A reorganized version of the input dataframe, with separate columns for
        each insert and its associated parameters. Each row corresponds to a unique combination of
        parameters, and each column contains the locus specification, flank size, offset, and
        orientation for one insert.

    Raises:
        AssertionError: If any of the insert-specific column names are not found in the input
            dataframe columns.

    """
    for col_name in parameters_combo_dataframe.columns:
        if "locus_specification" in col_name:
            split_df = parameters_combo_dataframe[col_name].str.split("#", expand=True)
            parameters_combo_dataframe = parameters_combo_dataframe.drop(
                columns=[col_name]
            )
            column_names = [col_name] + [x.split("=")[0] for x in split_df.iloc[0, 1:]]
            split_df.columns = column_names

            # Update the values in each cell of the split dataframe
            for column in column_names[1:]:
                split_df[column] = split_df[column].apply(lambda x: x.split("=")[1])

            new_df = pd.concat([parameters_combo_dataframe, split_df], axis=1)
            parameters_combo_dataframe = new_df

    for insert_name in insert_names_list:
        assert (
            f"{insert_name}_locus_specification" in parameters_combo_dataframe.columns
        ), f"{insert_name}_locus_specification not found in dataframe columns."
        assert (
            f"{insert_name}_flank_bp" in parameters_combo_dataframe.columns
        ), f"{insert_name}_flank_bp not found in dataframe columns."
        assert (
            f"{insert_name}_offset" in parameters_combo_dataframe.columns
        ), f"{insert_name}_offset not found in dataframe columns."
        assert (
            f"{insert_name}_orientation" in parameters_combo_dataframe.columns
        ), f"{insert_name}_orientation not found in dataframe columns."

        # Combine all columns into one column separated by "$"
        insert_cols = [
            f"{insert_name}_locus_specification",
            f"{insert_name}_flank_bp",
            f"{insert_name}_offset",
            f"{insert_name}_orientation",
        ]
        parameters_combo_dataframe[
            f"{insert_name}_insert"
        ] = parameters_combo_dataframe[insert_cols].apply(
            lambda x: "$".join(x.astype(str)), axis=1
        )
        parameters_combo_dataframe = parameters_combo_dataframe.drop(
            columns=insert_cols
        )
    return parameters_combo_dataframe
