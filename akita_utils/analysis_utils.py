import numpy as np
import pandas as pd
import pysam
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import logomaker as lm
from sklearn.preprocessing import normalize
from akita_utils.dna_utils import dna_rc, dna_1hot_index, dna_1hot


# SINGLE-MAP PLOTTING FUNCTION

def plot_map(matrix, vmin=-0.6, vmax=0.6, width=5, height=5, palette="RdBu_r"):
    """
    Plots a 512x512 log(obs/exp) map.

    Parameters
    ------------
    matrix : numpy array
        Predicted log(obs/exp) map.
    vmin : float
    vmax : float
        Minimum and maximum in the colormap scale.
    width : int
    height : int
        Width and height of a plotted map.
    """

    fig = plt.figure(figsize=(width, height))

    sns.heatmap(
        matrix,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        cmap=palette,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )
    plt.show()


# FLANKING SEQUENCES ANALYSIS

def collect_flanked_sequences(
    sites,
    flank_length=30,
    genome_path="/project/fudenber_735/genomes/mm10/mm10.fa",
):
    """
    Extracts and processes DNA sequences from a given genome, flanking specified genomic sites.

    This function reads a genome from a specified path and retrieves sequences that flank genomic sites of interest.
    Each site is extended by a given flank length on both sides. The function also accounts for the strand
    orientation, providing the reverse complement sequence if the strand is negative.

    Parameters:
    - sites (DataFrame): A pandas DataFrame containing the genomic sites of interest. The DataFrame must
      include columns 'chrom', 'start', 'end', and 'strand'.
    - flank_length (int, optional): The number of base pairs to include on each side of the site. Default is 30.
    - genome_path (str, optional): The file path to the genome fasta file. Default is "/project/fudenber_735/genomes/mm10/mm10.fa".

    Returns:
    - numpy.ndarray: An array where each element is a one-hot encoded representation of the flanked sequence at each site.
    """
    genome_open = pysam.Fastafile(genome_path)
    sites_dna_num = []

    for i in range(len(sites)):
        chrm, start, end, strand = sites.iloc[i][
            ["chrom", "start", "end", "strand"]
        ]
        start = start - flank_length
        end = end + flank_length
        seq = genome_open.fetch(chrm, start, end).upper()
        if strand == "-":
            seq = dna_rc(seq)
        sites_dna_num.append(dna_1hot_index(seq))

    genome_open.close()
    sites_dna_num = np.array(sites_dna_num)

    return sites_dna_num


def reorder_by_hamming_dist(dna_matrix, sub_index=(0, -1)):
    """
    Reorders a matrix of DNA sequences based on their pairwise Hamming distances.

    This function calculates the pairwise Hamming distances between rows (DNA sequences) in a given matrix.
    It then reorders the matrix such that similar sequences (based on the Hamming distance) are grouped together.
    The function allows for considering only a subset of each sequence for the distance calculation.

    Parameters:
    - dna_matrix (numpy.ndarray): A 2D numpy array where each row represents a DNA sequence.
    - sub_index (tuple, optional): A tuple (start, end) indicating the subset of each sequence to consider
      for the Hamming distance calculation. Default is (0, -1) which considers the full length of each sequence.

    Returns:
    - numpy.ndarray: The reordered matrix of DNA sequences, where sequences are ordered based on their
      similarity (lower Hamming distance).

    Note: This function assumes that all sequences in dna_matrix are of equal length.
    """
    num_seqs = len(dna_matrix)
    seq_dist = np.zeros((num_seqs, num_seqs))

    for i in range(num_seqs):
        seq_i = dna_matrix[i][sub_index[0] : sub_index[1]]
        for j in range(num_seqs):
            if i < j:
                seq_j = dna_matrix[j][sub_index[0] : sub_index[1]]
                seq_dist[i, j] = 1 - scipy.spatial.distance.hamming(
                    seq_i, seq_j
                )
    seq_dist = seq_dist + seq_dist.T

    reording = scipy.cluster.hierarchy.leaves_list(
        scipy.cluster.hierarchy.linkage(seq_dist)
    )
    return dna_matrix[reording]


def plot_seq_matrix(dna_matrix, cluster_by_hamming=True, sub_index=(0, -1)):
    """
    Plot a matrix representing DNA sequences, optionally clustering by Hamming distance.

    Parameters
    ----------
    dna_matrix : numpy.ndarray
        Matrix of DNA sequences encoded as integers (0 for 'A', 1 for 'C', 2 for 'G', 3 for 'T').
    cluster_by_hamming : bool, optional
        Whether to reorder rows by Hamming distance to subsequence defined by `sub_index` (default is True).
    sub_index : tuple of ints, optional
        Subsequence indices `(start, end)` to define the sequence for clustering by Hamming distance (default is (0, -1),
        representing the entire sequence).

    Returns
    -------
    None

    Notes
    -----
    This function plots a heatmap of the `dna_matrix` using a colormap that matches standard logo colors for DNA bases (A=green, C=blue, G=gold, T=red).
    If `cluster_by_hamming` is True, it reorders the rows of `dna_matrix` based on Hamming distance to the subsequence defined by `sub_index`.
    """
    # colormap matching logo colors
    cmap_acgt = colors.ListedColormap([
        'green', #a green
        'blue', #c blue
        'gold', #g gold
        'red' #t red
    ])

    if cluster_by_hamming:
        dna_matrix = reorder_by_hamming_dist(dna_matrix, sub_index=sub_index)

    plt.figure(figsize=(10,18))
    im = plt.matshow(
        dna_matrix, 
        cmap=cmap_acgt,
        fignum=False) 
    plt.colorbar(im, fraction=0.046, pad=0.04)


def plot_logo_from_counts(nt_count_table, logo_height = 3, logo_width = 0.45):
    """
    Plot a sequence logo from a nucleotide count table.

    Parameters
    ----------
    nt_count_table : pandas.DataFrame
        Table of nucleotide counts with columns ['A', 'C', 'G', 'T'].
    logo_height : float, optional
        Height of the logo plot (default is 3).
    logo_width : float, optional
        Width of each position in the logo plot (default is 0.45).

    Returns
    -------
    None

    Notes
    -----
    This function normalizes the nucleotide counts to probabilities, transforms them into information content,
    and plots a sequence logo using the Logomaker library.
    """
    dna_prob = normalize(nt_count_table, axis=1, norm="l1")
    dna_prob_df = pd.DataFrame(dna_prob, columns=["A", "C", "G", "T"])

    logo_params = {"df": lm.transform_matrix(dna_prob_df, from_type="probability", to_type="information"),
                "figsize": (logo_width * dna_prob_df.shape[0], logo_height),
                "show_spines": False,
                "vpad": 0.02}
    
    logo = lm.Logo(**logo_params)
    logo.ax.set_ylabel("Bits", fontsize=16)
    logo.ax.set_ylim(0, 2)
    logo.ax.set_yticks([0, 0.5, 1, 1.5, 2], minor=False)


def prepare_nt_count_table(
    sites,
    flank_length=30,
    genome_path="/project/fudenber_735/genomes/mm10/mm10.fa",
):
    """
    Prepares a nucleotide count table for a set of genomic sites with specified flanking regions.

    This function processes a list of genomic sites and extracts the corresponding DNA sequences,
    including flanking regions, from a specified genome. It then counts the occurrences of each
    nucleotide at each position across all sequences and compiles these counts into a table.

    Parameters:
    - sites (DataFrame): A pandas DataFrame containing the genomic sites of interest. The DataFrame must
      include columns 'chrom', 'start', 'end', and 'strand'.
    - flank_length (int, optional): The number of base pairs to include on each side of the site. Default is 30.
    - genome_path (str, optional): The file path to the genome fasta file. Default is "/project/fudenber_735/genomes/mm10/mm10.fa".

    Returns:
    - numpy.ndarray: A 2D numpy array with shape (sequence_length, 4), where sequence_length is the length
      of the flanked site. Each column corresponds to one of the four nucleotides (A, C, G, T), and each
      row represents the count of each nucleotide at that position across all sequences.
    """
    seq_length = flank_length * 2 + (
        sites.iloc[0]["end"] - sites.iloc[0]["start"]
    )
    genome_open = pysam.Fastafile(genome_path)

    nt_count = np.zeros(shape=(seq_length, 4))
    for i in range(len(sites)):
        chrm, start, end, strand = sites.iloc[i][
            ["chrom", "start", "end", "strand"]
        ]
        start = start - flank_length
        end = end + flank_length
        seq = genome_open.fetch(chrm, start, end).upper()
        if strand == "-":
            seq = dna_rc(seq)
        nt_count = nt_count + dna_1hot(seq)
    genome_open.close()
    return nt_count


# AVERAGING FUNCTIONS

def average_stat_over_targets(df, model_index, head_index, stat="SCD"):
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

    if head_index == 1:
        target_indices = 6
    else:
        target_indices = 5

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


def average_stat_for_shift(df, shift, model_index, head_index, stat="SCD"):
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


def split_by_percentile_groups(df, column_to_split, num_classes, 
                               upper_percentile=100, lower_percentile=0, 
                               category_colname="category"):
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
    bounds = np.linspace(lower_percentile, (upper_percentile-lower_percentile), num_classes + 1, dtype="int")
    df_out = pd.DataFrame()

    for i in range(num_classes):
            
        group_df = filter_dataframe_by_column(
            df,
            column_name=column_to_split,
            upper_threshold=bounds[i+1],
            lower_threshold=bounds[i],
            drop_duplicates=False
        )
        group_df[category_colname] = f"Group_{i}"
        df_out = pd.concat([df_out, group_df])

    return df_out

