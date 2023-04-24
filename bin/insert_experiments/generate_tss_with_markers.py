"""
This script explores experimental data containing information about different markers, such as NIPBL and H3K27 Acetylation. It creates three dataframes: transcription start sites, promoters with corresponding markers, and enhancers with corresponding markers. These dataframes are used to generate different layouts for virtual insertions.

Arguments:

up_stream_bps (int): the number of upstream basepairs that should be considered as promoter sequence.
"""

import pandas as pd
import numpy as np
import bbi
from gtfparse import read_gtf
import argparse
import bioframe
from akita_utils.dna_utils import scan_motif
from akita_utils.seq_gens import generate_spans_start_positions

import json
import pandas as pd

with open("file_paths.json", "r") as f:
    file_paths = json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-up_stream_bps",
        dest="up_stream_bps",
        help="number of basepairs upstream to consider as promoter",
        default=20000,
        type=int,
    )

    args = parser.parse_args()
    
    NBINS = 1

    df = generate_expt_feature_df(file_paths)

    tss_annotations = file_paths["proj_files"]["tss_annotations"]
    tss_df = read_gtf(tss_annotations)
    tss_intervals = get_tss_gene_intervals(tss_df)
    tss_intervals["tss"] = tss_intervals["start"].copy()

    ### create df with intervals & expression data
    tss_df = tss_intervals.merge(
        df.copy(), how="left", left_on="gene_id", right_on="Geneid"
    )
    tss_df = label_DE_status(tss_df)
    tss_df = tss_df.query("avg_counts> 5").copy()
    tss_df.to_csv(f"./data/tss_dataframe.tsv", sep="\t", index=False)

    

    enhancer_path = file_paths["enhancer_files"]["enh_chen_s1"]
    enhancer_df = bioframe.read_table(enhancer_path, schema="bed3", header=1)
    enhancer_df = bioframe_clean_autosomes(enhancer_df)
    enhancer_NIPBL_df = append_score_value_to_df(
        "Nipbl", "NIPBL_score", enhancer_df, nbins=NBINS
    )
    enhancer_H3K27Ac_df = append_score_value_to_df(
        "H3K27Ac", "H3K27Ac_score", enhancer_df, nbins=NBINS
    )
    enhancer_merged_df = pd.concat(
        [
            enhancer_df.reset_index(drop=True),
            enhancer_NIPBL_df.reset_index(drop=True),
            enhancer_H3K27Ac_df.reset_index(drop=True),
        ],
        axis=1,
    )
    enhancer_merged_df.to_csv("./data/enhancer_score_sample.csv")

    promoter_df = generate_promoter_df(tss_df, up_stream_bps=args.up_stream_bps)
    promoter_NIPBL_df = append_score_value_to_df(
        "Nipbl", "NIPBL_score", promoter_df, nbins=NBINS
    )
    promoter_H3K27Ac_df = append_score_value_to_df(
        "H3K27Ac", "H3K27Ac_score", promoter_df, nbins=NBINS
    )
    promoter_merged_df = pd.concat(
        [
            promoter_df.reset_index(drop=True),
            promoter_NIPBL_df.reset_index(drop=True),
            promoter_H3K27Ac_df.reset_index(drop=True),
        ],
        axis=1,
    )
    promoter_merged_df.to_csv(f"./data/promoter_score_sample_upstream_bp_{args.up_stream_bps}.csv")


# -------------------------------------------------------------------------------------------------
# used functions below from https://github.com/Fudenberg-Research-Group/transcription_3Dfolding


def generate_expt_feature_df(file_paths):
    """
    Generates a feature DataFrame for the experiment using the provided file paths.

    Args:
        file_paths (dict): A dictionary containing file paths for the following files:
            - proj_files.day1_sigRes: a CSV file containing day 1 significant results.
            - proj_files.feature_counts: a CSV file containing raw feature counts.
            - proj_files.vst_normalized_counts: a CSV file containing normalized feature counts.

    Returns:
        A pandas DataFrame containing the experiment feature data, with the following columns:
            - Geneid: the gene ID.
            - Chr: the chromosome.
            - Start: the start position of the feature.
            - End: the end position of the feature.
            - Strand: the strand of the feature.
            - Length: the length of the feature.
            - TreatmentMean: the mean value of the treatment.
            - Log2FoldChange: the log2 fold change of the feature.
            - pvalue: the p-value of the feature.
            - avg_counts: the average of raw feature counts across WT samples.
            - avg_vst_counts: the average of normalized feature counts across WT samples.

    Raises:
        FileNotFoundError: If any of the required files are not found.
        ValueError: If any of the required files are empty or contain invalid data.
    """

    WT_samples = ["KHRNA1", "KHRNA7", "KHRNA13", "KHRNA22", "KHRNA23", "KHRNA50"]

    day1_sigRes = file_paths["proj_files"]["day1_sigRes"]
    day1_res_df = pd.read_csv(day1_sigRes)

    # import table of raw feature counts and calculate average
    feature_counts = file_paths["proj_files"]["feature_counts"]
    feat_counts_df = pd.read_csv(feature_counts).rename(
        columns={"Unnamed: 0": "Geneid"}
    )
    feat_counts_df["avg"] = feat_counts_df[WT_samples].mean(axis="columns")

    # import table of normalized feature counts and calculate average
    vst_normalized_counts = file_paths["proj_files"]["vst_normalized_counts"]
    vst_counts_df = pd.read_csv(vst_normalized_counts).rename(
        columns={"Unnamed: 0": "Geneid"}
    )
    vst_counts_df["avg"] = vst_counts_df[WT_samples].mean(axis="columns")

    feat_counts_df = feat_counts_df.merge(
        vst_counts_df, on="Geneid", how="left", suffixes=("_counts", "_vst_counts")
    )
    feat_counts_df["avg_vst_counts"].fillna(feat_counts_df["avg_counts"], inplace=True)

    # add average normalized counts value to results df
    day1_res_df = day1_res_df.merge(
        feat_counts_df[["Geneid", "avg_vst_counts", "avg_counts"]],
        on="Geneid",
        how="outer",
    )
    expt_feature_df = day1_res_df.copy()

    return expt_feature_df


def append_score_value_to_df(score_name, new_column_name, df, nbins=1):
    bw_path = file_paths["bw_files"][score_name]
    score_matrix = generate_signal_matrix(df, bw_path, nbins=nbins)
    score_columns = [f"{new_column_name}_{i}" for i in range(nbins)]
    return pd.DataFrame(score_matrix, columns=score_columns)


def generate_signal_matrix(
    interval_df,
    chip_seq_file,
    columns=["chrom", "start", "end"],
    window_size=1000,
    window_type="extend",
    nbins=40,
):
    """
    Uses pybbi to measure signal over a set of input intervals.
    Returns a matrix [n_intervals x nbins] of the average of ChIP signal over
    each bin in the matrix.
    Parameters:
    -----------
    interval_df: pandas dataframe that has the list of intervals (usually,
                set of genes)
    chip_seq_file: filepath to ChIP-seq file of type bigWig or bigBed
    columns: the columns in interval_df that define the interval
    window_size: the distance of the
    window_type: 'extend': window_size defines padding length added to ends
                 'centered': window_size extends out from each side of middle
    nbins: number of bins the window is divided into.
    Returns:
    --------
    matrix: np.array of shape [n_intervals x nbins]
            rows correspond to intervals in interval_df and ChIP signal measured
            across column bins.
    """

    intervals = bioframe.from_any(interval_df[columns])
    intervals = intervals.rename(
        columns={columns[0]: "chrom", columns[1]: "start", columns[2]: "end"}
    )
    intervals = bioframe.sanitize_bedframe(intervals)

    if window_type == "extend":
        shapes = pd.Series(intervals["start"] - intervals["end"]).nunique()
        msg = """The size of intervals should be equal to perform stackup.
                 Try window_type: 'centered'"""
        assert shapes == 1, msg

        intervals = bioframe.expand(intervals, pad=window_size)

    if window_type == "centered":
        intervals = bioframe.expand(bioframe.expand(expanded, scale=0), pad=1000)

    with bbi.open(chip_seq_file) as f:
        matrix = f.stackup(
            intervals["chrom"], intervals["start"], intervals["end"], bins=nbins
        )

    return matrix


def bioframe_clean_autosomes(frame):
    """
    Takes a dataframe or bioframe and returns
    a sanitized bioframe with only the intervals on autosomes.
    """

    frame = frame[frame.chrom.str.match("^chr\d+$")]
    frame = bioframe.sanitize_bedframe(frame)

    return frame


def get_tss_gene_intervals(
    tss_df,
    return_cols=["gene_id", "chrom", "start", "end", "strand"],
    chrom_keep="autosomes",
):
    """
    In: a .gtf file containing the chr, start, end
    corresponding to the TSS for the transcripts ready from a
    genomic .gtf format annotation file.
    Output: a dataframe in bioframe format with a single TSS
    per gene, with non-autosomes removed.
    """

    # rename column to chrom to match bedframe/bioframe format
    tss_df = tss_df.rename(columns={"seqname": "chrom"})

    # Removing pseudo chromosomes
    if chrom_keep == "autosomes":
        tss_df = bioframe_clean_autosomes(tss_df)
    elif chrom_keep == "chromosomes":
        tss_df = bioframe_clean_chromosomes(tss_df)

    # drop duplicate TSSes
    return tss_df[return_cols].drop_duplicates(["gene_id"])


def label_DE_status(
    df,
    significance_col="padj",
    significance_threshold=0.05,
    fold_change_col="log2FoldChange",
    fold_change_threshold=0,
    DE_status_column="DE_status",
):
    """
    Appends a column with differential expression status to a DataFrame with
    padj and log2FoldChange columns
    Returns
    --------
    df_out : pd.DataFrame
        DataFrame with DE_status_column
    """
    df_out = df.copy()
    df_out[DE_status_column] = "nonsig"
    sig_inds = df[significance_col] < significance_threshold
    down_sig_inds = (df[fold_change_col] < fold_change_threshold) * sig_inds
    up_sig_inds = (df[fold_change_col] > fold_change_threshold) * sig_inds
    df_out.loc[up_sig_inds, DE_status_column] = "up"
    df_out.loc[down_sig_inds, DE_status_column] = "down"

    return df_out


def generate_promoter_df(feature_dataframe, up_stream_bps=10000):
    """
    Given a feature dataframe, generate a promoter dataframe by extending the
    start position upstream by up_stream_bps if the strand is positive, or extending
    the end position downstream by up_stream_bps if the strand is negative.

    Args:
    - feature_dataframe: Pandas dataframe containing the genomic coordinates of
      the features
    - up_stream_bps: Number of base pairs to extend upstream for positive strand
      features or downstream for negative strand features. Default is 10000.

    Returns:
    - A new Pandas dataframe with the updated genomic coordinates.
    """

    for row in feature_dataframe.itertuples():
        if row.strand == "+":
            feature_dataframe["start"].at[row.Index] = row.start - up_stream_bps
        else:
            feature_dataframe["end"].at[row.Index] = row.end + up_stream_bps

    return feature_dataframe


if __name__ == "__main__":
    main()
