"""
This scripts takes the seqs tsv with respective parameters to generate specified shuffled seqs fasta in specified output dir
"""

import numpy as np
import pandas as pd
import pysam
import argparse
import akita_utils.dna_utils


import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="genome_fasta", help="fasta file", required=True)
    parser.add_argument(
        "-tsv",
        dest="shffled_seqs_tsv",
        help="tsv with locus specifications",
        required=True,
    )
    parser.add_argument("-o", dest="output_dir", help="where to store the fasta file", required=True)
    parser.add_argument(
        "-sample_size",
        dest="sample_size",
        type=int,
        help="specific number of seqs to sample",
    )
    parser.add_argument(
        "-specific_locus",
        nargs="+",
        type=int,
        dest="specific_locus",
        help="list of specific seqs indicies in dataframe",
    )

    args = parser.parse_args()

    seqs_df = pd.read_csv(args.shffled_seqs_tsv, sep="\t")
    create_shuffled_seqs(
        args.genome_fasta,
        dataframe=seqs_df,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        specific_locus=args.specific_locus,
    )


def create_shuffled_seqs(
    genome_fasta,
    dataframe,
    output_dir,
    sample_size=None,
    specific_locus=None,
):
    """This function creates and saves shuffled sequences fasta files by sampling sequences from dataframe
    Args:
        genome_fasta(str) : path to fasta file
        dataframe : dataframe of sequences with respective parameters
        output_dir : where to store fatsa file
        sample_size : specific number of seqs to sample (dafaults to whole dataframe if not specified)
        specific_locus : only sample a specific seq in dataframe (defaults to all seqs in dataframe)
    Returns:
        shuffled_seqs : fasta file of shuffled sequences
    """
    num_seqs = dataframe.shape[0]
    genome_open = pysam.Fastafile(genome_fasta)

    if sample_size:
        num_seqs = sample_size

    with open(f"{output_dir}", "w") as f:
        if specific_locus:
            for ind in specific_locus:
                (
                    locus_specification,
                    shuffle_k,
                    ctcf_thresh,
                    scores_thresh,
                    scores_pixelwise_thresh,
                    GC_content,
                ) = dataframe.iloc[ind][
                    [
                        "locus_specification",
                        "shuffle_parameter",
                        "ctcf_selection_threshold",
                        "map_score_threshold",
                        "scores_pixelwise_thresh",
                        "GC_content",
                    ]
                ]
                chrom, start, end = locus_specification.split(",")
                seq = genome_open.fetch(chrom, int(start), int(end)).upper()
                seq_1hot = akita_utils.dna_utils.dna_1hot(seq)
                shuffled_seq_1hot = akita_utils.dna_utils.permute_seq_k(seq_1hot, k=shuffle_k)
                f.write(f">shuffled:{chrom},{start},{end}#{GC_content} \n")
                f.write(f"{akita_utils.dna_utils.dna_1hot_to_seq(shuffled_seq_1hot)} \n")
                log.info(f"finished saving seq_{ind}")

        else:
            for ind in range(num_seqs):
                (
                    locus_specification,
                    shuffle_k,
                    ctcf_thresh,
                    scores_thresh,
                    scores_pixelwise_thresh,
                    GC_content,
                ) = dataframe.iloc[ind][
                    [
                        "locus_specification",
                        "shuffle_parameter",
                        "ctcf_selection_threshold",
                        "map_score_threshold",
                        "scores_pixelwise_thresh",
                        "GC_content",
                    ]
                ]
                chrom, start, end = locus_specification.split(",")
                seq = genome_open.fetch(chrom, int(start), int(end)).upper()
                seq_1hot = akita_utils.dna_utils.dna_1hot(seq)
                shuffled_seq_1hot = akita_utils.dna_utils.permute_seq_k(seq_1hot, k=shuffle_k)
                f.write(f">shuffled:{chrom},{start},{end}#{GC_content} \n")
                f.write(f"{akita_utils.dna_utils.dna_1hot_to_seq(shuffled_seq_1hot)} \n")
                log.info(f"finished saving seq_{ind}")


if __name__ == "__main__":
    main()
