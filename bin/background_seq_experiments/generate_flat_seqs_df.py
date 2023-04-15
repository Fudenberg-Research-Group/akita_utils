"""
This script generates a dataframe for seqs to be used to generate flat seqs with
specified parameters (same as generate_shuffled_seqs_df.py) with two additional parameters.
it does a grid over all given parameters

(1) map_score_threshold
(2) scores_pixelwise_thresh

---------------- typical arguments for choice of bed (mouse or human) ------------------------
these are bed files with intervals the models were trained on.

seq_bed_file = '/project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed' #mouse
genome_fasta = '/project/fudenber_735/genomes/mm10/mm10.fa' #mouse
seq_bed_file = '/project/fudenber_735/tensorflow_models/akita/v2/data/hg38/sequences.bed' #human
genome_fasta = '/project/fudenber_735/genomes/hg38/hg38.fa'#human
-------------------------------------------------------------------------------------------------

To generate multiple maps, provide multiple value of same parameter in CLI, i.e  e.g. --shuffle_parameter 2 4 8
"""

# import general libraries

import os
import itertools
import pandas as pd
import numpy as np
import bioframe
import argparse
import akita_utils.tsv_gen_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        dest="genome_fasta",
        help="fasta file",
        required=True,
    )
    parser.add_argument(
        "-seq_bed_file",
        dest="seq_bed_file",
        help="bed file for the seqs under investigation",
        required=True,
    )
    parser.add_argument(
        "--output_filename",
        dest="output_filename",
        default="data/flat_seqs_test.tsv",
        help="output_filename",
    )
    parser.add_argument(
        "--shuffle_parameter",
        nargs="+",
        default=[8],
        type=int,
        help="integers separated by spaces",
    )
    parser.add_argument(
        "--ctcf_detection_threshold",
        default=[8],
        nargs="+",
        type=int,
        help="threshold of (CTCF PWM) * DNA OHE window value",
    )
    parser.add_argument(
        "--mutation_method",
        nargs="+",
        default=["permute_whole_seq"],
        help="List of strings from ['permute_whole_seq','randomise_whole_seq','randomise_motif','permute_motif','mask_motif']",
    )
    parser.add_argument(
        "--num_backgrounds",
        type=int,
        default=5,
        help="number of loci to select for background creation",
    )
    parser.add_argument(
        "--mode",
        default="uniform",
        help="loci selection criteria",
    )
    parser.add_argument(
        "--map_score_threshold",
        type=int,
        nargs="+",
        default=[60],
        help="maximum allowable map score, SCD",
    )
    parser.add_argument(
        "--scores_pixelwise_thresh",
        type=int,
        nargs="+",
        default=[0.2],
        help="maximum allowable intensity of a single pixel in a map",
    )
    args = parser.parse_args()

    # prepare dataframe with chromosomes and calculate GC content(using bioframe)
    seq_df = pd.read_csv(
        args.seq_bed_file,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "fold"],
    )
    general_seq_gc_df = bioframe.frac_gc(
        seq_df, bioframe.load_fasta(args.genome_fasta), return_input=True
    )

    grid_search_params = {
        "shuffle_parameter": args.shuffle_parameter,
        "ctcf_detection_threshold": args.ctcf_detection_threshold,
        "mutation_method": args.mutation_method,
        "map_score_threshold": args.map_score_threshold,  # this is SCD score
        "scores_pixelwise_thresh": args.scores_pixelwise_thresh,
    }

    # sampling seq_df dataframe respecting GC content
    seq_gc_df = akita_utils.tsv_gen_utils.filter_sites_by_score(
        general_seq_gc_df,
        score_key="GC",
        upper_threshold=99,
        lower_threshold=1,
        mode=args.mode,
        num_sites=args.num_backgrounds,
    )

    # fixing locus specific variables together before grid_search
    seq_gc_df = (
        seq_gc_df["chrom"].map(str)
        + ","
        + seq_gc_df["start"].map(str)
        + ","
        + seq_gc_df["end"].map(str)
        + "-"
        + seq_gc_df["GC"].map(str)
    )
    locus_list = seq_gc_df.values.tolist()

    grid_search_params["locus_specification"] = locus_list

    grid_search_param_set = list(
        itertools.product(*[v for v in grid_search_params.values()])
    )
    parameters_combo_dataframe = pd.DataFrame(
        grid_search_param_set, columns=grid_search_params.keys()
    )
    parameters_combo_dataframe[
        ["locus_specification", "GC_content"]
    ] = parameters_combo_dataframe["locus_specification"].str.split("-", expand=True)
    parameters_combo_dataframe.to_csv(f"{args.output_filename}", sep="\t", index=False)


if __name__ == "__main__":
    main()
