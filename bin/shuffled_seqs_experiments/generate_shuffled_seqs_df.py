"""
This script generates and saves a dataframe specifying sequence chrom, start, end, along with how they should be shuffled. The saved dataframe can then be used as (i) input to scripts that save shuffled fasta files, or (ii) as an input to shuffle sequences on the fly to generate respective scores.

The inputs to this script are:
(1) fasta file of appropriate genome.
(2) set of intervals to prepare for shuffling, in the bed format.

Other parameters include:

'shuffle_parameter', which specifies the kmer size to shuffle by
'ctcf_detection_threshold', which specifies the accuracy in idenfying motifs from a seq
'mutation_method', which can be any of ['permute_whole_seq','randomise_whole_seq','mask_motif','permute_motif','randomise_motif']
'output_filename', which is the name of the output tsv file
'num_seqs', which is sample size
'mode', locus GC content selection criteria which maybe 'uniform', 'tail', 'head', 'random'
-------------------------------------------------------------------------------------------------

The output dataframe has all possible combinations of the provided parameters.
To generate multiple maps, provide multiple value of same parameter in CLI, i.e  e.g. --shuffle_parameter 2 4 8

---------------- typical arguments for choice of bed (mouse or human) ------------------------
these are bed files with intervals the models were trained on.

seq_bed_file = '/project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed' #mouse
genome_fasta = '/project/fudenber_735/genomes/mm10/mm10.fa' #mouse
seq_bed_file = '/project/fudenber_735/tensorflow_models/akita/v2/data/hg38/sequences.bed' #human
genome_fasta = '/project/fudenber_735/genomes/hg38/hg38.fa'#human
-------------------------------------------------------------------------------------------------

"""

# import general libraries
import itertools
import pandas as pd
import bioframe
import argparse
import akita_utils.tsv_gen_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="genome_fasta", help="fasta file", required=True)
    parser.add_argument(
        "-seq_bed_file",
        dest="seq_bed_file",
        help="bed file for the seqs under investigation",
        required=True,
    )
    parser.add_argument(
        "--output_filename",
        dest="output_filename",
        default="data/shuffled_seqs.tsv",
        help="output_filename",
    )
    parser.add_argument(
        "--shuffle_parameter",
        nargs="+",
        default=[2, 4, 8],
        type=int,
        help="List of integers sepaerated by spaces eg 2 4",
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
        default=[
            "permute_whole_seq",
            "randomise_whole_seq",
            "randomise_motif",
            "permute_motif",
            "mask_motif",
        ],  # ["permute_whole_seq"],
        help="List of strings from ['permute_whole_seq','randomise_whole_seq','randomise_motif','permute_motif','mask_motif']",
    )
    parser.add_argument(
        "--num_seqs",
        type=int,
        default=20,
        help="number of seqs to select from dataframe",
    )
    parser.add_argument("--mode", default="uniform", help="loci selection criteria")
    args = parser.parse_args()

    # prepare dataframe with chromosomes and calculate GC content(using bioframe)
    seq_df = pd.read_csv(
        args.seq_bed_file,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "fold"],
    )
    general_seq_gc_df = bioframe.frac_gc(seq_df, bioframe.load_fasta(args.genome_fasta), return_input=True)

    grid_search_params = {
        "shuffle_parameter": args.shuffle_parameter,
        "ctcf_detection_threshold": args.ctcf_detection_threshold,
        "mutation_method": args.mutation_method,
    }
    
    # sampling seq_df dataframe respecting GC content
    seq_gc_df = akita_utils.tsv_gen_utils.filter_dataframe_by_column(
        general_seq_gc_df,
        column_name="GC",
        upper_threshold=99,
        lower_threshold=1,
        filter_mode=args.mode,
        num_rows=args.num_seqs,
    )

    # fixing locus specific chacteristics together before grid_search
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

    grid_search_param_set = list(itertools.product(*[v for v in grid_search_params.values()]))
    parameters_combo_dataframe = pd.DataFrame(grid_search_param_set, columns=grid_search_params.keys())
    parameters_combo_dataframe[["locus_specification", "GC_content"]] = parameters_combo_dataframe[
        "locus_specification"
    ].str.split("-", expand=True)

    parameters_combo_dataframe.to_csv(f"{args.output_filename}", sep="\t", index=False)


if __name__ == "__main__":
    main()
