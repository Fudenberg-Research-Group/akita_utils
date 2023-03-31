# import general libraries
"""
This script takes numerous parameters i.e

(1) 'ctcf_locus_specification_1' 
(2) 'ctcf_locus_specification_2',
(3) 'gene_locus_specification',
(4) 'enhancer_locus_specification',
(5) 'ctcf_flank_bp_1',
(6) 'ctcf_flank_bp_2',
(7) 'gene_flank_bp',
(8) 'enhancer_flank_bp',
(9) 'background_seqs',
(10) 'ctcf_offset_1',
(11) 'ctcf_offset_2',
(12) 'gene_offset',
(13) 'enhancer_offset',
(14) 'locus_orientation',


and creats a dataframe with different permutation of these parameters which can be used to generate scores for different scenarios in experimental setting i.e

Background with; 

    CTCFs alone
    Enhancers alone
    Promoters alone
    Ehancer-Promoter
    CTCF-Ehancer-Promoter
    
    
By changing offsets of CTCFs, on can experiment with;

    Boundary(close CTCFs): ['ctcf_offset_1' = 0,'ctcf_offset_2' = 120]
    TADs:  ['ctcf_offset_1' = -490000,'ctcf_offset_2' = 490000]

"""
import itertools
import os
import pysam
import bioframe
import gtfparse
import numpy as np
import pandas as pd
import akita_utils
import akita_utils.tsv_gen_utils
import gtfparse
import argparse
from pathlib import Path
from io import StringIO

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


######################################################################
# __main__
######################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rmsk_file",
        dest="rmsk_file",
        help="rmsk_file",
        default="/project/fudenber_735/genomes/mm10/database/rmsk.txt.gz",
    )
    parser.add_argument(
        "-jaspar_file",
        dest="jaspar_file",
        help="jaspar_file",
        default="/project/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz",
    )
    parser.add_argument(
        "-f",
        dest="genome_fasta",
        help="fasta file",
        default="/project/fudenber_735/genomes/mm10/mm10.fa",
    )
    parser.add_argument("-score_key", dest="score_key", default="SCD")
    parser.add_argument(
        "-h5_dirs",
        dest="h5_dirs",
        help="h5_dirs",
        default="/project/fudenber_735/tensorflow_models/akita/v2/analysis/permute_boundaries_motifs_ctcf_mm10_model*/scd.h5",
    )
    parser.add_argument("-mode", dest="mode", default="flat")
    parser.add_argument("-num_sites", dest="num_sites", default=10, type=int)
    parser.add_argument(
        "-background_seqs", nargs="+", dest="background_seqs", default=[0], type=int
    )
    parser.add_argument(
        "-o", dest="out_dir", default="default_exp_data", help="where to store output"
    )

    args = parser.parse_args()

    genome_open = pysam.Fastafile(args.genome_fasta)
    current_file_path = Path(
        __file__
    )  # needed to locate other parameter files, guess i will transform other files into args instead (TODO)

    # ---------------setting up a grid search over parameters-------------------------------
    # INSTRUCTIONS TO FILL IN PARAMETERS BELOW
    # Provide as many parameters as appropriate.
    # If you comment a parameter, check that you are confertable with the default value!
    # To simulate multiple values of the same parameter, provide in a list format i.e [first value,second value,third value, etc]
    # Remeber locus orientations should match inserts

    grid_search_params = {
        "background_seqs": args.background_seqs,
        "locus_orientation": [
            ">>>>"
        ],  # ,"<<>>","><>>","<>>>",["<",">"], #[">>","<<","<>","><"],
    }

    # ---------------importing CTCF motifs (Comment this block of the parameters if not inserting CTCFs)--------------------------------
    ctcf_locus_specification_list = generate_ctcf_positons(
        args.h5_dirs,
        args.rmsk_file,
        args.jaspar_file,
        args.score_key,
        args.mode,
        args.num_sites,
    )

    # configure the offsets carefully to create either boundaries or TADs
    grid_search_params["ctcf_locus_specification_1"] = [
        ctcf_locus_specification_list[0]
    ]
    grid_search_params["ctcf_flank_bp_1"] = [20]
    grid_search_params["ctcf_offset_1"] = [0]

    grid_search_params["ctcf_locus_specification_2"] = [
        ctcf_locus_specification_list[0]
    ]
    grid_search_params["ctcf_flank_bp_2"] = [20]
    grid_search_params["ctcf_offset_2"] = [120]

    # ---------------importing promoters (Comment this block of the parameters if not inserting promoters)--------------------------------

    # ***********************(to be depreciated)*********************************
    # TSS_data_tsv = current_file_path.parent / "data/tss_dataframe.tsv"
    # TSS_dataframe = pd.read_csv(TSS_data_tsv, sep="\t")
    # gene_locus_specification_list = generate_promoter_list(TSS_dataframe,genome_open, motif_threshold=0,up_stream_bps = 20000)
    # ***********************(to be depreciated)*********************************

    promoter_data_csv = current_file_path.parent / "data/promoter_score_sample.csv"
    promoter_dataframe = pd.read_csv(promoter_data_csv, sep=",")
    gene_locus_specification_list = generate_promoter_list_v2(
        promoter_dataframe, genome_open, motif_threshold=0
    )
    grid_search_params["gene_locus_specification"] = gene_locus_specification_list
    grid_search_params["gene_flank_bp"] = [0]
    grid_search_params["gene_offset"] = [
        100000
    ]  # np.logspace(5, 5.0, num=1, dtype = int), ,,-200000,-300000

    # ---------------importing enhancers (Comment this block of the parameters if not inserting enhancers)-------------------------------
    enhancer_data_csv = current_file_path.parent / "data/enhancer_score_sample.csv"
    enhancer_dataframe = pd.read_csv(enhancer_data_csv, sep=",")
    enhancer_locus_specification_list = generate_enhancer_list(
        enhancer_dataframe, genome_open, motif_threshold=0
    )  #   specification_list=[0],
    grid_search_params[
        "enhancer_locus_specification"
    ] = enhancer_locus_specification_list
    grid_search_params["enhancer_flank_bp"] = [0]
    grid_search_params["enhancer_offset"] = [
        50000
    ]  # ,450000,,-450000,400000,-50000,-400000

    # --------------- grid search of provided parameters -------------------------------
    grid_search_params_set = list(
        itertools.product(*[v for v in grid_search_params.values()])
    )
    parameters_combo_dataframe = pd.DataFrame(
        grid_search_params_set, columns=grid_search_params.keys()
    )
    fill_in_default_values(parameters_combo_dataframe)
    parameters_combo_dataframe = parameter_dataframe_reorganisation(
        parameters_combo_dataframe
    )
    parameters_combo_dataframe.to_csv(f"{args.out_dir}.tsv", sep="\t", index=False)


# -------------------------------------------------------------------------------------------------
# used functions below


def fill_in_default_values(dataframe):
    "filling default values in ungiven or commented parameters"
    parameter_space = [
        ("ctcf_locus_specification_1", None),
        ("ctcf_locus_specification_2", None),
        ("gene_locus_specification", None),
        ("enhancer_locus_specification", None),
        ("ctcf_flank_bp_1", 0),
        ("ctcf_flank_bp_2", 0),
        ("gene_flank_bp", 0),
        ("enhancer_flank_bp", 0),
        ("background_seqs", 0),
        ("ctcf_offset_1", None),
        ("ctcf_offset_2", None),
        ("gene_offset", None),
        ("enhancer_offset", None),
        ("locus_orientation", None),
    ]

    for parameter, default_value in parameter_space:
        if parameter not in dataframe.columns:
            dataframe.insert(1, parameter, default_value, allow_duplicates=False)


def generate_ctcf_positons(
    h5_dirs,
    rmsk_file,
    jaspar_file,
    score_key,
    mode,
    num_sites,
    weak_thresh_pct=1,
    strong_thresh_pct=99,
):
    sites = akita_utils.tsv_gen_utils.filter_boundary_ctcfs_from_h5(
        h5_dirs=h5_dirs,
        score_key=score_key,
        threshold_all_ctcf=5,
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
    seq_coords_df.reset_index(inplace=True)

    seq_coords_df = (
        seq_coords_df["chrom"].map(str)
        + ","
        + seq_coords_df["start"].map(str)
        + ","
        + seq_coords_df["end"].map(str)
        + ","
        + seq_coords_df["strand"].map(str)
        + "#"
        + seq_coords_df["genomic_" + score_key].map(str)
    )

    return seq_coords_df.values.tolist()


def generate_promoter_list(
    feature_dataframe, genome_open, up_stream_bps=10000, motif_threshold=0
):
    # ---------------(this method is going to be discontinued)-----------------
    for row in feature_dataframe.itertuples():
        if row.strand == "+":
            feature_dataframe["start"].at[row.Index] = row.start - up_stream_bps
        else:
            feature_dataframe["end"].at[row.Index] = row.end + up_stream_bps
    # -----------different method of scanning for motifs-------------------
    # feature_dataframe["promoter_num_of_motifs"] = None # initialisation
    # for row in feature_dataframe.itertuples():
    #     seq_1hot = akita_utils.dna_utils.dna_1hot(genome_open.fetch(row.chrom,row.start,row.end))
    #     motif = akita_utils.format_io.read_jaspar_to_numpy()
    #     num_of_motifs = len(akita_utils.seq_gens.generate_spans_start_positions(seq_1hot, motif, 8))
    #     feature_dataframe["promoter_num_of_motifs"].at[row.Index] = num_of_motifs

    feature_dataframe = filter_by_ctcf(feature_dataframe)
    feature_dataframe = feature_dataframe.rename(
        columns={"count": "promoter_num_of_motifs"}
    )
    feature_dataframe = feature_dataframe[
        True == (feature_dataframe["promoter_num_of_motifs"] <= motif_threshold)
    ]
    feature_dataframe.reset_index(drop=True, inplace=True)
    feature_dataframe["locus_specification"] = (
        feature_dataframe["chrom"].map(str)
        + ","
        + feature_dataframe["start"].map(str)
        + ","
        + feature_dataframe["end"].map(str)
        + ","
        + feature_dataframe["strand"].map(str)
        + "#"
        + feature_dataframe["Geneid"].map(str)
        + "#"
        + feature_dataframe["promoter_num_of_motifs"].map(str)
        + "#"
        + feature_dataframe["promoter_NIPBL_score_0"].map(str)
        + "#"
        + feature_dataframe["promoter_H3K27Ac_score_0"].map(str)
    )
    return feature_dataframe["locus_specification"].values.tolist()


def generate_enhancer_list(
    feature_dataframe, genome_open, motif_threshold=0, specification_list=None
):
    feature_dataframe["strand"] = "+"
    feature_dataframe = feature_dataframe.assign(
        enhancer_symbol=[
            f"enh_{i}" for i, row in enumerate(feature_dataframe.itertuples())
        ]
    )
    # -----------different method of scanning for motifs-------------------
    # feature_dataframe["enhancer_num_of_motifs"] = None # initialisation
    # for row in feature_dataframe.itertuples():
    #     seq_1hot = akita_utils.dna_utils.dna_1hot(genome_open.fetch(row.chrom,row.start,row.end))
    #     motif = akita_utils.format_io.read_jaspar_to_numpy()
    #     num_of_motifs = len(akita_utils.seq_gens.generate_spans_start_positions(seq_1hot, motif, 8))
    #     feature_dataframe["enhancer_num_of_motifs"].at[row.Index] = num_of_motifs

    # -----------different method of scanning for motifs-------------------
    feature_dataframe = filter_by_ctcf(feature_dataframe)
    feature_dataframe = feature_dataframe.rename(
        columns={"count": "enhancer_num_of_motifs"}
    )

    # -----------same futher analysis (irrespective of method used)-------------------
    feature_dataframe = feature_dataframe[
        True == (feature_dataframe["enhancer_num_of_motifs"] <= motif_threshold)
    ]
    feature_dataframe.reset_index(drop=True, inplace=True)
    feature_dataframe["locus_specification"] = (
        feature_dataframe["chrom"].map(str)
        + ","
        + feature_dataframe["start"].map(str)
        + ","
        + feature_dataframe["end"].map(str)
        + ","
        + feature_dataframe["strand"].map(str)
        + "#"
        + feature_dataframe["enhancer_symbol"].map(str)
        + "#"
        + feature_dataframe["enhancer_num_of_motifs"].map(str)
        + "#"
        + feature_dataframe["enhancer_NIPBL_score_0"].map(str)
        + "#"
        + feature_dataframe["enhancer_H3K27Ac_score_0"].map(str)
    )

    if specification_list:
        enhancer_locus_specification_list = []
        for ind in specification_list:
            enhancer_locus_specification_list += [
                feature_dataframe["locus_specification"].values.tolist()[ind]
            ]
    else:
        enhancer_locus_specification_list = feature_dataframe[
            "locus_specification"
        ].values.tolist()

    return enhancer_locus_specification_list


def generate_promoter_list_v2(
    feature_dataframe, genome_open, motif_threshold=1, specification_list=None
):
    # -----------different method of scanning for motifs-------------------
    # feature_dataframe["promoter_num_of_motifs"] = None # initialisation
    # for row in feature_dataframe.itertuples():
    #     seq_1hot = akita_utils.dna_utils.dna_1hot(genome_open.fetch(row.chrom,row.start,row.end))
    #     motif = akita_utils.format_io.read_jaspar_to_numpy()
    #     num_of_motifs = len(akita_utils.seq_gens.generate_spans_start_positions(seq_1hot, motif, 8))
    #     feature_dataframe["promoter_num_of_motifs"].at[row.Index] = num_of_motifs

    # -----------different method of scanning for motifs-------------------
    feature_dataframe = filter_by_ctcf(feature_dataframe)
    feature_dataframe = feature_dataframe.rename(
        columns={"count": "promoter_num_of_motifs"}
    )

    # -----------same futher analysis (irrespective of method used)-------------------
    feature_dataframe = feature_dataframe[
        True == (feature_dataframe["promoter_num_of_motifs"] <= motif_threshold)
    ]
    feature_dataframe.reset_index(drop=True, inplace=True)
    feature_dataframe["locus_specification"] = (
        feature_dataframe["chrom"].map(str)
        + ","
        + feature_dataframe["start"].map(str)
        + ","
        + feature_dataframe["end"].map(str)
        + ","
        + feature_dataframe["strand"].map(str)
        + "#"
        + feature_dataframe["Geneid"].map(str)
        + "#"
        + feature_dataframe["promoter_num_of_motifs"].map(str)
        + "#"
        + feature_dataframe["promoter_NIPBL_score_0"].map(str)
        + "#"
        + feature_dataframe["promoter_H3K27Ac_score_0"].map(str)
    )

    if specification_list:
        enhancer_locus_specification_list = []
        for ind in specification_list:
            enhancer_locus_specification_list += [
                feature_dataframe["locus_specification"].values.tolist()[ind]
            ]
    else:
        enhancer_locus_specification_list = feature_dataframe[
            "locus_specification"
        ].values.tolist()

    return enhancer_locus_specification_list


def parameter_dataframe_reorganisation(parameters_combo_dataframe):
    # log.info(f"columns {parameters_combo_dataframe.columns}  \n ********* {parameters_combo_dataframe['ctcf_locus_specification_1']} \n ******** {parameters_combo_dataframe['ctcf_locus_specification_1'].at[0]}")

    # adapting dataframe to desired look
    if parameters_combo_dataframe["ctcf_locus_specification_1"].at[0]:
        parameters_combo_dataframe["ctcf_genomic_score_1"] = parameters_combo_dataframe[
            "ctcf_locus_specification_1"
        ].str.split("#", expand=True)[1]
        parameters_combo_dataframe[
            "ctcf_locus_specification_1"
        ] = parameters_combo_dataframe["ctcf_locus_specification_1"].str.split(
            "#", expand=True
        )[
            0
        ]
    if parameters_combo_dataframe["ctcf_locus_specification_2"].at[0]:
        parameters_combo_dataframe["ctcf_genomic_score_2"] = parameters_combo_dataframe[
            "ctcf_locus_specification_2"
        ].str.split("#", expand=True)[1]
        parameters_combo_dataframe[
            "ctcf_locus_specification_2"
        ] = parameters_combo_dataframe["ctcf_locus_specification_2"].str.split(
            "#", expand=True
        )[
            0
        ]
    if parameters_combo_dataframe["gene_locus_specification"].at[0]:
        parameters_combo_dataframe["gene_symbol"] = parameters_combo_dataframe[
            "gene_locus_specification"
        ].str.split("#", expand=True)[1]
        parameters_combo_dataframe[
            "promoter_num_of_motifs"
        ] = parameters_combo_dataframe["gene_locus_specification"].str.split(
            "#", expand=True
        )[
            2
        ]
        parameters_combo_dataframe[
            "promoter_NIPBL_signal"
        ] = parameters_combo_dataframe["gene_locus_specification"].str.split(
            "#", expand=True
        )[
            3
        ]
        parameters_combo_dataframe[
            "promoter_H3K27Ac_signal"
        ] = parameters_combo_dataframe["gene_locus_specification"].str.split(
            "#", expand=True
        )[
            4
        ]
        parameters_combo_dataframe[
            "gene_locus_specification"
        ] = parameters_combo_dataframe["gene_locus_specification"].str.split(
            "#", expand=True
        )[
            0
        ]
    if parameters_combo_dataframe["enhancer_locus_specification"].at[0]:
        parameters_combo_dataframe["enhancer_symbol"] = parameters_combo_dataframe[
            "enhancer_locus_specification"
        ].str.split("#", expand=True)[1]
        parameters_combo_dataframe[
            "enhancer_num_of_motifs"
        ] = parameters_combo_dataframe["enhancer_locus_specification"].str.split(
            "#", expand=True
        )[
            2
        ]
        parameters_combo_dataframe[
            "enhancer_NIPBL_signal"
        ] = parameters_combo_dataframe["enhancer_locus_specification"].str.split(
            "#", expand=True
        )[
            3
        ]
        parameters_combo_dataframe[
            "enhancer_H3K27Ac_signal"
        ] = parameters_combo_dataframe["enhancer_locus_specification"].str.split(
            "#", expand=True
        )[
            4
        ]
        parameters_combo_dataframe[
            "enhancer_locus_specification"
        ] = parameters_combo_dataframe["enhancer_locus_specification"].str.split(
            "#", expand=True
        )[
            0
        ]

    parameters_combo_dataframe["insert_loci"] = (
        parameters_combo_dataframe["ctcf_locus_specification_1"].map(str)
        + "$"
        + parameters_combo_dataframe["ctcf_locus_specification_2"].map(str)
        + "$"
        + parameters_combo_dataframe["gene_locus_specification"].map(str)
        + "$"
        + parameters_combo_dataframe["enhancer_locus_specification"].map(str)
    )

    parameters_combo_dataframe["insert_flank_bp"] = (
        parameters_combo_dataframe["ctcf_flank_bp_1"].map(str)
        + "$"
        + parameters_combo_dataframe["ctcf_flank_bp_2"].map(str)
        + "$"
        + parameters_combo_dataframe["gene_flank_bp"].map(str)
        + "$"
        + parameters_combo_dataframe["enhancer_flank_bp"].map(str)
    )

    parameters_combo_dataframe["insert_offsets"] = (
        parameters_combo_dataframe["ctcf_offset_1"].map(str)
        + "$"
        + parameters_combo_dataframe["ctcf_offset_2"].map(str)
        + "$"
        + parameters_combo_dataframe["gene_offset"].map(str)
        + "$"
        + parameters_combo_dataframe["enhancer_offset"].map(str)
    )

    return parameters_combo_dataframe


def filter_by_ctcf(
    sites,
    ctcf_file="/project/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz",
    exclude_window=60,
    site_cols=["chrom", "start", "end"],
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
        pd.read_csv(
            StringIO("""chrom start end name score pval strand"""),
            sep=" ",
        )
    )

    ctcf_motifs = pd.read_table(
        ctcf_file,
        names=ctcf_cols,
    )

    ctcf_motifs = bioframe.expand(ctcf_motifs, pad=exclude_window)

    sites = bioframe.count_overlaps(
        sites, ctcf_motifs[site_cols]
    )  # , cols1=["chrom", "start_2", "end_2"]
    # sites = sites.iloc[sites["count"].values == 0]
    sites.reset_index(inplace=True, drop=True)

    return sites


######################################################################
# main
######################################################################
if __name__ == "__main__":
    main()
