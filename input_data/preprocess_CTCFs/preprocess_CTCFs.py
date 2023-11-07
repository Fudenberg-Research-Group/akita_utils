#!/usr/bin/env python

"""
This script generates a tsv with CTCFs overlapping boundaries that can be used as input to experiments 
(to do this, move to the experiments/ directory).

This requires the following inputs:
- CTCF motif positions as a jaspar tsv,
- insulation profile with called boundaries as a tsv, currently at 10kb resolution,
- chromosome lengths as a chrom.sizes file,
- model sequence length as json.

First, boundaries are filtered:
- boundaries on non-autosomal chromosomes are dropped,
- boundaries closer than model seq_length // 2 to the start or end of chromosomes are dropped.

Second, CTCF motifs are intersected with boundaries.

Further filtering of CTCFs:
- based on sites overlapping,
- based on overlapping with the rmsk table.

"""

from optparse import OptionParser
import json
import bioframe as bf
import numpy as np
import pandas as pd
import os
from akita_utils.tsv_gen_utils import (
    filter_by_chrmlen,
    filter_by_overlap_num,
    filter_by_chromID,
)
from akita_utils.format_io import read_jaspar_to_numpy, read_rmsk


def main():
    usage = "usage: %prog [options] <params_file> <vcf_file>"
    parser = OptionParser(usage)

    parser.add_option(
        "--model-params-file",
        dest="model_params_file",
        default="/project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json",
        help="Parameters of model to be used[Default: %default]",
    )
    parser.add_option(
        "--jaspar-file",
        dest="jaspar_file",
        default="/project/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz",
        help="Jaspar file with ctcf sites coordinates [Default: %default]",
    )
    parser.add_option(
        "--ctcf-filter-expand-window",
        dest="ctcf_filter_expand_window",
        default=60,
        type=int,
        help="window size for the ctcf-filtering [Default: %default]",
    )
    parser.add_option(
        "--rmsk-file",
        dest="rmsk_file",
        default="/project/fudenber_735/genomes/mm10/database/rmsk.txt.gz",
        help=" [Default: %default]",
    )
    parser.add_option(
        "--rmsk-filter-expand-window",
        dest="rmsk_filter_expand_window",
        default=20,
        type=int,
        help="window size for the rmsk-filtering [Default: %default]",
    )
    parser.add_option(
        "--chrom-sizes-file",
        dest="chrom_sizes_file",
        default="/project/fudenber_735/genomes/mm10/mm10.chrom.sizes.reduced",
        help=" [Default: %default]",
    )
    parser.add_option(
        "--boundary-file",
        dest="boundary_file",
        default="/project/fudenber_735/GEO/bonev_2017_GSE96107/distiller-0.3.1_mm10/results/coolers/features/bonev2017.HiC_ES.mm10.mapq_30.1000.window_200000.insulation",
        help=" [Default: %default]",
    )
    parser.add_option(
        "--boundary-strength-thresh",
        dest="boundary_strength_thresh",
        default=0.25,
        type=float,
        help="threshold on boundary strengths [Default: %default]",
    )
    parser.add_option(
        "--boundary-insulation-thresh",
        dest="boundary_insulation_thresh",
        default=0.00,
        type=float,
        help="threshold on boundary insulation score [Default: %default]",
    )
    parser.add_option(
        "--output-tsv-path",
        dest="output_tsv_path",
        default="./output/CTCFs_jaspar_filtered_mm10.tsv",
        type="str",
        help="Output path [Default: %default]",
    )
    parser.add_option(
        "--autosomes-only",
        dest="autosomes_only",
        default=True,
        action="store_true",
        help="Drop the sex chromosomes and mitochondrial DNA [Default: %default]",
    )

    (options, args) = parser.parse_args()

    if options.autosomes_only:
        chromID_to_drop = ["chrX", "chrY", "chrM"]

    if os.path.exists(options.output_tsv_path) is True:
        raise ValueError("boundary file already exists!")

    # get model seq_length
    with open(options.model_params_file) as params_open:
        params_model = json.load(params_open)["model"]
        seq_length = params_model["seq_length"]
    if seq_length != 1310720:
        raise Warning("potential incompatibilities with AkitaV2 seq_length")

    # load jaspar CTCF motifs
    jaspar_df = bf.read_table(options.jaspar_file, schema="jaspar", skiprows=1)
    if options.autosomes_only:
        jaspar_df = filter_by_chromID(jaspar_df, chromID_to_drop)
    jaspar_df.reset_index(drop=True, inplace=True)

    # read rmsk file
    rmsk_df = read_rmsk(options.rmsk_file)

    # load boundaries and use standard filters for their strength
    boundaries = pd.read_csv(options.boundary_file, sep="\t")

    window_size = options.boundary_file.split("window_")[1].split(".")[0]
    boundary_key, insulation_key = (
        f"boundary_strength_{window_size}",
        f"log2_insulation_score_{window_size}",
    )

    boundaries = boundaries.iloc[
        (boundaries[boundary_key].values > options.boundary_strength_thresh)
        * (
            boundaries[insulation_key].values
            < options.boundary_insulation_thresh
        )
    ]

    if options.autosomes_only:
        boundaries = filter_by_chromID(boundaries, chromID_to_drop)

    boundaries = filter_by_chrmlen(
        boundaries,
        options.chrom_sizes_file,
        seq_length,
    )

    boundaries.reset_index(drop=True, inplace=True)

    # overlapping CTCF df with boundaries df
    df_overlap = bf.overlap(
        boundaries, jaspar_df, suffixes=("", "_2"), return_index=False
    )

    # removing rows with no start and end info
    df_overlap = df_overlap[pd.notnull(df_overlap["start_2"])]
    df_overlap = df_overlap[pd.notnull(df_overlap["end_2"])]

    df_overlap["span"] = (
        df_overlap["start"].astype(str) + "-" + df_overlap["end"].astype(str)
    )

    df_keys = [
        "chrom",
        "start_2",
        "end_2",
        "span",
        "score_2",
        "strand_2",
        insulation_key,
        boundary_key,
    ]

    df_overlap = df_overlap[df_keys]

    # renaming
    df_overlap = df_overlap.rename(
        columns={
            "span": "boundary_span",
            "score_2": "jaspar_score",
            "start_2": "start",
            "end_2": "end",
            "strand_2": "strand",
        }
    )

    # filtering by CTCF
    filtered_df = filter_by_overlap_num(
        df_overlap,
        filter_df=jaspar_df,
        expand_window=options.ctcf_filter_expand_window,
        max_overlap_num=1,
    )

    # filtering by rmsk
    filtered_df = filter_by_overlap_num(
        filtered_df,
        rmsk_df,
        expand_window=options.rmsk_filter_expand_window,
        working_df_cols=["chrom", "start", "end"],
    )

    # adding seq_id
    filtered_df["seq_id"] = [
        seq_index for seq_index in range(len(filtered_df))
    ]

    # saving
    filtered_df.to_csv(options.output_tsv_path, sep="\t", index=False)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
