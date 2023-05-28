#!/bin/bash

# Define variables
PARAMS_FILE="/project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json"
JASPAR_FILE="/project/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz"
CHROM_SIZES_FILE="/project/fudenber_735/genomes/mm10/mm10.chrom.sizes.reduced"
BOUNDARY_FILE="/project/fudenber_735/GEO/bonev_2017_GSE96107/distiller-0.3.1_mm10/results/coolers/features/bonev2017.HiC_ES.mm10.mapq_30.1000.window_200000.insulation"
BOUNDARY_STRENGTH_THRESH=0.25
BOUNDARY_INSULATION_THRESH=0.00
BOUNDARY_OUTPUT_TSV="/scratch1/kamulege/boundaries.motifs.ctcf.mm10.tsv"


# Call Python script with arguments
python prepare_boundary_CTCF_tsv.py --params-file "$PARAMS_FILE" --jaspar-file "$JASPAR_FILE" --chrom-sizes-file "$CHROM_SIZES_FILE" --boundary-file "$BOUNDARY_FILE" --boundary-strength-thresh $BOUNDARY_STRENGTH_THRESH --boundary-insulation-thresh $BOUNDARY_INSULATION_THRESH --boundary-output-tsv "$BOUNDARY_OUTPUT_TSV"