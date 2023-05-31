#!/bin/bash

# parse command line arguments
genome_fasta="/project/fudenber_735/genomes/mm10/mm10.fa" 
models_dir="/project/fudenber_735/tensorflow_models/akita/v2/models"
tsv_file="/scratch1/kamulege/boundaries.motifs.ctcf.mm10.tsv"
# tsv_file="/project/fudenber_735/tensorflow_models/akita/v2/analysis/boundaries.motifs.ctcf.mm10.tsv"
out_dir="/scratch1/kamulege/disruption_genomic_scds_v2" 
models="4" 
scd_stats="SCD,SSD,INS-16,INS-64" 
batch_size=8 
head_index=1 
mutation_method="permute" 
max_proc=7 
processes=100
time="02:30:00" 
constraint="[xeon-6130|xeon-2640v4]" 


# Check if genome_fasta file exists
if [ ! -f "$genome_fasta" ]; then
    echo "Genome fasta file does not exist."
    exit 1
fi

# Check if models_dir directory exists
if [ ! -d "$models_dir" ]; then
    echo "Models directory does not exist."
    exit 1
fi

# Check if tsv_file file exists
if [ ! -f "$tsv_file" ]; then
    echo "TSV file does not exist."
    exit 1
fi

if [ ! -d "$out_dir" ]; then
    echo "Output directory does not exist. Creating directory..."
    mkdir -p "$out_dir"
fi


for model in $models

do 
    # run the command  
    python akita_motif_scd_multi.py "${models_dir}" "${tsv_file}" -f "${genome_fasta}" -o "${out_dir}" --batch-size "${batch_size}" --head-index "${head_index}" --model-index "${model}" -p "${processes}" --time "${time}" --mut-method "${mutation_method}" --constraint "${constraint}" --stats "${scd_stats}"
    sleep 15
done

echo All done