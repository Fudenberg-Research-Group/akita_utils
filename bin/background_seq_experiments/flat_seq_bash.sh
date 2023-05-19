#!/bin/bash
# A sample Bash script for generating flat seqs for different models

models_dir="/project/fudenber_735/tensorflow_models/akita/v2/models"
tsv_file="data/flat_seqs_mouse.tsv"
genome_fasta="/project/fudenber_735/genomes/mm10/mm10.fa"
output_dir="/scratch1/kamulege/flat_seqs_mouse_SCD_test"
models='3'


# Check if required files exist
if [ ! -d "$models_dir" ]; then
    echo "Error: models directory does not exist."
    exit 1
fi

if [ ! -f "$tsv_file" ]; then
    echo "Error: TSV file does not exist."
    exit 1
fi

if [ ! -f "$genome_fasta" ]; then
    echo "Error: genome FASTA file does not exist."
    exit 1
fi

if [ ! -d "$output_dir" ]; then
    echo "Output directory does not exist. Creating directory..."
    mkdir -p "$output_dir"
fi


for model in $models
do
     python multiGPU_generate_flat_seqs.py $models_dir $tsv_file -o $output_dir -f $genome_fasta --head-index 1 --model-index $model --batch-size 4 --max_iters 40 -p 1 --max_proc 7 --time 04:00:00 

     if [ $? -ne 0 ]; then
            echo "Error: failed to run experiments for model $model."
            exit 1
     fi
     
     sleep 15
done

echo "All done"
