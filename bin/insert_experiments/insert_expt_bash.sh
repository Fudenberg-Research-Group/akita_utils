#!/bin/bash

# Load configuration file
source config.ini

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

if ! ls -1 $background_seqs >/dev/null 2>&1; then
    echo "Error: no background sequences fasta files found."
    exit 1
fi

if [ ! -d "$output" ]; then
    echo "Output directory does not exist. Creating directory..."
    mkdir -p "$output"
fi


set -f

# Run experiments for each model
for model in $models
do
    python multiGPU_insert_experiment.py $models_dir $tsv_file -f $genome_fasta -o $output --head-index $head --model-index $model --batch-size $batch_size -p $processes --max_proc $max_proc --stats $stats --time $time --background-file $background_seqs 

    if [ $? -ne 0 ]; then
        echo "Error: failed to run experiments for model $model."
        exit 1
    fi

    sleep 15
done

set +f


echo "All done."
