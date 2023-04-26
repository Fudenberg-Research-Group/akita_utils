#!/bin/bash
# A sample Bash script for generating flat seqs for different models

models_dir="/project/fudenber_735/tensorflow_models/akita/v2/models"
tsv_file="data/flat_seqs_mouse.tsv"
genome_fasta="/project/fudenber_735/genomes/mm10/mm10.fa"
output_dir="/scratch1/kamulege/flat_seqs_mouse"
models='0 1 2 3 4 5 7'

for model in $models
do
 python multiGPU_generate_flat_seqs.py $models_dir $tsv_file -o $output_dir -f $genome_fasta --head-index 1 --model-index $model --batch-size 4 --max_iters 40 -p 10 --max_proc 7 --time 04:00:00 
 sleep 15
done

echo All done
