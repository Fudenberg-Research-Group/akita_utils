#!/bin/bash
# A sample Bash script for generating flat seqs for different models

models_dir="/project/fudenber_735/tensorflow_models/akita/v2/models"
tsv_file="data/flat_seqs_mouse.tsv"
genome_fasta="/project/fudenber_735/genomes/mm10/mm10.fa"

models='1'

for model in $models
do
 python multiGPU_generate_flat_seqs.py $models_dir $tsv_file -o data -f $genome_fasta --head-index 0 --model-index $model --batch-size 4 --max_iters 4 -m -p 1 --max_proc 7 --time 01:30:00 
 sleep 15
done

echo All done
