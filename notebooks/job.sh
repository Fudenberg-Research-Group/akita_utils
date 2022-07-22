#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --account=fudenber_735

# module purge
# module load gcc/8.3.0
# module load python/3.9.2

# module load anaconda3/2021.05

conda activate basenji-gpu

python background_scores_exploration.py -f /project/fudenber_735/genomes/mm10/mm10.fa --chrom-data /project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed --head-index 1 --batch-size 10 --max-iters 100 -m -o flat_test  --stats SCD,INS-16,INS-64 /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 motif_positions.8.mm10.tsv


python mutation_method_exploration.py -f /project/fudenber_735/genomes/mm10/mm10.fa --chrom-data /project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed --head-index 1 --batch-size 10 --max-iters 100 -m -o flat_test  --stats SCD,INS-16,INS-64 /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 motif_positions.8.mm10.tsv


# conda deactivate