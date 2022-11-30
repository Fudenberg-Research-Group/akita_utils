#!/bin/bash

# ===============================================================================
# General allocatocations

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --time=24:00:00


# ===============================================================================
# Specific allocations

# SBATCH --account=fudenber_735
#SBATCH --account=qcb_640
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:1


module load gcc/8.3.0
module load cudnn/8.0.4.30-11.0

eval "$(conda shell.bash hook)"
conda activate basenji-gpu # this is the enviroment with installed dependencies


# ===============================================================================
# To explore background scores for different parameters, un comment this block

# python background_scores_exploration.py -f /project/fudenber_735/genomes/mm10/mm10.fa --chrom-data /project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed --head-index 1 --batch-size 5 --max-iters 10 -m -o experiment_figures  --stats SCD,INS-16,INS-64 /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 motif_positions.8.mm10.tsv


# ===============================================================================
# To explore effect of different mutation methods on the scores, un comment this block 

python mutation_method_exploration.py -f /project/fudenber_735/genomes/mm10/mm10.fa --chrom-data /project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed --head-index 1 --batch-size 5 --max-iters 10 -m -o experiment_figures  --stats SCD,INS-16,INS-64 /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 motif_positions.8.mm10.tsv



# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'