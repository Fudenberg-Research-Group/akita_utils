#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --account=fudenber_735

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:1

module load gcc/8.3.0
module load cudnn/8.0.4.30-11.0

eval "$(conda shell.bash hook)"

conda activate basenji

python /home1/smaruj/akita_utils/bin/akita_padding_v2.py -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --batch-size 4 -o /home1/smaruj/akita_utils/bin/ins_test --stats SCD,INS-16,INS-64 /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 --background-file /project/fudenber_735/tensorflow_models/akita/v2/analysis/background_seqs.fa --one_side_radius 100 --num_background 1 --paddings_start 0 --paddings_end 3 --table /home1/smaruj/akita_utils/bin/test_4motifs.csv
