models_dir="/project/fudenber_735/tensorflow_models/akita/v2/models"
shuffled_seqs_tsv="/home1/kamulege/akita_utils/bin/background_seq_experiments/data/shuffled_seqs_human.tsv"
genome_fasta="/project/fudenber_735/genomes/hg38/hg38.fa"

python multiGPU_generate_scores_for_shuffled_seqs.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model0_best.h5 $shuffled_seqs_tsv -f $genome_fasta -o /home1/kamulege/akita_utils/bin/background_seq_experiments/data/shuffled_seqs_scores_human --head-index 0 --model-index 1 --batch-size 4 --stats SCD,MPS,CS -p 7 --max_proc 7 --time 01:00:00 &   