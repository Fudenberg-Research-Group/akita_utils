models_dir="/project/fudenber_735/tensorflow_models/akita/v2/models"
shuffled_seqs_fasta_file="/scratch1/kamulege/shuffled_seqs_copy.fa"
genome_fasta="/project/fudenber_735/genomes/hg38/hg38.fa"

models='4 5 6 7'

for model in $models
do
 python multiGPU_generate_shuffled_seqs_scores_from_fasta.py $models_dir $shuffled_seqs_fasta_file -o data -f $genome_fasta --head-index 0 --model-index $model -m --plot-freq 10 --batch-size 4 --stats SCD,MPS,CS -p 1 --max_proc 7 --time 00:10:00 &      
 sleep 15
 
done

echo All done