### Commands

   - generating tsv table for bulk simulations of shuffled seqs
        
        `python generate_scores_for_shuffled_seqs_df.py` 
        

   - Akita predictions in for bulk simulations of shuffled seqs

        `python multiGPU_generate_scores_for_shuffled_seqs.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/background_seq_experiments/data/shuffled_seqs.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /home1/kamulege/akita_utils/bin/background_seq_experiments/data/shuffled_seqs_scores_1 --head-index 1 --model-index 1 --batch-size 4 --stats SCD,MPS,CS -p 7 --max_proc 7 --time 01:00:00 &`
        
        
   - generating tsv table for creating final flat seqs
   
      `python generate_flat_seqs_df.py`
     

   - generating flat seqs fasta file and pdfs

        `python multiGPU_generate_flat_seqs.py /project/fudenber_735/tensorflow_models/akita/v2/models data/flat_seqs.tsv -o data -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 6 --batch-size 4 --max_iters 40 -s -m -p 3 --max_proc 7 --time 02:00:00 &`
        
        
        
        
        `python generate_shuffled_seqs_scores_from_fasta.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /scratch1/kamulege/shuffled_seqs.fa --stats SCD,MPS,CS


python generate_shuffled_seqs_fasta.py -f /project/fudenber_735/genomes/mm10/mm10.fa -tsv /home1/kamulege/akita_utils/bin/background_seq_experiments/data/flat_seqs.tsv -o /scratch1/kamulege