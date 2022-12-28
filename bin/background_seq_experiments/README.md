### Commands

   - generating tsv table for bulk simulations of shuffled seqs
        
        `python generate_scores_for_shuffled_seqs_df.py` 
        

   - Akita predictions in for bulk simulations of shuffled seqs

        `python multiGPU_generate_scores_for_shuffled_seqs.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/background_seq_experiments/data/shuffled_seqs.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /home1/kamulege/akita_utils/bin/background_seq_experiments/data/shuffled_seqs_scores --head-index 1 --model-index 1 --batch-size 4 --stats SCD,MSS,MPS -p 6 --max_proc 7 --time 01:00:00 &`
        
        
        
        
        
   - generating tsv table for creating final flat seqs
   
      `python generate_flat_seqs_df.py`
     

   - generating flat seqs fasta file and pdfs

        `python multiGPU_generate_flat_seqs.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/background_seq_experiments/data/flat_seqs.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /home1/kamulege/akita_utils/bin/background_seq_experiments/data/flat_seqs --head-index 1 --model-index 1 --batch-size 4 --max_iters 40 -s -m -p 1 --max_proc 7 --time 02:00:00 &`
        
