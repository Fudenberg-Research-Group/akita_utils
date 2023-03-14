### Commands

   - generating tsv table for bulk simulations of shuffled seqs (mouse/human)
        
            python generate_shuffled_seqs_df.py -f /project/fudenber_735/genomes/mm10/mm10.fa -chrom_seq_bed_file /project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed
                
   - Akita predictions for shuffled seqs 
   
            python generate_scores_for_shuffled_seqs.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/background_seq_experiments/data/shuffled_seqs.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /home1/kamulege/akita_utils/bin/background_seq_experiments/data/pure_shuffled_seqs_scores --batch-size 4 --stats SCD,MPS,CS &
        
   - Akita predictions for shuffled seqs in bulk using multiple processes (considering improvements here)

            python multiGPU_generate_scores_for_shuffled_seqs.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/background_seq_experiments/data/shuffled_seqs.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /home1/kamulege/akita_utils/bin/background_seq_experiments/data/shuffled_seqs_scores_1 --head-index 1 --model-index 1 --batch-size 4 --stats SCD,MPS,CS -p 7 --max_proc 7 --time 01:00:00 &    
        
<!-- -------------------------------------------------------------------------------------------- -->

   - generating fasta files for given data/dataframe (mouse/human) 
        
            python generate_shuffled_seqs_fasta.py -f /project/fudenber_735/genomes/mm10/mm10.fa -tsv /home1/kamulege/akita_utils/bin/background_seq_experiments/data/flat_seqs_mouse.tsv -o /scratch1/kamulege/shuffled_seq_fasta_mouth.fa -sample 5 -specific_locus 0

   - generating scores from shuffled seqs fasta file 
   
           python generate_shuffled_seqs_scores_from_fasta.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /scratch1/kamulege/shuffled_seqs.fa --stats SCD,MPS,CS -o data/shuffled_seqs_from_fasta
       
   - generating scores from shuffled seqs fasta file using multiple processes
         
          python multiGPU_generate_shuffled_seqs_scores_from_fasta.py /project/fudenber_735/tensorflow_models/akita/v2/models /scratch1/kamulege/shuffled_seqs_copy.fa -o data -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 0 --model-index 1 -m --plot-freq 10 --batch-size 4 --stats SCD,MPS,CS -p 1 --max_proc 7 --time 00:10:00 & 
<!-- -------------------------------------------------------------------------------------------- -->

   - generating tsv table for creating final flat seqs (mouse/human)
   
          python generate_flat_seqs_df.py -f /project/fudenber_735/genomes/mm10/mm10.fa -chrom_seq_bed_file /project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed -o data/flat_seqs_mouse.tsv
      
   - generating flat seqs fasta file for further expts and sample pdfs
   
           python generate_flat_seqs.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /scratch1/kamulege/flat_seqs_x.tsv --stats SCD,MPS,CS -o data/flat_seqs_x
   
   - generating flat seqs fasta file for further expts and sample pdfs using multiple processes

            python multiGPU_generate_flat_seqs.py /project/fudenber_735/tensorflow_models/akita/v2/models data/flat_seqs_mouse.tsv -o data/flat_seqs_mouse -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 6 --batch-size 4 --max_iters 40 -s -m -p 1 --max_proc 7 --time 02:00:00 &