### Commands

   - (1) create a table specifying how flat sequences should be generated. Includes: mutation method (default is "permute_whole_seq"), shuffle parameter, score thresholds (MPS_thresh, SCD_thresh) and ctcf_detection_thresholds. Can be used with either human or mouse genomes.
   
          python generate_flat_seqs_df.py -f /project/fudenber_735/genomes/mm10/mm10.fa -seq_bed_file /project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed --output_filename data/flat_seqs_mouse.tsv --num_backgrounds 100
      
   - (2) generate flat seqs fasta file. These sequences can be used as background sequences for virtual insertion experiments. Optionally plots pdfs of predictions for these flat sequences.
   
           python generate_flat_seqs.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/background_seq_experiments/data/flat_seqs_mouse.tsv --stats SCD,MPS,CS -o /scratch1/kamulege/flat_seqs_test -f /project/fudenber_735/genomes/mm10/mm10.fa
   
   - (3) generating flat seqs fasta file for further expts and sample pdfs using multiple processes

            python multiGPU_generate_flat_seqs.py /project/fudenber_735/tensorflow_models/akita/v2/models data/flat_seqs_mouse.tsv -o /scratch1/kamulege/flat_seqs_mouse -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 6 --batch-size 4 --max_iters 40 -p 10 --max_proc 7 --time 04:00:00 
          