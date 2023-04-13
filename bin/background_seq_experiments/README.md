### Commands

   - (1) generating tsv table for creating final flat seqs (mouse/human)
   
          python generate_flat_seqs_df.py -f /project/fudenber_735/genomes/mm10/mm10.fa -seq_bed_file /project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed --output_filename data/flat_seqs_mouse.tsv
      
   - (2) generating flat seqs fasta file for further expts and sample pdfs
   
           python generate_flat_seqs.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /scratch1/kamulege/flat_seqs_x.tsv --stats SCD,MPS,CS -o data/flat_seqs_x -f /project/fudenber_735/genomes/mm10/mm10.fa
   
   - (3) generating flat seqs fasta file for further expts and sample pdfs using multiple processes

            python multiGPU_generate_flat_seqs.py /project/fudenber_735/tensorflow_models/akita/v2/models data/flat_seqs_mouse.tsv -o data/flat_seqs_mouse -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 6 --batch-size 4 --max_iters 40 -s -m -p 1 --max_proc 7 --time 02:00:00 &
          