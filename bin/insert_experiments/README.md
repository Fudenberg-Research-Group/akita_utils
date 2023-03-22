### Commands

   - generating tss tsv
   
       `python generate_tss_with_markers.py`
       
       
   - generating tsv table
   
        `python generate_promoter_df.py`
                
        
   - Akita predictions test
   
        `python insert_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/insert_experiments/only_ctcfs.tsv.tsv  -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --batch-size 4 -m -o ins_test  --stats SCD,INS-16,INS-64`
        
        
   - Akita predictions test multiple processes
   
        `python multiGPU_insert_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models /home1/kamulege/akita_utils/bin/insert_experiments/data/Enhancers_alone.tsv.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch1/kamulege/insert_experiments/Enhancers_alone --head-index 1 --model-index 1 --batch-size 4  --stats SCD -p 10 --max_proc 7 --time 0:30:00 &`