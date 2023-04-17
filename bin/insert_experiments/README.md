### Commands

   - generating tss and other insertion tsv(s)
   
         python generate_tss_with_markers.py
       
       
   - generating tsv table
   
        - the json has a structure as in `data.json` file, for multiple values of a parameter put in a list
        
   
         python generate_expt_df_copy.py --json-file /home1/kamulege/akita_utils/bin/insert_experiments/data.json
                
        
   - Akita experiment predictions 
   
         python insert_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/insert_experiments/default_exp_data.tsv  -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --batch-size 4 -m -o ins_test  --stats SCD
        
        
   - Akita experiment predictions multiple processes
   
         python multiGPU_insert_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models /home1/kamulege/akita_utils/bin/insert_experiments/data/default_exp_data.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch1/kamulege/insert_experiments/boundary_promoter_enhancer_test --head-index 1 --model-index 1 --batch-size 4  --stats SCD -p 10 --max_proc 7 --time 0:30:00 &