### Commands

- generating tsv table
    
        ```bash 
        python generate_promoter_expt_df.py
        ```
    
    
    
- Akita predictions

        ```bash
        python multiGPU_insert_promoter_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/insert_promoter_experiment/data/parameters_combo.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /home1/kamulege/akita_utils/bin/insert_promoter_experiment/data/promoter_scores --head-index 1 --model-index 1 --batch-size 4  --stats SCD -p 7 --max_proc 7 --time 15:00:00
        ```