### Commands
<<<<<<< HEAD

    - generating tsv table for bulk simulations
        
        `python background_scores_exploration_bulk_df.py` 
        

    - Akita predictions in for bulk simulations

        `python multiGPU_background_scores_exploration_bulk.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/background_seq_experiments/data/parameters_combo.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /home1/kamulege/akita_utils/bin/background_seq_experiments/data/background_scores --head-index 1 --model-index 1 --batch-size 4  --stats SCD -p 6 --max_proc 7 --time 01:00:00`
        
        
        
        
        
    - generating tsv table for creating final flat maps
    
     got to `background_scores_exploration_bulk_df.py` and put in desired parameters then create tsv file same way as before. 
     (TIP) give it a different name to one used in bulk simulation, here i used `background_seq.tsv`
     

    - generating flat maps fasta file and pdfs

        `python multiGPU_generate_flat_background.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/background_seq_experiments/data/background_seq.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /home1/kamulege/akita_utils/bin/background_seq_experiments/data/background_seqs --head-index 1 --model-index 1 --batch-size 4 -s -m --stats SCD -p 1 --max_proc 7 --time 01:00:00`
        
=======
    - generating tsv table in /data
        ```python background_scores_exploration_bulk_df.py```
        
    - Akita predictions in /data

        ```python multiGPU_background_scores_exploration_bulk.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/background_seq_experiments/data/parameters_combo.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /home1/kamulege/akita_utils/bin/background_seq_experiments/data/background_scores --head-index 1 --model-index 1 --batch-size 4  --stats SCD -p 6 --max_proc 7```
>>>>>>> ca6d3bf7c1c3741116e21503d5e4fcc00c138994
