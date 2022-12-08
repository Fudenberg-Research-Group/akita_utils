### Commands
    - generating tsv table in /data
        ```python background_scores_exploration_bulk_df.py```
        
    - Akita predictions in /data

        ```python multiGPU_background_scores_exploration_bulk.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/background_seq_experiments/data/parameters_combo.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /home1/kamulege/akita_utils/bin/background_seq_experiments/data/background_scores --head-index 1 --model-index 1 --batch-size 4  --stats SCD -p 6 --max_proc 7```
