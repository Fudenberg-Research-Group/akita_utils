### Commands
1. Small Experiment (with both ">>" and "<<" orientations)
    - generating tsv table
        ```
        python generate_symmetric_flank_df.py --num-strong 10 --num-weak 10 --orientation-string ">>" --flank-range 0,30 --number-backgrounds 10 --filename right_out.tsv --verbose
        ```
    - Akita predictions
        ```
        python multiGPU-virtual_symmetric_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 right_out.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 8 --max_proc 16
        ```

2. Big Experiment
    - generating tsv table
        ```
        python generate_symmetric_flank_df.py --num-strong 100 --num-weak 100 --orientation-string ">>" --flank-range 0,30 --number-backgrounds 3 --filename right_big_experiment.tsv --verbose
        ```
    - Akita predictions
        ```
        python multiGPU-virtual_symmetric_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 right_big_experiment.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16,INS-64,INS-256 -p 16 --max_proc 16
        ```