### Commands
1. 2022-11-07_spacing_0.5Mb_100Smotifs (all orientations)
    - orientation: left
        - backgrounds: 1,2
        ```
        python generate_log-spacing_df.py --num-strong 100 --num-weak 0 --backgrounds-indices 1,2 --orientation-string "<<" --space-range 1,500000 --filename left_0.5Mb_strong100_bg12 --num_log-intervals 400 --verbose
        ```
        ```
        python multiGPU-virtual_symmetric_experiment_spacing.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 left_0.5Mb_strong100_bg12.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o left_0.5Mb_strong100_bg12 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14
        ```
        - backgrounds: 3,4
        ```
        python generate_log-spacing_df.py --num-strong 100 --num-weak 0 --backgrounds-indices 3,4 --orientation-string "<<" --space-range 1,500000 --filename left_0.5Mb_strong100_bg34 --num_log-intervals 400 --verbose
        ```
        ```
        python multiGPU-virtual_symmetric_experiment_spacing.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 left_0.5Mb_strong100_bg34.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o left_0.5Mb_strong100_bg34 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14
        ```

    - orientation: right
        - backgrounds: 1,2
        ```
        python generate_log-spacing_df.py --num-strong 100 --num-weak 0 --backgrounds-indices 1,2 --orientation-string ">>" --space-range 1,500000 --filename right_0.5Mb_strong100_bg12 --num_log-intervals 400 --verbose
        ```
        ```
        python multiGPU-virtual_symmetric_experiment_spacing.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 right_0.5Mb_strong100_bg12.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o right_0.5Mb_strong100_bg12 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14
        ```
        - backgrounds: 3,4
        ```
        python generate_log-spacing_df.py --num-strong 100 --num-weak 0 --backgrounds-indices 3,4 --orientation-string ">>" --space-range 1,500000 --filename right_0.5Mb_strong100_bg34 --num_log-intervals 400 --verbose
        ```
        ```
        python multiGPU-virtual_symmetric_experiment_spacing.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 right_0.5Mb_strong100_bg34.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o right_0.5Mb_strong100_bg34 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14
        ```
    - orientation: convergent
        - backgrounds: 1,2
        ```
        python generate_log-spacing_df.py --num-strong 100 --num-weak 0 --backgrounds-indices 1,2 --orientation-string "><" --space-range 1,500000 --filename convergent_0.5Mb_strong100_bg12 --num_log-intervals 400 --verbose
        ```
        ```
        python multiGPU-virtual_symmetric_experiment_spacing.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 convergent_0.5Mb_strong100_bg12.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o convergent_0.5Mb_strong100_bg12 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14
        ```
        - backgrounds: 3,4
        ```
        python generate_log-spacing_df.py --num-strong 100 --num-weak 0 --backgrounds-indices 3,4 --orientation-string "><" --space-range 1,500000 --filename convergent_0.5Mb_strong100_bg34 --num_log-intervals 400 --verbose
        ```
        ```
        python multiGPU-virtual_symmetric_experiment_spacing.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 convergent_0.5Mb_strong100_bg34.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o convergent_0.5Mb_strong100_bg34 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14
        ```
    - orientation: divergent
        - backgrounds: 1,2
        ```
        python generate_log-spacing_df.py --num-strong 100 --num-weak 0 --backgrounds-indices 1,2 --orientation-string "<>" --space-range 1,500000 --filename divergent_0.5Mb_strong100_bg12 --num_log-intervals 400 --verbose
        ```
        ```
        python multiGPU-virtual_symmetric_experiment_spacing.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 divergent_0.5Mb_strong100_bg12.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o divergent_0.5Mb_strong100_bg12 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14
        ```
        - backgrounds: 3,4
        ```
        python generate_log-spacing_df.py --num-strong 100 --num-weak 0 --backgrounds-indices 3,4 --orientation-string "<>" --space-range 1,500000 --filename divergent_0.5Mb_strong100_bg34 --num_log-intervals 400 --verbose
        ```
        ```
        python multiGPU-virtual_symmetric_experiment_spacing.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 divergent_0.5Mb_strong100_bg34.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o divergent_0.5Mb_strong100_bg34 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14
        ```

2. grid-plotting
(creating a grid of plots for one sequence and non-consecutive list of flanks)

- grid-plotting version of generate_symmetric_flank_df.py
```
python grid_plotting_generate_log-spacing.py --seq-index 2 --backgrounds-indices 1,2 --orientation-string ">>" --all-permutations --filename 2_grid --verbose

python grid_plotting_generate_log-spacing.py --seq-index 10 --backgrounds-indices 1,2 --orientation-string ">>" --all-permutations --filename 10_grid --verbose
```

- grid-plotting version of virtual_experiment.py
```
python grid_plotting_virtual_symmetric_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 2_grid.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o 2_grid --head-index 1 --model-index 1 --batch-size 4

python grid_plotting_virtual_symmetric_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 10_grid.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o 10_grid --head-index 1 --model-index 1 --batch-size 4

```