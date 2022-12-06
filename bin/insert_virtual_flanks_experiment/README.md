### Commands
1. 2022-10-10_flank0-30_200motifs (all orientations)
    - generating tsv table
        ```
        python generate_symmetric_flank_df.py --num-strong 100 --num-weak 100 --orientation-string ">>" --flank-range 0,30 --backgrounds-indices 0,1,2 --filename 2022-10-10_flank0-30_200motifs_convergent.tsv --verbose
        ```
    - Akita predictions

        - orientation: right
        ```
        python multiGPU-virtual_symmetric_experiment_flanks.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/smaruj/akita_utils/bin/insert_virtual_flanks_experiment/data/2022-10-10_flank0-30_200motifs_right.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o 2022-10-19_flank0-30_200motifs_right --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16,INS-64,INS-256 -p 7 --max_proc 7
        ```
        - orientation: left
        ```
        python multiGPU-virtual_symmetric_experiment_flanks.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/smaruj/akita_utils/bin/insert_virtual_flanks_experiment/data/2022-10-10_flank0-30_200motifs_left.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o 2022-10-19_flank0-30_200motifs_left --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16,INS-64,INS-256 -p 7 --max_proc 7
        ```
        - orientation: convergent
        ```
        python multiGPU-virtual_symmetric_experiment_flanks.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/smaruj/akita_utils/bin/insert_virtual_flanks_experiment/data/2022-10-10_flank0-30_200motifs_convergent.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o 2022-10-19_flank0-30_200motifs_convergent --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16,INS-64,INS-256 -p 7 --max_proc 7
        ```
        - orientation: divergent
        ```
        python multiGPU-virtual_symmetric_experiment_flanks.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/smaruj/akita_utils/bin/insert_virtual_flanks_experiment/data/2022-10-10_flank0-30_200motifs_divergent.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o 2022-10-19_flank0-30_200motifs_divergent --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16,INS-64,INS-256 -p 7 --max_proc 7
        ```

2.  grid-plotting
(creating a grid of plots for one sequence and non-consecutive list of flanks)

- grid-plotting version of generate_symmetric_flank_df.py
```
python grid-plotting_generate_symmetric_flank_df.py --seq-index 11 --backgrounds-indices 1,2 --orientation-string ">>" --all-permutations --filename seq_11 --verbose
```

- grid-plotting version of virtual_symmetric_experiment_flanks.py
```
python grid_plotting_virtual_symmetric_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 seq_11.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o seq_11 --head-index 1 --model-index 1 --batch-size 4
```

3. Same core-motif experiment
(core motif of sequence 11)
```
python same-core_generate_symmetric_flank_df.py --num-strong 100 --num-weak 0 --orientation-string ">>" --flank-range 0,30 --backgrounds-indices 1,2 --filename 2022-11-8_flank0-30_motif11_100motifs_right --verbose
```

