### Commands
1. 2022-09-30_flank0-30_10motifs (with both ">>" and "<<" orientations)
    - generating tsv table
        ```
        python generate_symmetric_flank_df.py --num-strong 10 --num-weak 10 --orientation-string ">>" --flank-range 0,30 --filename 2022-09-30_flank0-30_10motifs_left.tsv --verbose
        ```
    - Akita predictions
        ```
        python multiGPU-virtual_symmetric_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 ./data/2022-09-30_flank0-30_10motifs_left.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 8 --max_proc 16
        ```

2. 2022-10-10_flank0-30_200motifs (all orientations)
    - generating tsv table
        ```
        python generate_symmetric_flank_df.py --num-strong 100 --num-weak 100 --orientation-string ">>" --flank-range 0,30 --backgrounds-indices 0,1,2 --filename 2022-10-10_flank0-30_200motifs_convergent.tsv --verbose
        ```
    - Akita predictions
        ```
        python multiGPU-virtual_symmetric_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 ./data/2022-10-10_flank0-30_200motifs_convergent.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16,INS-64,INS-256 -p 16 --max_proc 16
        ```



## rerunning big experiments (2022-10-19_flank0-30_200motifs)
(using the same tsv tables)

- >>
```
python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/smaruj/akita_utils/bin/insert_virtual_flanks_experiment/data/2022-10-10_flank0-30_200motifs_right.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o 2022-10-19_flank0-30_200motifs_right --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16,INS-64,INS-256 -p 7 --max_proc 7
```
- <<
```
python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/smaruj/akita_utils/bin/insert_virtual_flanks_experiment/data/2022-10-10_flank0-30_200motifs_left.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o 2022-10-19_flank0-30_200motifs_left --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16,INS-64,INS-256 -p 7 --max_proc 7
```
- ><
```
python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/smaruj/akita_utils/bin/insert_virtual_flanks_experiment/data/2022-10-10_flank0-30_200motifs_convergent.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o 2022-10-19_flank0-30_200motifs_convergent --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16,INS-64,INS-256 -p 7 --max_proc 7
```
- <>
```
python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/smaruj/akita_utils/bin/insert_virtual_flanks_experiment/data/2022-10-10_flank0-30_200motifs_divergent.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o 2022-10-19_flank0-30_200motifs_divergent --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16,INS-64,INS-256 -p 7 --max_proc 7
```
