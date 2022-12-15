### Commands
- generating test-experiment table
```
python generate_fixed_core_df.py --fixed-core-num 2 --flank-sets-num 5 --orientation-string ">>" --flank-range 0,20 --backgrounds-indices 1 --filename test --verbose
```

```
python virtual_fixed_core_experiment_flanks.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 test.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o TEST --head-index 1 --model-index 1 --batch-size 4  --stats SCD
```




<!--         ```
        python multiGPU-virtual_symmetric_experiment_flanks.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/smaruj/akita_utils/bin/insert_virtual_flanks_experiment/data/2022-10-10_flank0-30_200motifs_divergent.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o 2022-10-19_flank0-30_200motifs_divergent --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16,INS-64,INS-256 -p 7 --max_proc 7
        ``` -->