### Commands
- generating test-experiment table
```
python generate_fixed_core_df.py --fixed-core-num 2 --flank-sets-num 5 --orientation-string ">>" --flank-range 0,20 --backgrounds-indices 1 --filename test --verbose
```
- prediction
```
python virtual_fixed_core_experiment_flanks.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 test.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o TEST --head-index 1 --model-index 1 --batch-size 4  --stats SCD
```
- multi-GPU prediction
```
python multiGPU-virtual_fixed_core_experiment_flanks.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 test.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o TEST2 --head-index 1 --model-index 1 --batch-size 4  --stats SCD -p 7 --max_proc 7
```


Experiment 1:
- 10 strong cores (between 96 and 99 percentile)
- 10 random flanks for each (between 75 and 99 percentile)
(for the above threshold I changes generate_fixed_core_df.py script manually)
- backgrounds: 1,2,3 
```
python generate_fixed_core_df.py --fixed-core-num 10 --flank-sets-num 10 --orientation-string ">>" --flank-range 0,30 --backgrounds-indices 1,2,3 --filename 10vs10 --verbose
```

```
python multiGPU-virtual_fixed_core_experiment_flanks.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 10vs10.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o TEST2 --head-index 1 --model-index 1 --batch-size 4  --stats SCD -p 7 --max_proc 7 
```
```
-r
```

Experiment 2:
- 10 strong cores (between 96 and 99 percentile)
- 10 random flanks for each (between 75 and 99 percentile)
- backgrounds: 1,2
```
python generate_fixed_core_df.py --fixed-core-num 15 --flank-sets-num 15 --orientation-string ">>" --flank-range 0,30 --backgrounds-indices 1,2,3 --filename 15vs15strong --verbose
```
```
python multiGPU-virtual_fixed_core_experiment_flanks.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 15vs15strong.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o TEST --head-index 1 --model-index 1 --batch-size 4  --stats SCD -p 7 --max_proc 7 
```