
# feature-specific impact of CTCF motifs

1. tsv has been generated using the tsv-generator created for flanking experiment (the --num-weak argument has been temporarily turned off)

dots:
```python generate_symmetric_flank_df.py --num-strong 25 --orientation-string "><" --flank-range 20,20 --filename score_test --flank-spacer-sum 100000 --verbose```

boundaries:
```python generate_symmetric_flank_df.py --num-strong 25 --orientation-string "<>" --flank-range 20,20 --filename boundaries_test --flank-spacer-sum 100000 --verbose```



2. Running test experiment using prediction-generating script
```python virtual_symmetric_experiment_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 small_test.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o TEST --stats diffSCD --head-index 1 --model-index 0 --batch-size 4```


```python virtual_symmetric_experiment_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 score_test.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o 25motifs --stats diffSCD --head-index 1 --model-index 0 --batch-size 8```




10K experiment

dots:
```python generate_symmetric_flank_df.py --num-strong 1000 --orientation-string "><" --flank-range 20,20 --filename dots_10k --flank-spacer-sum 100000 --verbose```

boundaries:
```python generate_symmetric_flank_df.py --num-strong 1000 --orientation-string "<>" --flank-range 20,20 --filename boundaries_10k --flank-spacer-sum 100000 --verbose```

multuGPU
- dots
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 dots_10k.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /project/fudenber_735/data/dots_vs_boundaries/dots_10Kmotifs --stats diffSCD --head-index 1 --model-index 0 --batch-size 8 -p 10 --max_proc 10```

- boundaries
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 boundaries_10k.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /project/fudenber_735/data/dots_vs_boundaries/boundaries_10Kmotifs --stats diffSCD --head-index 1 --model-index 0 --batch-size 8 -p 10 --max_proc 10```


/project/fudenber_735/data/dots_vs_boundaries


python multiGPU-results_collector.py -o 10K -p 10

python /home1/smaruj/akita_utils/bin/insert_virtual_dots_vs_boundaries/multiGPU-results_collector.py -o boundaries_10Kmotifs -p 10