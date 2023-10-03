
# feature-specific impact of CTCF motifs

################## diff way to save files #####################
after exp_id and seq_id has been added to the tsv

# model 1
multiGPU
- boundaries
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 new_correct_all_motifs_boundary.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/corrected_dots_vs_boundaries/boundaries_all_motifs_m1 --stats diffSCD --head-index 1 --model-index 1 --batch-size 8 -p 20 --max_proc 20```

- dots
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 new_correct_all_motifs_dot.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/corrected_dots_vs_boundaries/dots_all_motifs_m1 --stats diffSCD --head-index 1 --model-index 1 --batch-size 8 -p 20 --max_proc 20```


# model 2
multiGPU
- boundaries
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f2c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f2c0/train/model1_best.h5 new_correct_all_motifs_boundary.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/corrected_dots_vs_boundaries/boundaries_all_motifs_m2 --stats diffSCD --head-index 1 --model-index 2 --batch-size 8 -p 20 --max_proc 20```

- dots
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f2c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f2c0/train/model1_best.h5 new_correct_all_motifs_dot.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/corrected_dots_vs_boundaries/dots_all_motifs_m2 --stats diffSCD --head-index 1 --model-index 2 --batch-size 8 -p 20 --max_proc 20```




