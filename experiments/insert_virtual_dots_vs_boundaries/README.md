
# feature-specific impact of CTCF motifs

# TEST with saving maps
```python virtual_symmetric_experiment_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 ./ctcf_tsv/mismatched_boundary.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/BIGTEST_boundaries --save-maps```
```python virtual_symmetric_experiment_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 ./ctcf_tsv/mismatched_dot.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/BIGTEST_dots --stats SCD,dot-score,cross-score,x-score --save-maps```

# rerunning the experiment with all the ctcfs that went through filtering (without saving maps)
# model 1
multiGPU
- boundaries
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 ./ctcf_tsv/filtered_base_mouse_ctcf_boundary.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/all_dots_vs_boundaries/boundaries_all_motifs_m1 --batch-size 8 -p 10 --max_proc 10```

- dots
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 ./ctcf_tsv/filtered_base_mouse_ctcf_dot.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/all_dots_vs_boundaries/dots_all_motifs_m1 --stats SCD,dot-score,cross-score,x-score --batch-size 8 -p 10 --max_proc 10```

With maps:
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 ./ctcf_tsv/filtered_base_mouse_ctcf_boundary.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/all_dots_vs_boundaries/maps_boundaries_all_motifs_m1 --save-maps --batch-size 8 -p 20 --max_proc 20```

- dots
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 ./ctcf_tsv/filtered_base_mouse_ctcf_dot.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/all_dots_vs_boundaries/maps_dots_all_motifs_m1 --save-maps --stats SCD,dot-score,cross-score,x-score --batch-size 8 -p 20 --max_proc 20```


# model 2
multiGPU
- boundaries
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f2c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f2c0/train/model1_best.h5 ./ctcf_tsv/filtered_base_mouse_ctcf_boundary.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/all_dots_vs_boundaries/boundaries_all_motifs_m2 --batch-size 8 -p 20 --max_proc 20```

- dots
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f2c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f2c0/train/model1_best.h5 ./ctcf_tsv/filtered_base_mouse_ctcf_dot.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/all_dots_vs_boundaries/dots_all_motifs_m2 --stats SCD,dot-score,cross-score,x-score --batch-size 8 -p 20 --max_proc 20```


# testing collect_h5 for small set of maps
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 ./ctcf_tsv/boundary_filtered_mismatched_ctcf.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/MAP_TEST_boundaries --save-maps --batch-size 8 -p 2 --max_proc 2 --name bou```
```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 ./ctcf_tsv/dot_filtered_mismatched_ctcf.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/MAP_TEST_dots --save-maps --batch-size 8 -p 2 --max_proc 2 --name dot```


```python multiGPU_dots_vs_boundaries.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 ./ctcf_tsv/dot_filtered_mismatched_ctcf.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch2/smaruj/DUPA_dots --save-maps --batch-size 8 -p 2 --max_proc 2 --name dot```


Jobs collecting:
Stats:
```python collect_jobs_and_clean.py /scratch2/smaruj/all_dots_vs_boundaries/dots_all_motifs_m1```

Maps:
```python collect_jobs_and_clean.py /scratch2/smaruj/DUPA_dots -s```








