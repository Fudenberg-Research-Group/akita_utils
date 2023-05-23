
# Comparing SCD with and without background-prediction subtraction   

1. tsv has been generated using the tsv-generator created for flanking experiment (the --num-weak argument has been temporarily turned off)     
`python generate_symmetric_flank_df.py --num-strong 1000 --orientation-string ">>" --flank-range 20,20 --filename 10k_test --verbose`

2. Running test experiment using prediction-generating script     
`python virtual_symmetric_experiment_compare_SCD.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 test.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o TEST --head-index 1 --model-index 1 --batch-size 4`

3. Running multi-GPU script      
`python multiGPU-virtual_symmetric_compare_SCD.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/smaruj/akita_utils/bin/insert_virtual_compare_SCDs/10k_test.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o 10K --head-index 1 --model-index 0 --batch-size 8 -p 10 --max_proc 10`

4. Collecting results into one h5 file     
`python multiGPU-results_collector.py -o 10K -p 10`

