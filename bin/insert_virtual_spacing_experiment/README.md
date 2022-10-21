### Commands
1. 2022-10-11_200_10W_10S_spacings_0-1000_right_bg1234 (only the "<<" orientations)
    - generating tsv table
        ```
        python generate_log-spacing_df.py --num-strong 10 --num-weak 10 --backgrounds-indices 1,2,3,4 --orientation-string "<<" --filename 2022-10-11_200_10W_10S_spacings_0-1000_right_bg1234 --verbose
        ```
    - Akita predictions
        ```
        python multiGPU-virtual_symmetric_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 2022-10-11_200_10W_10S_spacings_0-1000_right_bg1234.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 16 --max_proc 16
        ```

2. 2022-10-12_200_100W_100S_spacings_0-1000_(orientation)_bg12 (all orientations)
    - generating tsv table
        ```
        python generate_log-spacing_df.py --num-strong 100 --num-weak 100 --orientation-string ">>" --backgrounds-indices 1,2 --filename 2022-10-12_200_100W_100S_spacings_0-1000_right_bg12 --verbose
        ```
    - Akita predictions
        ```
        python multiGPU-virtual_symmetric_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 2022-10-12_200_100W_100S_spacings_0-1000_right_bg12.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 16 --max_proc 16
        ```
3. 
    - generating tsv table
        ```
        python generate_log-spacing_df.py --num-strong 3 --num-weak 0 --backgrounds-indices 1,2,3,4 --orientation-string ">>" --space-range 1,1000000 --num_log-intervals 1000 --filename 2022-10-15_0W_3S_spacings_0-1000000_right_bg1234 --verbose
        ```
    - Akita predictions
        ```
        python multiGPU-virtual_symmetric_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 2022-10-11_200_10W_10S_spacings_0-1000_right_bg1234.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 16 --max_proc 16
        ```
python generate_log-spacing_df.py --num-strong 2 --num-weak 2 --backgrounds-indices 1,2 --orientation-string "<<" --filename movie_test --verbose

python virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 first_movie.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -m -o first_movie_2 --plot-freq 2 --head-index 1 --model-index 1 --batch-size 4 --stats SCD

python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/smaruj/akita_utils/bin/insert_virtual_spacing_experiment/first_movie.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -m -o /home1/smaruj/akita_utils/bin/insert_virtual_spacing_experiment/first_movie --head-index 1 --model-index 1 --batch-size 4  --stats SCD -p 10 --max_proc 10


### further

# <<
python generate_log-spacing_df.py --num-strong 100 --num-weak 100 --backgrounds-indices 1,2 --orientation-string "<<" --space-range 1,500000 --filename out_100_12 --num_log-intervals 600 --verbose

python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 out_100_12_only_strong.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o out_100_12_only_strong --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14

python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 out_100_34_only_strong.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o out_100_34_only_strong --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14

# >>
python generate_log-spacing_df.py --num-strong 100 --num-weak 100 --backgrounds-indices 3,4 --orientation-string ">>" --space-range 1,500000 --filename out_right --num_log-intervals 600 --verbose

python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 right_0.5Mb_strong100_bg12.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o right_0.5Mb_strong100_bg12 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14

python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 right_0.5Mb_strong100_bg34.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o right_0.5Mb_strong100_bg34 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14




### movies
python generate_log-spacing_df.py --num-strong 30 --num-weak 30 --backgrounds-indices 1 --orientation-string "<<" --space-range 1,500000 --filename out_good_seqs_movies --num_log-intervals 200 --verbose


# 21 done
python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 sequence_21.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -m -o sequence_21 --plot-freq 1 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14

# 22
python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 sequence_22.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -m -o sequence_22 --plot-freq 1 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14

# 23
python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 sequence_23.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -m -o sequence_23 --plot-freq 1 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14

# 25
python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 sequence_25.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -m -o sequence_25 --plot-freq 1 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14

# 27 done
python multiGPU-virtual_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 sequence_27.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -m -o sequence_27 --plot-freq 1 --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 14 --max_proc 14




# checking insertion points
python virtual_symmetric_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 test.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 1 --batch-size 4 --stats SCD