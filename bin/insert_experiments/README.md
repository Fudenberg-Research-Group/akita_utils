### Commands

   - generating tss and other insertion tsv(s)
   
         python generate_tss_with_markers.py
       
       
   - generating tsv table of inserts from provided json file
   
        - the json has a structure as in `data.json` file, each dictionary should correspond to a single dataset. single insert insert looks as below

    [        
    {   "feature_name": "name of your insert",
        "path": "/path/to/insert/dataframe.tsv",     
        "offset": list of offsets to test for your insert e.g [1120],        
        "flank_bp": list of flank basepairs to test for your insert e.g [0],        
        "orientation": list of orientations to test for your insert e.g [">"],
        "filter_out_inserts_with_ctcf_motifs": choice to filter CTCF motifs or not, true
    }
    ]
                
         python generate_expt_df.py --json-file /home1/kamulege/akita_utils/bin/insert_experiments/data.json -o zero_motif_promoters_5000.tsv
                
        
   - Akita experiment predictions 
   
         python insert_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/insert_experiments/hetero_and_boundary.tsv  -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --batch-size 4 -m -o /home1/kamulege/akita_utils/bin/insert_experiments/hetero_and_boundary  --stats SCD
        
        
   - Akita experiment predictions multiple processes
   
         python multiGPU_insert_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models /home1/kamulege/akita_utils/bin/insert_experiments/zero_motif_promoters_5000.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch1/kamulege/insert_experiments/zero_motif_promoters_5000_test --head-index 1 --model-index 1 --batch-size 4  --stats SCD -p 1 --max_proc 7 --time 0:10:00 &