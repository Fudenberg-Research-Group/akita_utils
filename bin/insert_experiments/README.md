### Commands

   - generating tss and other insertion tsv(s)
   
         python generate_tss_with_markers.py
       
       
   - generating tsv table of inserts from provided json file
   
        - the json has a structure as in `data.json` file, each dictionary should correspond to a single dataset. single insert insert looks as below
```
    [        
    {   "feature_name": "name of your insert",
        "path": "/path/to/insert/dataframe.tsv",     
        "offset": list of offsets to test for your insert e.g [1120],        
        "flank_bp": list of flank basepairs to test for your insert e.g [0],        
        "orientation": list of orientations to test for your insert e.g [">"],
        "filter_out_inserts_with_ctcf_motifs": choice to filter CTCF motifs or not, true
    }
    ]
```                
         python generate_expt_df.py --json-file /home1/kamulege/akita_utils/bin/insert_experiments/data.json -o data/12_uniformly_selcted_model2_motifs.tsv --background_seqs 0 

   - For Akita experiment predictions on multiple processes, feed your custom parameters to `insert_expt_bash.sh` and run `sbatch insert_expt_bash.sh` 

   - For non multiple processes Akita experiment predictions, feed `insert_experiment.py` your custom parameters and run as shown below
   
         python insert_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/insert_experiments/one_strong_motif_in_different_backgrounds.tsv  -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --batch-size 4 -m -o /scratch1/kamulege/insert_experiments/one_strong_motif_in_different_backgrounds --stats SCD --background-file  /home1/kamulege/akita_utils/bin/background_seq_experiments/data/background_seqs/job0/background_seqs.fa
        
        
   