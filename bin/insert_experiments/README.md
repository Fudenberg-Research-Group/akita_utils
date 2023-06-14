# Commands

###   - generating tss and other insertion tsv(s)
   
         python preprocess_tss_to_tsv.py
       
       
###   - generating tsv table of inserts from provided json file
   
- the json has a structure as in `data.json` file, each dictionary element in the list should correspond to a single dataset. A single insert may look as below
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
- After structuring the data.json file, the you could run the `generate_expt_df.py` script with the created json file, output_file path and the particular background_seqs_indices you will try. e.g

         python generate_expt_df.py --json-file /path/to/data.json -o path/to/output_filename.tsv --background_seqs 0 

###   - To generate Akita predictions on multiple processes, feed your custom parameters to `multi-model_modular-insert.sh` and run `sbatch multi-model_modular-insert.sh` 


###   - For more hands on experience, below are the unwrapped commands for single and multiple processes Akita experiment predictions, feed `insert_experiment.py` your custom parameters and run as shown below
   
         python insert_experiment.py /path/to/a/single/akita/v2/model/train/params.json /path/to/a/single/akita/v2/model.h5 /path/to/experiment.tsv  -f /path/to/genomes/mm10/mm10.fa --head-index 1 --batch-size 4 -m -o /path/to/output/directory --stats SCD --background-file  /path/to/background_seqs.fa
        
        
   