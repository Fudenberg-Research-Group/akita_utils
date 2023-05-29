### Commands

#### Creating sample data

   - (1) create a table specifying the boundaries chosen and the CTCF motifs in the respective boundaries. The chosen ones in this particular experiment were Bonev mm10 ESC TAD boundaries of 10kb with a given insulation score and the CTCF motifs were overlaps with JASPAR CTCF motif
   
   - other considerations made were (1) remove motifs that overlap repeats (2) only consider motifs in boundaries where mutating *all* CTCFs has at least some effect (so that we aren’t studying regions where the impact of CTCF is either negligible for biological reasons, or issues with the model)
   
   - there is a script to help you create this table called `prepare_boundary_CTCF_tsv.sh` so you could fill in the required arguments and run the script like `sbatch prepare_boundary_CTCF_tsv.sh`. Alternatively you could pick the command from the script and feed in the command like with your custom arguments.  
   
#### Creating scores for the sample data

   - (2) generate deletion scores that correspond to the created tsv in step (1)
   
   - there is a script to help generate these scores called `akita_motif_scd.sh` so you could feed it your custom arguments but it was designed for multiple processes. In case you want a single process then you could just set `processes=1` or you can use the `akita_motif_scd.py` fed with your custom arguments.