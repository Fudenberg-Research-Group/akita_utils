### Commands

   - generating tsv table
    
        `python generate_expt_composite_df.py -num_sites 2 -o data/promoter_test.tsv`
    
    
   - Akita predictions
    
        `python multiGPU_insert_promoter_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/insert_promoter_experiment/data/parameters_combo_no_swap.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /home1/kamulege/akita_utils/bin/insert_promoter_experiment/data/promoter_scores_no_swap_no_gene_flank_origninal --head-index 1 --model-index 1 --batch-size 4  --stats SCD -p 7 --max_proc 7 --time 5:00:00 &`
              
        
   - Akita predictions test
   
        `python insert_promoter_experiment_test.py /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f1c0/train/model1_best.h5 /home1/kamulege/akita_utils/bin/insert_promoter_experiment/data/promoters_2000.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /home1/kamulege/akita_utils/bin/insert_promoter_experiment/data/promoters_2000_plot_expt --head-index 1 --model-index 1 --batch-size 4  --stats SCD 
        
   - Akita predictions test multiple processes
      
      
    `python multiGPU_insert_promoter_experiment_test.py /project/fudenber_735/tensorflow_models/akita/v2/models /home1/kamulege/akita_utils/bin/insert_promoter_experiment/data/tail_sample_expt_for_plots.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa -o /scratch1/kamulege/tail_sample_expt_for_plots --head-index 1 --model-index 1 --batch-size 4  --stats SCD -p 10 --max_proc 7 --time 2:50:00 -m  --plot-freq 10 &`
  