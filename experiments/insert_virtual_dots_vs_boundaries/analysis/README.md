# Analysis Directory README   

## Description     

This directory contains the analysis scripts and associated files for Akita Utils. The analysis outputs are saved in the form of statistical metrics and contact frequency matrices, both stored in h5 file format. Users can choose to save either statistical metrics or both statistical metrics and contact frequency maps during the d process.  

## Directory Structure    

- **stats_analysis/**: Notebooks for stats data processing and analysis.   
- **maps_analysis/**: Notebooks for maps data processing and analysis.   
- **README.md**: This README file.

## Usage

### Analyzing Statistical Metrics:

1. Navigate to the `stats_analysis` directory: `cd /stats_analysis`
2. Run the Jupyter notebooks in the following order:
   - **preprocess_scores.ipynb**: Generates a TSV file with statistical metrics averaged over targets (cell types) and background sequences.
   - **analysis_plotting_averaged_scores.ipynb**: Allows further analysis and visualization of the averaged statistical metrics.

The summarized TSV file with averaged metrics is located at `./../processed_data/filtered_base_mouse_ctcf_scored_and_averaged.tsv`.


### Analyzing Contact Frequency Maps:

1. Run the Jupyter notebooks (in any order):
   - **plotting_maps_by_seqid.ipynb**: Enables visualization and analysis of contact frequency maps based on sequence identifiers.
   - **plotting_scores_exploration.ipynb**: Provides exploratory analysis tools for the saved contact frequency maps.
  
## Note

For more detailed information about the project, dependencies, or acknowledgments, please refer to the main project README at the repository root.
