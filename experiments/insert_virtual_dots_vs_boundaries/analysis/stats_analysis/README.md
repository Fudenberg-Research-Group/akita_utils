# CTCF Binding Sites Analysis

This directory contains two Jupyter Notebooks that facilitate the analysis and visualization of the **feature-specific impact of CTCF motifs** experiment.

### 1. `preprocess_scores.ipynb`

This notebook processes the dot-scores and boundary-scores and saves it as a TSV file for further analysis. Follow the instructions to run the notebook and generate the required TSV file.

#### Instructions:

- Execute the notebook to preprocess the scores data.
- The processed data will be saved as a TSV file.

### 2. `analysis_plotting_averaged_scores.ipynb`

This notebook analyzes the processed scores data, focusing on the correlation between dot-scores and boundary-scores for CTCF binding sites inserted into background sequences. It generates visualizations and saves relevant plots in the `/plots` directory.

#### Instructions:

- Ensure the `preprocess_scores.ipynb` notebook has been run and the TSV file is available.
- Run `analysis_plotting_averaged_scores.ipynb` to visualize and analyze the data.
- Plots will be saved in the `/plots` directory for reference.

For any questions or assistance, feel free to reach out. Happy analyzing!
