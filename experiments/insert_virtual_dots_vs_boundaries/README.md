
# feature-specific impact of CTCF motifs

This directory contains scripts and data for conducting the **feature-specific impact of CTCF motifs** experiment. Here's a brief guide to the logical sequence of steps and organization within this directory:

### 1. **Data Generation:**

- **Generating Input TSV:**
  - Use `generate_tsv_dot_boundary_scenario.py` to generate a TSV file containing CTCF coordinates. This file serves as input for subsequent experiments.
  - Output: TSV file with CTCF coordinates (found in `/input_data`).

### 2. **Virtual Symmetric Experiment:**

- **Running Virtual Symmetric Experiment:**
  - Utilize `virtual_symmetric_experiment_dots_vs_boundaries.py` for the main experiment comparing dot-scores vs. boundary-scores.
  - For multi-GPU setups, run `multiGPU_dots_vs_boundaries.py`.
  - Experiment setup parameters (e.g., model indices) can be configured.
  - Output: Dot and boundary scores in H5 format (located outside this directory).

### 3. **Automated Experiment Execution:**

- **Automated Experiment Scripts:**
  - Execute experiments automatically using `generate_boundary_experiment.sh` for boundary scenarios (`sbatch generate_boundary_experiment.sh`).
  - Run `generate_dot_experiment.sh` for dot scenarios (`sbatch generate_dot_experiment.sh`).
  - Parameters and setups can be modified within these script files.
  
### 4. **Data Organization:**

- **Input Data:**
  - Raw input TSV files for both dot and boundary experiments are stored in `/input_data`.

- **Processed Data:**
  - Processed dot and boundary scores in TSV format can be found in `/processed_data`.

### 5. **Data Analysis:**

- **Data Analysis:**
  - Navigate to the `/analysis` directory to perform data processing and analysis.
  - Use the analysis scripts and notebooks for further investigation and visualization.

For any questions or assistance, feel free to reach out. Happy experimenting!
