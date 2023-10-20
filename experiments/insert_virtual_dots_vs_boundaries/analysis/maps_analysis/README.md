# Maps Analysis Notebooks

## Overview

Welcome to the map data analysis folder! This repository contains two Jupyter Notebooks designed to facilitate efficient exploration and analysis of our generated data. The new developmental decisions implemented in the data generation process are detailed below, providing a streamlined approach to handle extensive output data.

### Developmental Decisions:

1. **Job-Specific Output**: The output data is organized into job-specific directories (e.g., `/job0`) to maintain a structured data hierarchy, ensuring easier management and accessibility.

2. **Selective Access with Sequence ID and Background ID**: Utilizing unique sequence IDs and background IDs, we can pinpoint the specific job containing the predicted map. This selective access enhances data retrieval efficiency.

3. **Efficient Data Retrieval**: Instead of combining all map outputs into a single large file, we access only the necessary `MAPS_OUT.h5` file within the designated job directory, optimizing both time and computational resources.

## Notebooks

### 1. `plotting_maps_by_seqid.ipynb`

This notebook is designed to plot maps corresponding to a set of requested sequence IDs. By leveraging the selective access approach, you can efficiently visualize specific maps without the need to sift through extensive datasets.

#### Instructions:

- Provide the desired sequence IDs as input.
- Run the notebook to generate visualizations of the corresponding maps.

### 2. `plotting_scores_exploration.ipynb`

This notebook facilitates the exploration of maps with statistical metrics falling within a specified range. It enables you to analyze maps based on their scores, e.g. boundary SCD, or cross-score.

#### Instructions:

- Define the desired score range and other metrics criteria.
- Run the notebook to generate plots showcasing maps that meet the specified criteria.

## Getting Started

To begin your analysis, simply open the respective notebook and follow the instructions provided within. If you encounter any issues or have specific questions, feel free to reach out for assistance.

Happy analyzing!
