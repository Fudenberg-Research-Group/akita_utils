---
# Akita Utils

[![CI](https://github.com/Fudenberg-Research-Group/akita_utils/actions/workflows/main.yml/badge.svg)](https://github.com/Fudenberg-Research-Group/akita_utils/actions/workflows/main.yml)

## Description

Akita Utils is a set of functions to aid analysis of Akita. Akita is a deep learning CNN model that predicts contact frequency maps from the DNA sequence. This repository includes scripts and tools to efficiently analyze model's predictions.

Scripts used for cross-species AkitaV2 training and model weights are available from the [Basenji repository](https://github.com/calico/basenji/tree/master/manuscripts/akita/v2).

Akita Utils have been used to perform *in silico* experiments to extract the sequence contributions of CTCF to genome folding. The code for these experiments and computational analysis of Akita.V2’s predictions can be found in the [akitaX1-analyses repository](https://github.com/Fudenberg-Research-Group/akitaX1-analyses).

Preprint available here: *link to be added very soon*

## Installation

To install Akita Utils, run the following commands:

```bash
git clone https://github.com/Fudenberg-Research-Group/akita_utils.git
cd akita_utils
make install
```

Working environment specifying requirements can be installed as
```bash
conda env create -f basenji_py3.9_tf2.15.yml
```

Alternatively, install the requirements below:

- numpy
- pandas
- scipy
- tensorflow
- h5py
- bioframe
- seaborn

## Usage

For usage examples, please refer to the [akitaX1-analyses repository](https://github.com/Fudenberg-Research-Group/akitaX1-analyses) repository. 

We recommend starting with the following tutorials:
- `akitaXi-analyses/tutorials/disruption_tutorial.ipynb`
- `akitaXi-analyses/tutorials/insertion_tutorial.ipynb`
  
These tutorials will help you understand the basic functionalities and applications of akita_utils.

## Contact Information

Feedback and questions are appreciated. Please contact us at: fudenber at usc fullstop edu & smaruj at usc fullstop edu.

## Repository structure

- `./akita_utils`: Contains helper functions split by application, e.g., dna_utils, h5_utils, seq_genes.
- `./cli`: Contains a script for collecting h5 files output jobs.
- `./tests`: Contains test functions for the akita_utils functions.

## License

This project is licensed under the MIT License.
