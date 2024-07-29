---
# Akita Utils

[![CI](https://github.com/Fudenberg-Research-Group/akita_utils/actions/workflows/main.yml/badge.svg)](https://github.com/Fudenberg-Research-Group/akita_utils/actions/workflows/main.yml)

Awesome akita_utils created by Fudenberg-Research-Group: [Fudenberg Team](https://fudenberg.team/)

## Description

Akita Utils is a set of functions for the analysis of Akita.V2 predictions. Akita is a deep learning CNN model that predicts contact frequency maps from the DNA sequence. This repository includes scripts and tools to efficiently analyze model's predictions.

Scripts used for cross-species AkitaV2 training and model weights are available from the [Basenji repository](https://github.com/calico/basenji/tree/master/manuscripts/akita/v2).

Akita Utils has been used to perform experiments on extracting sequence contributions of CTCF to genome folding. The code for these experiments and computational analysis of Akita.V2’s predictions can be found in the [akitaX1-analyses repository](https://github.com/Fudenberg-Research-Group/akitaX1-analyses).

Preprint available here: *link to be added very soon*

## Installation

To install Akita Utils, run the following commands:

```bash
git clone https://github.com/Fudenberg-Research-Group/akita_utils.git
cd akita_utils
make install
```

Working environment specifying requirements can be installed as
```py
conda env create -f basenji_py3.9_tf2.15.yml
```

Ensure you have the following dependencies installed:

- numpy
- pandas
- scipy
- tensorflow
- h5py
- bioframe
- seaborn

## Usage

Here are some examples of how to use Akita Utils in your projects:

```py
import akita_utils
from akita_utils.plot_utils import plot_map
from akita_utils.stats_utils import calculate_scores
```

You can also run Akita Utils from the command line:

```bash
$ python -m akita_utils
#or
$ akita_utils
```

## Contact Information

Feedback and questions are appreciated. Please contact us at: fudenber at usc fullstop edu & smaruj at usc fullstop edu.

## Table of Contents

- `./akita_utils`: Contains helper functions split by application, e.g., dna_utils, h5_utils, seq_genes.
- `./cli`: Contains a script for collecting h5 files output jobs.
- `./tests`: Contains test functions for the akita_utils functions.

## License

This project is licensed under the MIT License.
