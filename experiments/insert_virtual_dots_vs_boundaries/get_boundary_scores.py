
###########################################################
# Author: Paulina N. Smaruj                               #
# Date: 09.27.2023                                        #
# Version: 0                                              #
# Purpose: The script is supposed to iterate over .h5     #
# file with saved maps and calculate SCD (boundary        #
# strength) scores). The output is numpy array of the     #
# size = (nr_experiments, nr_targets)                     #
###########################################################


### imports ###

import pandas as pd
import pysam
import numpy as np
import akita_utils
import h5py
import time

from akita_utils.utils import ut_dense
from akita_utils.stats_utils import get_reference_map_matrix, get_map_matrix, plot_map, calculate_scores

# testing

import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs_get_boundary_scores.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

### functions ###



### paths and parameters

# tsv with sites that the boundary experiment was performed with
# and implying the number of sites
tsv_filepath = "./new_correct_all_motifs_boundary.tsv"
df = pd.read_table(tsv_filepath, sep="\t")

# human (0) or mouse (1)
head_index = 1
if head_index == 1:
    nr_targets = 6
elif head_index == 0:
    nr_targets = 5

nr_backgrounds = 10
nr_sites = len(df)

model_index = 2

# name of h5 file
name = "OUT.h5"
# number of jobs the process was split into
num_procs = 20
# dir with jobs directories
out_dir = f"/scratch2/smaruj/corrected_dots_vs_boundaries/boundaries_all_motifs_m{model_index}"

SCD = np.zeros((nr_sites, nr_targets))

for job_index in range(num_procs):
    print(f"Working on job {job_index}.")
    logger.info(f"Working on job {job_index}.")
        
    job_h5_file = f"{out_dir}/job{job_index}/{name}"

    # getting reference matrix
    job_h5_open = h5py.File(job_h5_file, "r")
    reference_matrix = get_reference_matrix(job_h5_open, head_index, model_index, nr_backgrounds)
    
    num_maps = 0
    start = time.time()

    # append to numpy array
    for key in job_h5_open.keys():

        # filtering only those keys with maps
        if (key[0] == "e" and key != "end" and key !="exp_id"):

            num_maps += 1
            
            identifiers = key.split("_")
            exp_id = int(identifiers[0][1:])
            seq_id = int(identifiers[1][1:])
            target_id = int(identifiers[4][1:])
            background_id = int(identifiers[5][1:])
            
            if exp_id % 1000 == 0 and target_id == 0:
                print(f"\tWorking on experiment: {exp_id}.")
                logger.info(f"\tWorking on experiment: {exp_id}.")
                
            map_matrix = np.array(job_h5_open[key])
            
            SCDscore = calculate_SCD(map_matrix, reference_map_matrix=reference_matrix[:, :, background_id])
            
            SCD[exp_id, target_id] += SCDscore
    
    job_h5_open.close()

    # overwrite npy file
    np.save(f"./m{model_index}_boundary_scores.npy", SCD)
    
    end = time.time()
    time_diff = round((end - start), 2)
    
    print(f"Working on job {job_index} has been finished.")
    print(f"This h5 file contains {num_maps}. Score calculation took {time_diff} seconds.")
    print()

    logger.info(f"Working on job {job_index} has been finished.")
    logger.info(f"This h5 file contains {num_maps}. Score calculation took {time_diff} seconds.")

