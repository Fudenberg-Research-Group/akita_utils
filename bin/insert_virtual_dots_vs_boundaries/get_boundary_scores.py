
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

def calculate_SCD(map_matrix, reference_map_matrix=None):
    """
    Calculates SCD score for a multiple-target prediction matrix.
    If reference_matrix is not given, it is assumed that an insertion-into-background
    experiment has been performed so reference values are close to 0.

    Parameters
    ------------
    map_matrix : numpy array
        Array with contact change maps predicted by Akita, usually of a size (512 x 512 x num_targets)
    reference_map_matrix : numpy array
        Array with contact change maps predicted by Akita for a reference sequence, usually of a size (512 x 512 x num_targets)

    Returns
    ---------
    num_targets-long vector with SCD score calculated for eacg target.
    """
    
    if type(reference_map_matrix) != np.ndarray:
        map_matrix = map_matrix.astype("float32")
        return np.sqrt((map_matrix**2).sum(axis=(0, 1)) * (1 / 2))
    else:
        map_matrix = map_matrix.astype("float32")
        reference_map_matrix = reference_map_matrix.astype("float32")
        return np.sqrt(
            ((map_matrix - reference_map_matrix) ** 2).sum(axis=(0, 1))
            * (1 / 2)
        )


def get_map_matrix(hf, num_sequences, head_index, model_index, num_background):
    """averaged over targets"""
    
    num_targets = 6
    if head_index != 1:
        num_targets = 5
    
    map_size = np.array(hf[f"e0_h{head_index}_m{model_index}_t0_b0"]).shape[0]
    
    map_matrix = np.zeros((map_size, map_size, num_sequences, num_background))

    for seq_index in range(num_sequences):
        for target_index in range(num_targets):
            for background_index in range(num_background):
                map_matrix[:, :, seq_index, background_index] += np.array(hf[f"e{seq_index}_h{head_index}_m{model_index}_t{target_index}_b{background_index}"])
    
    map_matrix = map_matrix / num_targets
    return map_matrix


def get_reference_matrix(hf, head_index, model_index, num_background):
        
    num_targets = 6
    if head_index != 1:
        num_targets = 5
        
    map_size = np.array(hf[f"ref0_h{head_index}_m{model_index}_t0"]).shape[0]
    
    ref_map_matrix = np.zeros((map_size, map_size, num_background))
    
    for target_index in range(num_targets):
        for background_index in range(num_background):
            ref_map_matrix[:, :, background_index] += np.array(hf[f"ref{background_index}_h{head_index}_m{model_index}_t{target_index}"])
    
    ref_map_matrix = ref_map_matrix / num_targets
    return ref_map_matrix


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

