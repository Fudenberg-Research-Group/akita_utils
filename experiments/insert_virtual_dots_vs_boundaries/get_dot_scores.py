
###########################################################
# Author: Paulina N. Smaruj                               #
# Date: 09.27.2023                                        #
# Version: 0                                              #
# Purpose: The script is supposed to iterate over .h5     #
# file with saved maps and calculate SCD and dot scores   #
# (dot-score, cross-score, x-score). The outputs are      #
# numpy arrays of the size = (nr_experiments, nr_targets) #
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

file_handler = logging.FileHandler('logs_get_dot_scores.log')
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


### dot-specific functions 

def get_bin(
    window_start,
    window_end,
    map_size=512,
    bin_size=2048,
    input_size=1310720,
):

    """
    Returns a list of bins overlapping the given window.

    Parameters
    ------------
    window_start : int
    window_end : int
        Start and end of the window that overlapping bins we want to find.
        Note, those values are in the local coordinates, so between 0 and input_size.
    map_size : int
        Size of the maps (equivalent to number of bins).
    bin_size : int
        The length of each bin in base pairs (bp).
    input_size : int
        Length of model's input sequence.

    Returns
    ---------
    bin_index : int
        The bin overlapping the given window.
    """

    window_size = window_end - window_start

    size_after_cropping = map_size * bin_size
    size_difference = input_size - size_after_cropping
    one_side_cropped_length = size_difference // 2

    corrected_window_start = window_start - one_side_cropped_length
    corrected_window_end = window_end - one_side_cropped_length

    first_bin_covered = corrected_window_start // bin_size
    last_bin_covered = corrected_window_end // bin_size
    
    assert first_bin_covered == last_bin_covered
    
    return first_bin_covered


def get_insertion_start_pos(insert_bp, spacer_bp, num_inserts, seq_length = 1310720):

    insert_plus_spacer_bp = insert_bp + 2 * spacer_bp
    multi_insert_bp = num_inserts * insert_plus_spacer_bp
    insert_start_bp = seq_length // 2 - multi_insert_bp // 2

    insertion_starting_positions = []
    for i in range(num_inserts):
        offset = insert_start_bp + i * insert_plus_spacer_bp + spacer_bp
        insertion_starting_positions.append(offset)
        
    return insertion_starting_positions


def map_sum(map_fragment):
    
    return (map_fragment**2).sum(axis=(0,1))


def get_lines(row_line, col_line, dot_band_size):
    
    upper_horizontal = row_line - (dot_band_size//2)
    lower_horizontal = row_line + (dot_band_size//2)
    if (dot_band_size % 2) == 1:
        lower_horizontal += 1
    
    left_vertical = col_line - (dot_band_size//2)
    right_vertical = col_line + (dot_band_size//2)
    if (dot_band_size % 2) == 1:
        right_vertical += 1
        
    return upper_horizontal, lower_horizontal, left_vertical, right_vertical


def nsq_dot_score(map_matrix, row_line, col_line, dot_band_size=3):
    
    upper_horizontal, lower_horizontal, left_vertical, right_vertical = get_lines(row_line, col_line, dot_band_size)
        
    # central, dot part
    dot_score = map_sum(map_matrix[upper_horizontal:lower_horizontal, left_vertical:right_vertical])
    return dot_score


def nsq_dot_x_score(map_matrix, row_line, col_line, dot_band_size=3, boundary_band_size=10):
    
    upper_horizontal, lower_horizontal, left_vertical, right_vertical = get_lines(row_line, col_line, dot_band_size)
    
    # central, dot part
    dot_size = dot_band_size**2
    dot_score = map_sum(map_matrix[upper_horizontal:lower_horizontal, left_vertical:right_vertical]) / dot_size
        
    # x-parts
    x_score = 0
    x_size = (boundary_band_size**2)*4
    for matrix_part in [map_matrix[upper_horizontal - boundary_band_size: upper_horizontal, left_vertical-boundary_band_size: left_vertical], 
                        map_matrix[upper_horizontal - boundary_band_size: upper_horizontal, right_vertical: right_vertical+boundary_band_size], 
                        map_matrix[lower_horizontal: lower_horizontal+boundary_band_size, left_vertical-boundary_band_size: left_vertical], 
                        map_matrix[lower_horizontal: lower_horizontal+boundary_band_size, right_vertical: right_vertical+boundary_band_size]]:
        x_score += map_sum(matrix_part)

    x_score = x_score / x_size
    
    return dot_score - x_score


def nsq_dot_score_cross(map_matrix, row_line, col_line, dot_band_size=3, boundary_band_size=10):
    
    upper_horizontal, lower_horizontal, left_vertical, right_vertical = get_lines(row_line, col_line, dot_band_size)
    
    # central, dot part
    dot_size = dot_band_size**2
    dot_score = map_sum(map_matrix[upper_horizontal:lower_horizontal, left_vertical:right_vertical]) / dot_size
    
    # cross-parts
    cross_score = 0
    cross_size = dot_band_size*boundary_band_size*4
    for matrix_part in [map_matrix[upper_horizontal - boundary_band_size: upper_horizontal, left_vertical:right_vertical], 
                        map_matrix[upper_horizontal:lower_horizontal, left_vertical-boundary_band_size: left_vertical], 
                        map_matrix[upper_horizontal:lower_horizontal, right_vertical: right_vertical+boundary_band_size], 
                        map_matrix[lower_horizontal: lower_horizontal+boundary_band_size, left_vertical:right_vertical]]:
        cross_score += map_sum(matrix_part)
        
    cross_score = cross_score / cross_size
    
    return dot_score - cross_score


### paths and parameters

# tsv with sites that the boundary experiment was performed with
# and implying the number of sites
tsv_filepath = "./new_correct_all_motifs_dot.tsv"
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
out_dir = f"/scratch2/smaruj/corrected_dots_vs_boundaries/dots_all_motifs_m{model_index}"

# dot-specific parameters
insert_len = 59
starts = get_insertion_start_pos(insert_bp=insert_len, spacer_bp=199980, num_inserts=2)
row_line, col_line = get_bin(starts[0], starts[0]+insert_len), get_bin(starts[1], starts[1]+insert_len)

# numpy arrays initialization
SCD = np.zeros((nr_sites, nr_targets))
dots = np.zeros((nr_sites, nr_targets))
x_score = np.zeros((nr_sites, nr_targets))
cross_score = np.zeros((nr_sites, nr_targets))


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
            dot = nsq_dot_score(map_matrix, row_line=row_line, col_line=col_line)
            x = nsq_dot_x_score(map_matrix, row_line=row_line, col_line=col_line)
            cross = nsq_dot_score_cross(map_matrix, row_line=row_line, col_line=col_line)
            
            SCD[exp_id, target_id] += SCDscore
            dots[exp_id, target_id] += dot
            x_score[exp_id, target_id] += x
            cross_score[exp_id, target_id] += cross
            
    job_h5_open.close()

    # overwrite npy file
    np.save(f"./m{model_index}_dot_scores_SCD.npy", SCD)
    np.save(f"./m{model_index}_dot_scores_dot.npy", dots)
    np.save(f"./m{model_index}_dot_scores_x.npy", x_score)
    np.save(f"./m{model_index}_dot_scores_cross.npy", cross_score)
    
    end = time.time()
    time_diff = round((end - start), 2)
    
    print(f"Working on job {job_index} has been finished.")
    print(f"This h5 file contains {num_maps}. Score calculation took {time_diff} seconds.")
    print()

    logger.info(f"Working on job {job_index} has been finished.")
    logger.info(f"This h5 file contains {num_maps}. Score calculation took {time_diff} seconds.")


