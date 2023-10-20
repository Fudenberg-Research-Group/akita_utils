
import pandas as pd
import numpy as np

def find_lower_boundary(request_exp_id, chunks_bounds):
    """
    Find the index of the biggest bound smaller than the requested experiment ID.

    This function takes a requested experiment ID and a list of chunk boundaries. It iterates
    through the boundaries and finds the index of the biggest boundary that is smaller than the
    requested experiment ID. This index indicates the corresponding chunk where the requested
    experiment is located.

    Parameters:
    - request_exp_id (int): Requested experiment ID to locate within the chunks.
    - chunks_bounds (list): List of chunk boundaries used for splitting the data.

    Returns:
    int: Index of the biggest boundary smaller than the requested experiment ID.
    """
    biggest_chunks_bound_index = 0
    for chunks_bound_index in range(len(chunks_bounds)):
        chunks_bound = chunks_bounds[chunks_bound_index]
        if chunks_bound < request_exp_id:
            biggest_chunks_bound_index = chunks_bound_index
        if chunks_bound > request_exp_id:
            return biggest_chunks_bound_index


def which_job(request_seq_id, request_bg_id, split_df, num_chunks):
    """
    Determine the job index for a requested experiment based on sequence and background IDs.

    This function calculates the experiment ID based on the provided sequence ID, background ID,
    and the total number of sites. It then splits the experiment data into chunks and determines
    the index of the chunk where the requested experiment is located.

    Parameters:
    - request_seq_id (int): Requested sequence ID.
    - request_bg_id (int): Requested background ID.
    - split_df (pd.DataFrame): DataFrame containing split experiment data.
    - nr_sites (int): Total number of sites.
    - num_chunks (int): Number of chunks used for splitting the data.

    Returns:
    int: Index of the chunk where the requested experiment is located.
    """
    request_exp_id = split_df.iloc[request_seq_id]["exp_id"]
    chunks_bounds = np.linspace(0, len(split_df), num_chunks + 1, dtype="int")
    
    return find_lower_boundary(request_exp_id, chunks_bounds)

