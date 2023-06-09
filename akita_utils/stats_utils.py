import numpy as np

# SCORES
# 1) Insulation score

def _single_map_insulation(target_map, window=10):
    """
    Calculate insulation in a window-size diamond around the central pixel.
    
    Parameters
    ------------
    target_map : numpy array
        Array with contact change maps predicted by Akita, usually of a size (512 x 512)
    window : int
        Size of a diamond taken into account in a metric calculation.

    Returns
    ---------
    score : ISN-window score for a given target
    """
    
    map_size = target_map.shape[0]
    if window > map_size // 2:
        raise ValueError("window cannot be larger than map")
    mid = map_size // 2
    lo = max(0, mid + 1 - window)
    hi = min(mid + window, map_size)
    score = np.nanmean(target_map[lo : (mid + 1), mid:hi])
    return score

def calculate_INS(map_matrix, window=10):
    """
    Calculate insulation in a window-size diamond around the central pixel
    for a set of num_targets contact difference maps.
    
    Parameters
    ------------
    map_matrix : numpy array
        Array with contact change maps predicted by Akita, usually of a size (512 x 512 x num_targets)
    window : int
        Size of a diamond taken into account in a metric calculation.

    Returns
    ---------
    scores : num_targets-long vector with INS-window scores
    """
    
    num_targets = map_matrix.shape[-1]
    scores = np.zeros((num_targets,))
    for target_index in range(num_targets):
        scores[target_index] = _single_map_insulation(map_matrix[:, :, target_index], window=window)
    return scores

# 2) SCD (Square Contact Differences)

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
    num_targets-long vector with SCD score calculated for each target.
    """
    if type(reference_map_matrix) != np.ndarray:
        return np.sqrt((map_matrix**2).sum(axis=(0,1)) * (1/2))
    else:
        return np.sqrt(
            ((map_matrix - reference_map_matrix) ** 2).sum(axis=(0,1))  * (1/2)
        )

# 3) dot score

def calculate_dot_score(map_matrix):
    """

    Parameters
    ------------
    map_matrix : numpy array
        Array with contact change maps predicted by Akita, usually of a size (512 x 512 x num_targets)

    Returns
    ---------
    """
    raise NotImplementedError("To be implemented")

# 4) flames score

def calculate_flames_score(map_matrix):
    """

    Parameters
    ------------
    map_matrix : numpy array
        Array with contact change maps predicted by Akita, usually of a size (512 x 512 x num_targets)

    Returns
    ---------
    """
    raise NotImplementedError("To be implemented")


# calculating all desired scores for a set of maps

def calculate_scores(stat_metrics, map_matrix, reference_map_matrix=None):
    """
    Calculates statistical metrics for a multiple-target prediction map-matrix.
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
    A dictionary with names of stat_metrics as keys and num_targets-long vector with scores calculated for each target as values.
    """
    scores = {}
    
    if "SCD" in stat_metrics:
        SCDs = calculate_SCD(map_matrix, None)
        scores["SCD"] = SCDs
    
    if "diffSCD" in stat_metrics:
        diffSCDs = calculate_SCD(map_matrix, reference_map_matrix)
        scores["diffSCD"] = diffSCDs
    
    if np.any((["INS" in i.split("-")[0] for i in stat_metrics])):
        for stat in stat_metrics:
            if stat.split("-")[0] == "INS":
                window = stat.split("-")[1]
                INS = calculate_INS(map_matrix, window)
                scores[stat] = INS
    
    # new scores will be added soon...
    
    return scores
