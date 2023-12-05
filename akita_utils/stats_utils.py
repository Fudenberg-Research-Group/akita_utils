
import numpy as np
from akita_utils.utils import ut_dense
import matplotlib.pyplot as plt
import seaborn as sns

# STANDARDS
MOTIF_LEN = 19
FLANK_LEN = 20
INSERT_LEN = 2 * FLANK_LEN + MOTIF_LEN

# MATRIX TRANSFORMATION

def get_reference_map_matrix(hf, head_index, model_index, num_background, diagonal_offset=2):
    """
    Collect all the reference predictions from the h5 file and transform it to 
    log(exp/obs) maps returned as an array of size (num_background, map_size, map_size, num_targets).

    Parameters
    ------------
    hf : h5 object
        Opened h5 file.
    head_index : int
        Head index used to get a prediction (Mouse: head_index=1; Human: head_index=0).
    model_index : int
        Index of one of 8 models that has been used to make predictions (an index between 0 and 7).
    num_background : int
        Number of background sequences used in the experiment.
    diagonal_offset : int
        Number of diagonals that are added as zeros in the conversion.
        Typically 2 diagonals are ignored in Hi-C data processing.

    Returns
    ---------
    ref_map_matrix : numpy array
        An array of size (num_background, map_size, map_size, num_targets) with log(exp/obs) maps.
    """
    num_targets = 6
    if head_index != 1:
        num_targets = 5

    map_size = len(ut_dense(hf[f"refmap_h{head_index}_m{model_index}"][0,:,:], diagonal_offset))
    ref_map_matrix = np.zeros((num_background, map_size, map_size, num_targets))

    for background_index in range(num_background):

        preds_matrix = hf[f"refmap_h{head_index}_m{model_index}"][background_index,:,:]
        map_matrix = ut_dense(preds_matrix, diagonal_offset)
        
        ref_map_matrix[background_index, :, :, :] += map_matrix
    
    return ref_map_matrix


def get_map_matrix(hf, head_index, model_index, num_experiments, diagonal_offset=2):
    """
    Collect all the experimental predictions from the h5 file and transform it to
    log(exp/obs) maps returned as an array of size (num_experiments, map_size, map_size, num_targets)).

    Parameters
    ------------
    hf : h5 object
        Opened h5 file.
    head_index : int
        Head index used to get a prediction (Mouse: head_index=1; Human: head_index=0).
    model_index : int
        Index of one of 8 models that has been used to make predictions (an index between 0 and 7).
    num_experiments : int
        Number of experiments (nr_sequences x nr_backgrounds used)
    diagonal_offset : int
        Number of diagonals that are added as zeros in the conversion.
        Typically 2 diagonals are ignored in Hi-C data processing.

    Returns
    ---------
    exp_map_matrix : numpy array
        An array of size (num_experiments, map_size, map_size, num_targets)) with log(exp/obs) maps.
    """
    
    num_targets = 6
    if head_index != 1:
        num_targets = 5
    
    map_size = len(ut_dense(hf[f"map_h{head_index}_m{model_index}"][0,:,:], diagonal_offset))
    
    exp_map_matrix = np.zeros((num_experiments, map_size, map_size, num_targets))

    for exp_index in range(num_experiments):
        
        preds_matrix = hf[f"map_h{head_index}_m{model_index}"][exp_index,:,:]
        map_matrix = ut_dense(preds_matrix, diagonal_offset)
        
        exp_map_matrix[exp_index, :, :, :] += map_matrix

    return exp_map_matrix


# SINGLE-MAP PLOTTING FUNCTION

def plot_map(matrix, vmin=-0.6, vmax=0.6, width=5, height=5, palette="RdBu_r"):
    """
    Plots a 512x512 log(obs/exp) map.

    Parameters
    ------------
    matrix : numpy array
        Predicted log(obs/exp) map.
    vmin : float
    vmax : float
        Minimum and maximum in the colormap scale.
    width : int
    height : int
        Width and height of a plotted map.
    """

    fig = plt.figure(figsize=(width, height))

    sns.heatmap(
        matrix,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        cmap=palette,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )
    plt.show()


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
    scores : num_targets-long vector with ISN-window scores
    """

    num_targets = map_matrix.shape[-1]
    scores = np.zeros((num_targets,))
    for target_index in range(num_targets):
        scores[target_index] = _single_map_insulation(
            map_matrix[:, :, target_index], window=window
        )
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


# 3) functions required for dot scores calculation

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


def get_insertion_start_pos(insert_bp=59, spacer_bp=199980, num_inserts=2, seq_length = 1310720):
    """
    Returns a list of insertion start positions.
    It is assumed that inserts are inserted centrally around the background sequence's
    midpoint and that spacers between inserts are the same and equal to spacer_bp.

    Parameters
    ------------
    insert_bp : int
        Length of a single insert (bp) [WITH FLANKS!].
    spacer_bp : int
        Length of a spacing sequence between consecutive inserts (bp).
    num_inserts : int
        Desired number of inserts to be inserted.
    seq_length : int
        The length the prediction vector.

    Returns
    ---------
    insertion_starting_positions : list
        The insertion start positions.
    """
    
    insert_plus_spacer_bp = insert_bp + 2 * spacer_bp
    multi_insert_bp = num_inserts * insert_plus_spacer_bp
    insert_start_bp = seq_length // 2 - multi_insert_bp // 2

    insertion_starting_positions = []
    for i in range(num_inserts):
        offset = insert_start_bp + i * insert_plus_spacer_bp + spacer_bp
        insertion_starting_positions.append(offset)
        
    return insertion_starting_positions

def map_sum(map_fragment):
    """
    Returns a sum of values for a 2D matrix.
    """
    return (map_fragment**2).sum(axis=(0,1))


def get_lines(row_line, col_line, dot_band_size):
    """
    Returns a list of columns and rows limiting the fragment of a map
    that calculation of (local) dot score is based on. 

    Parameters
    ------------
    row_line : int
    col_line : int
        The dot is expected to be on the crossing of a matrix row with row_line row index,
        and a matrix column with col_line column index.
    dot_band_size : int
        Specifies how big fragment of a matrix around the expected dot should
        be taken into account in a dot score calculation.

    Returns
    ---------
    insertion_starting_positions : list
        The insertion start positions.
    """
    upper_horizontal = row_line - (dot_band_size//2)
    lower_horizontal = row_line + (dot_band_size//2)
    if (dot_band_size % 2) == 1:
        lower_horizontal += 1
    
    left_vertical = col_line - (dot_band_size//2)
    right_vertical = col_line + (dot_band_size//2)
    if (dot_band_size % 2) == 1:
        right_vertical += 1
        
    return upper_horizontal, lower_horizontal, left_vertical, right_vertical


def calculate_dot_score(map_matrix, row_line, col_line, dot_band_size=3, reference_map_matrix=None):
    """
    Returns a dot score, which is an averge matrix value in a (dot_band_size x dot_band_size) matrix fragment
    around the point where a dot is expected (so, where the row_line crosses the col_line)

    Parameters
    ------------
    map_matrix : numpy array
        Array with contact change maps predicted by Akita, usually of a size (512 x 512 x num_targets)
    row_line : int
    col_line : int
        The dot is expected to be on the crossing of a matrix row with row_line row index,
        and a matrix column with col_line column index.
    dot_band_size : int
        Specifies how big fragment of a matrix around the expected dot should
        be taken into account in a dot score calculation.

    Returns
    ---------
    dot_score : numpy array
        The dot score calculated for each target index.
    """

    upper_horizontal, lower_horizontal, left_vertical, right_vertical = get_lines(row_line, col_line, dot_band_size)
        
    # central, dot part
    dot_score = map_sum(map_matrix[upper_horizontal:lower_horizontal, left_vertical:right_vertical])
    return dot_score


def calculate_dot_x_score(map_matrix, row_line, col_line, dot_band_size=3, boundary_band_size=10, reference_map_matrix=None):
    """
    Returns an x score, which is an averge matrix value in a (dot_band_size x dot_band_size) matrix fragment
    around the point where a dot is expected (so, where the row_line crosses the col_line) subtracted by 
    an average matrix value in an x-shaped neighborhood (boundary_band_size x boundary_band_size; directions: NW, NE, SW, SE) 
    matrix fragment.

    Parameters
    ------------
    map_matrix : numpy array
        Array with contact change maps predicted by Akita, usually of a size (512 x 512 x num_targets)
    row_line : int
    col_line : int
        The dot is expected to be on the crossing of a matrix row with row_line row index,
        and a matrix column with col_line column index.
    dot_band_size : int
        Specifies how big fragment of a matrix around the expected dot should
        be taken into account in a dot matrix (inner) average calculation.
    boundary_band_size : int
        Specifies how big fragment of a matrix outside the expected dot should
        be taken into account in an x-shaped neighborhood area (outer) average calculation.
    Returns
    ---------
    dot_score - x_score : numpy array
        The x score calculated for each target index.
    """
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


def calculate_dot_cross_score(map_matrix, row_line, col_line, dot_band_size=3, boundary_band_size=10, reference_map_matrix=None):
    """
    Returns a cross score, which is an averge matrix value in a (dot_band_size x dot_band_size) matrix fragment
    around the point where a dot is expected (so, where the row_line crosses the col_line) subtracted by 
    an average matrix value in a cross-shaped neighborhood (dot_band_size x boundary_band_size, directions: N, S, E, W) 
    matrix fragment.

    Parameters
    ------------
    map_matrix : numpy array
        Array with contact change maps predicted by Akita, usually of a size (512 x 512 x num_targets)
    row_line : int
    col_line : int
        The dot is expected to be on the crossing of a matrix row with row_line row index,
        and a matrix column with col_line column index.
    dot_band_size : int
        Specifies how big fragment of a matrix around the expected dot should
        be taken into account in a dot matrix (inner) average calculation.
    boundary_band_size : int
        Specifies how big fragment of a matrix outside the expected dot should
        be taken into account in a cross-shaped neighborhood area (outer) average calculation.
    Returns
    ---------
    dot_score - cross_score : numpy array
        The cross score calculated for each target index.
    """
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


# 4) flames score

def calculate_flames_score(map_matrix):
    raise NotImplementedError("Will be added soon")


# calculating all desired scores for a set of maps


def calculate_scores(stat_metrics, map_matrix, reference_map_matrix=None, **kwargs):

    scores = {}

    if "SCD" in stat_metrics:
        SCDs = calculate_SCD(map_matrix, reference_map_matrix)
        scores["SCD"] = SCDs

    if np.any((["INS" in stat.split("-")[0] for stat in stat_metrics])):
        for stat in stat_metrics:
            if stat.split("-")[0] == "INS":
                window = int(stat.split("-")[1])
                INS = calculate_INS(map_matrix, window)
                refINS = calculate_INS(reference_map_matrix, window)
                scores[f"alt_{stat}"] = INS
                scores[f"ref_{stat}"] = refINS

    if ("dot-score" in stat_metrics) or ("cross-score" in stat_metrics) or ("x-score" in stat_metrics):
        starting_positions = get_insertion_start_pos()
        row_line, col_line = get_bin(starting_positions[0], starting_positions[0] + INSERT_LEN), get_bin(starting_positions[1], starting_positions[1] + INSERT_LEN)

        if "dot-score" in stat_metrics:
            dot_score = calculate_dot_score(map_matrix, row_line, col_line, reference_map_matrix=reference_map_matrix)
            scores["dot-score"] = dot_score

        if "x-score" in stat_metrics:
            x_score = calculate_dot_x_score(map_matrix, row_line, col_line, reference_map_matrix=reference_map_matrix)
            scores["x-score"] = x_score
            
        if "cross-score" in stat_metrics:
            cross_score = calculate_dot_cross_score(map_matrix, row_line, col_line, reference_map_matrix=reference_map_matrix)
            scores["cross-score"] = cross_score
            
    # new scores will be added soon...

    return scores
