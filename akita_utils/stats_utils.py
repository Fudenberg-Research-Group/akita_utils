import numpy as np
from .numpy_utils import ut_dense

# STANDARDS
MOTIF_LEN = 19
FLANK_LEN = 20
INSERT_LEN = 2 * FLANK_LEN + MOTIF_LEN


# MATRIX TRANSFORMATION

def get_reference_map_matrix(
    hf, head_index, model_index, num_background, diagonal_offset=2
):
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

    map_size = len(
        ut_dense(
            hf[f"refmap_h{head_index}_m{model_index}"][0, :, :],
            diagonal_offset,
        )
    )
    ref_map_matrix = np.zeros(
        (num_background, map_size, map_size, num_targets)
    )

    for background_index in range(num_background):
        preds_matrix = hf[f"refmap_h{head_index}_m{model_index}"][
            background_index, :, :
        ]
        map_matrix = ut_dense(preds_matrix, diagonal_offset)

        ref_map_matrix[background_index, :, :, :] += map_matrix

    return ref_map_matrix


def get_map_matrix(
    hf, head_index, model_index, num_experiments, diagonal_offset=2
):
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

    map_size = len(
        ut_dense(
            hf[f"map_h{head_index}_m{model_index}"][0, :, :], diagonal_offset
        )
    )

    exp_map_matrix = np.zeros(
        (num_experiments, map_size, map_size, num_targets)
    )

    for exp_index in range(num_experiments):
        preds_matrix = hf[f"map_h{head_index}_m{model_index}"][exp_index, :, :]
        map_matrix = ut_dense(preds_matrix, diagonal_offset)

        exp_map_matrix[exp_index, :, :, :] += map_matrix

    return exp_map_matrix


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
    num_targets-long vector with SCD score cf"map{exp_index}alculated for each target.
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


def calculate_SSD(map_matrix, reference_map_matrix=None):
    """
    Calculates SSD score for a multiple-target prediction matrix.
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
    num_targets-long vector with SSD score calculated for each target.
    """

    if type(reference_map_matrix) != np.ndarray:
        map_matrix = map_matrix.astype("float32")
        return map_matrix.sum(axis=(0, 1))
    else:
        map_matrix = map_matrix.astype("float32")
        reference_map_matrix = reference_map_matrix.astype("float32")
        return map_matrix.sum(axis=(0, 1)) - reference_map_matrix.sum(
            axis=(0, 1)
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
    size_after_cropping = map_size * bin_size
    size_difference = input_size - size_after_cropping
    one_side_cropped_length = size_difference // 2

    corrected_window_start = window_start - one_side_cropped_length
    corrected_window_end = window_end - one_side_cropped_length

    first_bin_covered = corrected_window_start // bin_size
    last_bin_covered = corrected_window_end // bin_size

    assert first_bin_covered == last_bin_covered

    return first_bin_covered


def get_insertion_start_pos(
    insert_bp=59, spacer_bp=199980, num_inserts=2, seq_length=1310720
):
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
    return (map_fragment**2).sum(axis=(0, 1))


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
    upper_horizontal = row_line - (dot_band_size // 2)
    lower_horizontal = row_line + (dot_band_size // 2)
    if (dot_band_size % 2) == 1:
        lower_horizontal += 1

    left_vertical = col_line - (dot_band_size // 2)
    right_vertical = col_line + (dot_band_size // 2)
    if (dot_band_size % 2) == 1:
        right_vertical += 1

    return upper_horizontal, lower_horizontal, left_vertical, right_vertical


def calculate_dot_score(
    map_matrix, row_line, col_line, dot_band_size=3, reference_map_matrix=None
):
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

    (
        upper_horizontal,
        lower_horizontal,
        left_vertical,
        right_vertical,
    ) = get_lines(row_line, col_line, dot_band_size)

    # central, dot part
    dot_score = map_sum(
        map_matrix[
            upper_horizontal:lower_horizontal, left_vertical:right_vertical
        ]
    )
    return dot_score


def calculate_dot_x_score(
    map_matrix,
    row_line,
    col_line,
    dot_band_size=3,
    boundary_band_size=10,
    reference_map_matrix=None,
):
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
    (
        upper_horizontal,
        lower_horizontal,
        left_vertical,
        right_vertical,
    ) = get_lines(row_line, col_line, dot_band_size)

    # central, dot part
    dot_size = dot_band_size**2
    dot_score = (
        map_sum(
            map_matrix[
                upper_horizontal:lower_horizontal, left_vertical:right_vertical
            ]
        )
        / dot_size
    )

    # x-parts
    x_score = 0
    x_size = (boundary_band_size**2) * 4
    for matrix_part in [
        map_matrix[
            upper_horizontal - boundary_band_size : upper_horizontal,
            left_vertical - boundary_band_size : left_vertical,
        ],
        map_matrix[
            upper_horizontal - boundary_band_size : upper_horizontal,
            right_vertical : right_vertical + boundary_band_size,
        ],
        map_matrix[
            lower_horizontal : lower_horizontal + boundary_band_size,
            left_vertical - boundary_band_size : left_vertical,
        ],
        map_matrix[
            lower_horizontal : lower_horizontal + boundary_band_size,
            right_vertical : right_vertical + boundary_band_size,
        ],
    ]:
        x_score += map_sum(matrix_part)

    x_score = x_score / x_size

    return dot_score - x_score


def calculate_dot_cross_score(
    map_matrix,
    row_line,
    col_line,
    dot_band_size=3,
    boundary_band_size=10,
    reference_map_matrix=None,
):
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
    (
        upper_horizontal,
        lower_horizontal,
        left_vertical,
        right_vertical,
    ) = get_lines(row_line, col_line, dot_band_size)

    # central, dot part
    dot_size = dot_band_size**2
    dot_score = (
        map_sum(
            map_matrix[
                upper_horizontal:lower_horizontal, left_vertical:right_vertical
            ]
        )
        / dot_size
    )

    # cross-parts
    cross_score = 0
    cross_size = dot_band_size * boundary_band_size * 4
    for matrix_part in [
        map_matrix[
            upper_horizontal - boundary_band_size : upper_horizontal,
            left_vertical:right_vertical,
        ],
        map_matrix[
            upper_horizontal:lower_horizontal,
            left_vertical - boundary_band_size : left_vertical,
        ],
        map_matrix[
            upper_horizontal:lower_horizontal,
            right_vertical : right_vertical + boundary_band_size,
        ],
        map_matrix[
            lower_horizontal : lower_horizontal + boundary_band_size,
            left_vertical:right_vertical,
        ],
    ]:
        cross_score += map_sum(matrix_part)

    cross_score = cross_score / cross_size

    return dot_score - cross_score


# 4) Sliding INS Score


def slide_diagonal_insulation(target_map, window=10):
    """
    Calculate insulation by sliding a window along the diagonal of the target map.

    Parameters
    ------------
    target_map : numpy array
        Array with contact change maps predicted by Akita, usually of a size (512 x 512)
    window : int
        Size of the sliding window.

    Returns
    ---------
    scores : list
        List of ISN-window scores for each position along the diagonal.
    """

    map_size = target_map.shape[0]
    scores = np.empty((map_size,))
    scores[:] = np.nan

    for mid in range(window, (map_size - window)):
        lo = max(0, mid + 1 - window)
        hi = min(map_size, mid + window)
        score = np.nanmean(target_map[lo : (mid + 1), mid:hi])
        scores[mid] = score

    return scores


def extract_window_from_vector(vector, window=10, width=3):
    """
    Extract a window of size 3*window centered around the center of the vector.

    Parameters
    ------------
    vector : list or numpy array
        Input vector.
    window : int, optional
        Size of the window. Default is 16.

    Returns
    ---------
    window_vector : list or numpy array
        Extracted window of size 3*window.
    """
    center_index = len(vector) // 2
    start_index = max(center_index - width * window, 0)
    end_index = min(center_index + width * window, len(vector))
    window_vector = vector[start_index:end_index]
    return window_vector


def min_insulation_offset_from_center(
    target_map, window=10, crop_around_center=True, crop_width=3
):
    """
    Calculate the offset from the center position for the position along the diagonal
    with the minimum insulation score for a sliding window along the diagonal.

    Parameters
    ------------
    target_map : numpy array
        Array with contact change maps predicted by Akita, usually of a size (512 x 512).
    window : int, optional
        Size of the sliding window for insulation calculation. Default is 10.

    Returns
    ---------
    offset_from_center : int
        Offset from the center position based on the position with the minimum insulation score.
    """
    map_size = target_map.shape[0]
    center_position = map_size // 2
    bin_shift = 0

    insulation_scores = slide_diagonal_insulation(target_map, window)

    if crop_around_center:
        bin_shift = max(center_position - crop_width * window, 0)
        insulation_scores = extract_window_from_vector(
            insulation_scores, window=window, width=3
        )

    min_score = np.nanmin(insulation_scores)

    # Find indices with min_score
    indices_tuple = np.where(insulation_scores == min_score)
    indices_list = list(indices_tuple[0])

    # in case there are more than one index with min_score
    # Calculate midpoint of the array
    midpoint = len(insulation_scores) // 2

    # Find index closest to the midpoint among indices with the same score
    closest_index = None
    min_difference = float("inf")
    for index in indices_list:
        difference = abs(index - midpoint)
        if difference < min_difference:
            closest_index = index
            min_difference = difference

    closest_index = closest_index + bin_shift
    offset_from_center = closest_index - center_position
    return offset_from_center


def calculate_offset_INS(
    map_matrix, window=10, crop_around_center=True, crop_width=3
):
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
    scores : num_targets-long vector with ISN-window offset
    """

    num_targets = map_matrix.shape[-1]
    scores = np.zeros((num_targets,))
    for target_index in range(num_targets):
        offset_from_center = min_insulation_offset_from_center(
            map_matrix[:, :, target_index],
            window=window,
            crop_around_center=crop_around_center,
            crop_width=crop_width,
        )
        scores[target_index] = offset_from_center
    return scores


# calculating all desired scores for a set of maps


def calculate_scores(
    stat_metrics, map_matrix, reference_map_matrix=None, **kwargs
):
    scores = {}

    if "SCD" in stat_metrics:
        SCDs = calculate_SCD(map_matrix, reference_map_matrix)
        scores["SCD"] = SCDs

    if "SSD" in stat_metrics:
        SSDs = calculate_SSD(map_matrix, reference_map_matrix)
        scores["SSD"] = SSDs

    if np.any((["INS" in stat.split("-")[0] for stat in stat_metrics])):
        for stat in stat_metrics:
            if stat.split("-")[0] == "INS":
                window = int(stat.split("-")[1])
                INS = calculate_INS(map_matrix, window)
                refINS = calculate_INS(reference_map_matrix, window)
                scores[f"alt_{stat}"] = INS
                scores[f"ref_{stat}"] = refINS

    if np.any((["OFF" in stat.split("-")[0] for stat in stat_metrics])):
        for stat in stat_metrics:
            if stat.split("-")[0] == "OFF":
                window = int(stat.split("-")[1])
                OFF = calculate_offset_INS(
                    map_matrix, window, crop_around_center=True, crop_width=3
                )
                scores[f"{stat}"] = OFF

    if (
        ("dot-score" in stat_metrics)
        or ("cross-score" in stat_metrics)
        or ("x-score" in stat_metrics)
    ):
        starting_positions = get_insertion_start_pos()
        row_line, col_line = get_bin(
            starting_positions[0], starting_positions[0] + INSERT_LEN
        ), get_bin(starting_positions[1], starting_positions[1] + INSERT_LEN)

        if "dot-score" in stat_metrics:
            dot_score = calculate_dot_score(
                map_matrix,
                row_line,
                col_line,
                reference_map_matrix=reference_map_matrix,
            )
            scores["dot-score"] = dot_score

        if "x-score" in stat_metrics:
            x_score = calculate_dot_x_score(
                map_matrix,
                row_line,
                col_line,
                reference_map_matrix=reference_map_matrix,
            )
            scores["x-score"] = x_score

        if "cross-score" in stat_metrics:
            cross_score = calculate_dot_cross_score(
                map_matrix,
                row_line,
                col_line,
                reference_map_matrix=reference_map_matrix,
            )
            scores["cross-score"] = cross_score

    # new scores will be added soon...

    return scores


# CALCULATING SCORES (DF) BASED ON KEYWORDS


def calculate_INS_keywords(df, keywords, drop=True):
    """
    Calculates INS values based on specified keywords in the DataFrame.

    This function takes a DataFrame, a list of keywords, and an optional drop parameter as input.
    If any of the keywords contain "INS", the function calculates the insertion (INS) values
    by subtracting the "ref_INS" column from the "alt_INS" column for each specified window size.
    If drop is True, the function drops the "alt_INS" and "ref_INS" columns after calculation.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing necessary columns for calculations.
    keywords : list
        A list of keywords to identify INS window sizes in the DataFrame columns.
    drop : bool, optional
        If True, drops "alt_INS" and "ref_INS" columns after calculation. Default is True.

    Returns
    -------
    pd.DataFrame
        If any of the keywords contain "INS", a DataFrame with additional columns
        containing calculated INS values is returned. Otherwise, the function
        returns unchanged input DataFrame.

    Raises:
    Exception: If "alt_INS" or "ref_INS" columns cannot be found for any specified keyword.
    """
    if any("INS" in keyword for keyword in keywords):
        df_out = df.copy(deep=True)
        windows = []

        for keyword in keywords:
            if "INS" in keyword:
                window = int(keyword.split("-")[1])
                if window not in windows:
                    windows.append(window)

        for window in windows:
            key = f"INS-{window}"
            if "ref_" + key in df.columns and "alt_" + key in df.columns:
                df_out[key] = df_out["alt_" + key] - df_out["ref_" + key]
                if drop:
                    df_out = df_out.drop(columns=["alt_" + key, "ref_" + key])
            else:
                raise Exception(
                    f"alt_ and ref_ columns cannot be found for the following keyword: {keyword}"
                )

        return df_out

    else:
        return df


def calculate_INS_by_targets_keywords(
    df, keywords, max_nr_targets=6, max_nr_heads=2, max_nr_models=8, drop=True
):
    """
    Calculates INS values based on specified keywords in the DataFrame considering targets, heads, and models.

    This function takes a DataFrame, a list of keywords, and optional parameters for the maximum number of
    targets, heads, and models. It calculates the insertion (INS) values for each target index, head index,
    and model index specified in the DataFrame columns. The function assumes a standard naming convention
    for the columns where INS values are stored, including head index, model index, and target index
    [{score}_h{head_index}_m{model_index}_t{target_index}].

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing necessary columns for calculations.
    keywords : list
        A list of keywords to identify INS window sizes in the DataFrame columns.
    max_nr_targets : int, optional
        The maximum number of target indices to consider. Default is 6.
    max_nr_heads : int, optional
        The maximum number of head indices to consider. Default is 2.
    max_nr_models : int, optional
        The maximum number of model indices to consider. Default is 8.
    drop : bool, optional
        If True, drops "alt_INS" and "ref_INS" columns after calculation. Default is True.

    Returns
    -------
    pd.DataFrame
        If any of the keywords contain "INS", DataFrame with additional columns containing calculated INS values
        for each target, head, and model index is returned. Otherwise, the functionreturns unchanged input DataFrame.
    """

    if any("INS" in keyword for keyword in keywords):
        df_out = df.copy(deep=True)
        windows = []

        for keyword in keywords:
            if "INS" in keyword:
                window = int(keyword.split("-")[1])
                if window not in windows:
                    windows.append(window)

        for window in windows:
            for head_index in range(max_nr_heads):
                for model_index in range(max_nr_models):
                    for target_index in range(max_nr_targets):
                        key = f"INS-{window}_h{head_index}_m{model_index}_t{target_index}"

                        if (
                            "ref_" + key in df.columns
                            and "alt_" + key in df.columns
                        ):
                            df_out[key] = (
                                df_out["alt_" + key] - df_out["ref_" + key]
                            )
                            if drop:
                                df_out = df_out.drop(
                                    columns=["alt_" + key, "ref_" + key]
                                )

        return df_out

    else:
        return df
