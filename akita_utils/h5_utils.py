import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import pandas as pd
import os
from akita_utils.utils import ut_dense
from akita_utils.stats_utils import calculate_scores


# METADATA FUNCTIONS


def prepare_metadata_dir(model_file, genome_fasta, seqnn_model):
    """
    Creates a metadata dictionary, optional in h5 file initializing.

    Parameters
    ------------
    model_file : str
        Path to the model file.
    genome_fasta : str
        Path to the genome file (mouse or human).
    seqnn_model : object
        Loaded model.

    Returns
    ---------
    metadata_dict : dir
        A dictionary with additional metadata.
    """

    head_index = int(model_file.split("model")[-1][0])
    model_index = int(model_file.split("c0")[0][-1])

    metadata_dict = {
        "model_index": model_index,
        "head_index": head_index,
        "genome": genome_fasta.split("/")[-1],
        "seq_length": seqnn_model.seq_length,
        "diagonal_offset": seqnn_model.diagonal_offset,
        "prediction_vector_length": seqnn_model.target_lengths[0],
        "target_crops": seqnn_model.target_crops,
        "num_targets": seqnn_model.num_targets(),
    }

    return metadata_dict


# H5 INITIALIZATION FUNCTIONS


def initialize_stat_output_h5(
    out_dir,
    model_file,
    stat_metrics,
    seq_coords_df,
    add_metadata=False,
    **kwargs,
):
    """
    Initializes an h5 file to save statistical metrics calculated from Akita's predictions.

    Parameters
    ------------
    out_dir : str
        Path to the desired location of the output h5 file.
    model_file : str
        Path to the model file.
    stat_metrics : list
        List of stratistical metrics that are supposed to be calculated.
    seq_coords_df : dataFrame
        Pandas dataframe where each row represents one experiment (so one set of prediction).
    add_metadata : Boolean
        True if metadata is supposed to be added.
        If so, two additional arguments have to be provided:
            genome_fasta : str
                Path to the genome file (mouse or human).
            seqnn_model : object
                Loaded model.

    Returns
    ---------
    h5_outfile : h5py object
        An initialized h5 file.
    """

    h5_outfile = h5py.File("%s/STATS_OUT.h5" % out_dir, "w")
    seq_coords_df_dtypes = seq_coords_df.dtypes

    head_index = int(model_file.split("model")[-1][0])
    model_index = int(model_file.split("c0")[0][-1])

    num_targets = 6
    if head_index == 0:
        num_targets = 5

    num_experiments = len(seq_coords_df)

    if add_metadata:
        h5_outfile.attrs["date"] = str(date.today())

        metadata_dict = prepare_metadata_dir(
            model_file, genome_fasta, seqnn_model
        )
        h5_outfile.attrs.update(metadata_dict)

    for key in seq_coords_df:
        if seq_coords_df_dtypes[key] is np.dtype("O"):
            # Find the maximum string length in the column
            max_length = seq_coords_df[key].str.len().max()
            dtype_str = f"S{max_length}"

            # Create the dataset with the determined dtype
            h5_outfile.create_dataset(
                key, data=seq_coords_df[key].values.astype(dtype_str)
            )

        else:
            h5_outfile.create_dataset(key, data=seq_coords_df[key])

    # initialize keys for statistical metrics collection
    for stat_metric in stat_metrics:
        if stat_metric in seq_coords_df.keys():
            raise KeyError("check input tsv for clashing score name")

        if "INS" not in stat_metric:
            h5_outfile.create_dataset(
                f"{stat_metric}_h{head_index}_m{model_index}",
                shape=(num_experiments, num_targets),
                dtype="float16",
                compression=None,
            )
        else:
            h5_outfile.create_dataset(
                "ref_" + f"{stat_metric}_h{head_index}_m{model_index}",
                shape=(num_experiments, num_targets),
                dtype="float16",
                compression=None,
            )
            h5_outfile.create_dataset(
                "alt_" + f"{stat_metric}_h{head_index}_m{model_index}",
                shape=(num_experiments, num_targets),
                dtype="float16",
                compression=None,
            )

    return h5_outfile


def initialize_maps_output_h5(out_dir, model_file, seqnn_model, seq_coords_df):
    """
    Initializes an h5 file to save vectors predicted by Akita.

    Parameters
    ------------
    out_dir : str
        Path to the desired location of the output h5 file.
    model_file : str
        Path to the model file.
    seqnn_model : object
        Loaded model.
    seq_coords_df : dataFrame
        Pandas dataframe where each row represents one experiment (so one set of prediction).

    Returns
    ---------
    h5_outfile : h5py object
        An initialized h5 file.
    """
    h5_outfile = h5py.File("%s/MAPS_OUT.h5" % out_dir, "w")
    seq_coords_df_dtypes = seq_coords_df.dtypes

    head_index = int(model_file.split("model")[-1][0])
    model_index = int(model_file.split("c0")[0][-1])
    prediction_vector_length = seqnn_model.target_lengths[0]

    num_targets = seqnn_model.num_targets()

    num_experiments = len(seq_coords_df)

    for key in seq_coords_df:
        if seq_coords_df_dtypes[key] is np.dtype("O"):
            h5_outfile.create_dataset(
                key, data=seq_coords_df[key].values.astype("S")
            )
        else:
            h5_outfile.create_dataset(key, data=seq_coords_df[key])

    h5_outfile.create_dataset(
        f"map_h{head_index}_m{model_index}",
        shape=(num_experiments, prediction_vector_length, num_targets),
        dtype="float16",
    )

    return h5_outfile


def initialize_maps_output_references(
    out_dir, model_file, seqnn_model, num_backgrounds=10
):
    """
    Initializes an h5 file to save reference vectors predicted by Akita.

    Parameters
    ------------
    out_dir : str
        Path to the desired location of the output h5 file.
    model_file : str
        Path to the model file.
    seqnn_model : object
        Loaded model.
    num_backgrounds : int
        Number of reference maps to be saved.

    Returns
    ---------
    h5_outfile : h5py object
        An initialized h5 file.
    """
    head_index = int(model_file.split("model")[-1][0])
    model_index = int(model_file.split("c0")[0][-1])
    prediction_vector_length = seqnn_model.target_lengths[0]

    h5_outfile = h5py.File(f"%s/REFMAPS_OUT_m{model_index}.h5" % out_dir, "w")

    num_targets = seqnn_model.num_targets()

    h5_outfile.create_dataset(
        f"refmap_h{head_index}_m{model_index}",
        shape=(num_backgrounds, prediction_vector_length, num_targets),
        dtype="float16",
    )

    return h5_outfile


# WRITING TO H5 FUNCTIONS


def write_stat_metrics_to_h5(
    prediction_matrix,
    reference_prediction_matrix,
    h5_outfile,
    experiment_index,
    head_index,
    model_index,
    diagonal_offset=2,
    stat_metrics=["SCD"],
):
    """
    Writes to an h5 file saving statistical metrics calculated from Akita's predicftions.

    Parameters
    ------------
    prediction_matrix : numpy matrix
        Matrix collecting Akita's predictions.
    reference_prediction_matrix : numpy matrix
        Matrix collecting Akita's reference predictions.
    h5_outfile : h5py object
        An initialized h5 file.
    experiment_index : int
        Index identifying one experiment.
    head_index : int
        Head index used to get a prediction (Mouse: head_index=1; Human: head_index=0).
    model_index : int
        Index of one of 8 models that has been used to make predictions (an index between 0 and 7).
    diagonal_offset : int
        Number of diagonals that are added as zeros in the conversion.
        Typically 2 diagonals are ignored in Hi-C data processing.
    stat_metrics : list
        List of stratistical metrics that are supposed to be calculated.

    Returns
    ---------
    h5_outfile : h5py object
        An overwritten h5 file.
    """

    # increase dtype
    prediction_matrix = prediction_matrix.astype("float32")
    if type(reference_prediction_matrix) == np.ndarray:
        reference_prediction_matrix = reference_prediction_matrix.astype(
            "float32"
        )

    # convert prediction vectors to maps
    map_matrix = ut_dense(prediction_matrix, diagonal_offset)
    if type(reference_prediction_matrix) == np.ndarray:
        ref_map_matrix = ut_dense(reference_prediction_matrix, diagonal_offset)

    # getting desired scores
    if type(reference_prediction_matrix) == np.ndarray:
        scores = calculate_scores(stat_metrics, map_matrix, ref_map_matrix)
    else:
        scores = calculate_scores(stat_metrics, map_matrix)

    for key in scores:
        h5_outfile[f"{key}_h{head_index}_m{model_index}"][
            experiment_index
        ] = scores[key].astype("float16")


def write_maps_to_h5(
    vector_matrix,
    h5_outfile,
    experiment_index,
    head_index,
    model_index,
    reference=False,
):
    """
    Writes entire maps to an h5 file.

    Parameters
    ------------
    vector_matrix : numpy matrix
        Matrix collecting Akita's prediction maps. Shape: (map_size, map_size, num_targets).
    h5_outfile : h5py object
        An initialized h5 file.
    experiment_index : int
        Index identifying one experiment.
    head_index : int
        Head index used to get a prediction (Mouse: head_index=1; Human: head_index=0).
    model_index : int
        Index of one of 8 models that has been used to make predictions (an index between 0 and 7).
    reference : Boolean
        Assigned to True when the reference predictions are saved.
    """

    prefix = "map"
    if reference:
        prefix = "refmap"

    for target_index in range(vector_matrix.shape[-1]):
        h5_outfile[f"{prefix}_h{head_index}_m{model_index}"][
            experiment_index, :, target_index
        ] += vector_matrix[:, target_index]


def save_maps(
    prediction_matrix,
    experiment_index,
    head_index,
    model_index,
    diagonal_offset=2,
    plot_dir=None,
    plot_lim_min=0.1,
    plot_freq=100,
):
    """
    Writes to an h5 file saving statistical metrics calculated from Akita's predicftions.

    Parameters
    ------------
    prediction_matrix : numpy matrix
        Matrix collecting Akita's predictions.
    experiment_index : int
        Index identifying one experiment.
    head_index : int
        Head index used to get a prediction (Mouse: head_index=1; Human: head_index=0).
    model_index : int
        Index of one of 8 models that has been used to make predictions (an index between 0 and 7).
    diagonal_offset : int
        Number of diagonals that are added as zeros in the conversion.
        Typically 2 diagonals are ignored in Hi-C data processing.
    plot_dir : str
        Path to the desired location of the output plots (plots will not be saved if plot_dir == None).
    plot_lim_min : float
        Negative minimum and positive maximum values that will be used to plot maps.
    plot_freq : int
        A plot of one out of plot_freq number of predictions is saved.
    """

    if (plot_dir is not None) and (np.mod(experiment_index, plot_freq) == 0):
        print("plotting prediction: ", experiment_index)

        # increase dtype
        prediction_matrix = prediction_matrix.astype("float32")

        # convert prediction vectors to maps
        map_matrix = ut_dense(prediction_matrix, diagonal_offset)

        _, axs = plt.subplots(1, prediction_matrix.shape[-1], figsize=(24, 4))

        for target_index in range(prediction_matrix.shape[-1]):
            map_target = map_matrix[..., target_index]
            vmin = min(map_target.min(), map_target.min())
            vmax = max(map_target.max(), map_target.max())
            vmin = min(-plot_lim_min, vmin)
            vmax = max(plot_lim_min, vmax)
            sns.heatmap(
                map_target,
                ax=axs[target_index],
                center=0,
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu_r",
                xticklabels=False,
                yticklabels=False,
            )

        plt.tight_layout()

        plt.savefig(
            f"e{experiment_index}_h{head_index}_m{model_index}_t{target_index}.pdf"
        )
        plt.close()


# AVERAGING FUNCTIONS


def average_over_keys(h5_file, df, keywords):
    """
    Averages data over keys containing specific keywords in the provided HDF5 file.

    This function takes an HDF5 file, a DataFrame, and a list of keywords as input.
    It searches for keys in the HDF5 file containing each specified keyword,
    extracts the data from those keys, averages the data over targets and/or models,
    and adds a new column to the input DataFrame with the averaged values.

    Parameters
    ----------
    h5_file : h5py.File)
        The HDF5 file object containing the data.
    df : pd.DataFrame)
        The DataFrame to which the averaged values will be added.
    keywords : list):
        A list of keywords to search for in the HDF5 keys.

    Returns
    -------
    df : pd.DataFrame
        The input DataFrame with additional columns containing averaged values
                  corresponding to the specified keywords.

    Raises
    ------
    Exception: If no matching keys are found for any of the specified keywords.
    """

    for keyword in keywords:
        # collecting all keys with keyword in the name
        keys = [
            key
            for key in h5_file.keys()
            if keyword in key and key not in keyword
        ]
        if not keys:
            raise Exception(
                f"There are no matching keys for the following keyword: {keyword}"
            )

        data = pd.DataFrame()
        for key in keys:
            nr_targets = h5_file[key][()].shape[1]
            for target_index in range(nr_targets):
                series = pd.Series(
                    h5_file[key][:, target_index],
                    name=key + f"_t{target_index}",
                )
                data = pd.concat([data, series], axis=1)

        # averaging over targets and / or models
        average = data.mean(axis=1)
        df = pd.concat([df, pd.Series(average, name=keyword)], axis=1)

    return df


def collect_all_keys_with_keywords(h5_file, df, keywords, ignore_keys=[]):
    """
    Collects data from keys containing specific keywords in the provided HDF5 file.

    This function takes an HDF5 file, a DataFrame, and a list of keywords as input.
    It searches for keys in the HDF5 file containing each specified keyword,
    extracts the data from those keys, and aggregates the data into a DataFrame.

    Parameters
    ----------
    h5_file : h5py.File
        The HDF5 file object containing the data.
    df : pd.DataFrame
        The initial DataFrame to which the collected data will be added.
    keywords : list
        A list of keywords to search for in the HDF5 keys.

    Returns
    -------
    data : pd.DataFrame
        A DataFrame containing data from keys with the specified keywords.

    Raises
    ------
    Exception: If no matching keys are found for any of the specified keywords.
    """

    data = pd.DataFrame()

    for keyword in keywords:
        # collecting all keys with keyword in the name
        keys = [
            key
            for key in h5_file.keys()
            if (
                keyword in key
                and key not in keyword
                and key not in ignore_keys
            )
        ]

        if not keys:
            raise Exception(
                f"There are no matching keys for the following keyword: {keyword}"
            )

        for key in keys:
            nr_targets = h5_file[key][()].shape[1]
            for target_index in range(nr_targets):
                series = pd.Series(
                    h5_file[key][:, target_index],
                    name=key + f"_t{target_index}",
                )
                data = pd.concat([data, series], axis=1)

    return data


# COLLECTING DATA FROM MULTIPLE HDF5 FILES & CHECKING START OF JOBS


def job_started(out_dir, job_index, h5_file_name="STATS_OUT.h5"):
    """
    Check if a specific job has started and generated its output file.

    This function verifies if the output file for a particular job exists. It constructs
    the path to the job's output file using the specified directory, job index, and file name.
    Returns True if the file exists, indicating the job's start, or False otherwise.

    Parameters
    ------------
    out_dir : str
        Path to the directory containing job-specific output files.
    job_index : int
        Index of the job to be checked for completion.
    h5_file_name : str, optional
        Name of the output HDF5 file. Default is "STATS_OUT.h5".

    Returns:
    bool: True if the job's output file exists, indicating job completion. False otherwise.
    """
    out_file = "%s/job%d/%s" % (out_dir, job_index, h5_file_name)
    return os.path.isfile(out_file) or os.path.isdir(out_file)


def infer_num_jobs(out_dir):
    """
    Infer the number of jobs from a directory containing job-related files.

    This function looks for directories named "jobX" within the specified directory
    and determines the highest job index present, then returns the number of jobs
    (highest index + 1).

    Parameters
    ------------
    out_dir : str
        Path to the directory containing job-related files.

    Returns
    ---------
    num_job : int
        Number of jobs found in the specified directory.
    """
    highest_index = 0

    for r, d, folder in os.walk(out_dir):
        for folder_name in d:
            if folder_name[:3] == "job":
                index = int(folder_name.split("job")[1].split(".")[0])
                if index > highest_index:
                    highest_index = index

    num_job = highest_index + 1
    return num_job


def collect_h5(
    out_dir, seq_coords_df, h5_file_name="STATS_OUT.h5", virtual_exp=True
):
    """
    Aggregates data from multiple job-specific HDF5 files into a single consolidated HDF5 file.

    This function processes a series of job-specific HDF5 files located within subdirectories
    named "jobX" (where X is the job index) under the specified output directory. It extracts
    and combines statistical data and sequences coordinates from these files into one comprehensive
    HDF5 file. The function handles different dimensions of data, including 1D (e.g., single metrics),
    2D (e.g., statistical matrices), and 3D (e.g., prediction vectors), and organizes them accordingly
    in the final HDF5 file. It dynamically sets the data type for string datasets based on the maximum
    string length found in the `seq_coords_df` DataFrame to optimize storage.

    Parameters:
    -----------
    out_dir : str
        The path to the directory containing the job-specific HDF5 files.
    seq_coords_df : DataFrame
        A pandas DataFrame containing the initial sequence coordinates and possibly other related
        data before distribution among the job-specific files. It is used to initialize datasets in
        the final HDF5 file, especially for string-type data where the maximum length is determined dynamically.
    h5_file_name : str, optional
        The name for the output HDF5 file where the aggregated data will be stored. Defaults to "STATS_OUT.h5".
    virtual_exp : bool, optional
        A flag indicating whether the experiment is virtual. This affects how certain data are structured,
        particularly in how reference maps are handled. Defaults to True, meaning each background sequence
        might have a unique reference, as opposed to being compared against every genomic sequence.

    Returns:
    --------
    None
        The function does not return any value. It creates or overwrites an HDF5 file at the specified
        location with the combined data from the individual job-specific files.

    Raises:
    -------
    Exception
        If an unexpected 3-dimensional key is encountered, indicating an inconsistency in the expected
        data structure.

    """
    num_jobs = infer_num_jobs(out_dir)

    num_experiments = len(seq_coords_df)
    if virtual_exp is True:
        num_backgrounds = len(seq_coords_df["background_index"].unique())

    seq_coords_df_dtypes = seq_coords_df.dtypes

    # initialize final final h5 based on the 0th-job file
    final_h5_file = "%s/%s" % (out_dir, h5_file_name)
    final_h5_open = h5py.File(final_h5_file, "w")

    # initializing
    for key in seq_coords_df:
        if seq_coords_df_dtypes[key] is np.dtype("O"):
            # Find the maximum string length in the column
            max_length = seq_coords_df[key].str.len().max()
            dtype_str = f"S{max_length}"

            # Create the dataset with the determined dtype
            final_h5_open.create_dataset(
                key, data=seq_coords_df[key].values.astype(dtype_str)
            )

        else:
            final_h5_open.create_dataset(key, data=seq_coords_df[key])

    print("Getting keys from job 0")
    job0_h5_file = "%s/job0/%s" % (out_dir, h5_file_name)
    job0_h5_open = h5py.File(job0_h5_file, "r")

    for key in job0_h5_open.keys():
        if (job0_h5_open[key].ndim == 1) and (key not in final_h5_open):
            final_h5_open.create_dataset(
                key, shape=(num_experiments,), dtype=job0_h5_open[key].dtype
            )

        elif job0_h5_open[key].ndim == 2:
            # keys with saved stat metrics
            _, num_targets = job0_h5_open[key].shape

            final_h5_open.create_dataset(
                key,
                shape=(
                    num_experiments,
                    num_targets,
                ),
                dtype=job0_h5_open[key].dtype,
            )

        elif job0_h5_open[key].ndim == 3:
            # keys with saved prediction vectors

            if key.split("_")[0] == "map":
                _, prediction_vector_length, num_targets = job0_h5_open[
                    key
                ].shape

                final_h5_open.create_dataset(
                    key,
                    shape=(
                        num_experiments,
                        prediction_vector_length,
                        num_targets,
                    ),
                    dtype=job0_h5_open[key].dtype,
                )
            elif key.split("_")[0] == "refmap":
                if virtual_exp:
                    # then there is one reference for each background sequence
                    final_h5_open.create_dataset(
                        key,
                        shape=(
                            num_backgrounds,
                            prediction_vector_length,
                            num_targets,
                        ),
                        dtype=job0_h5_open[key].dtype,
                    )
                else:
                    # otherwise, there is a separate reference for each genomic prediction
                    # (e.g. permuted sequence is compared with the original genomic sequence)
                    final_h5_open.create_dataset(
                        key,
                        shape=(
                            num_experiments,
                            prediction_vector_length,
                            num_targets,
                        ),
                        dtype=job0_h5_open[key].dtype,
                    )

            else:
                raise Exception(f"Unexpected 3-dimensional key: {key}")

    job0_h5_open.close()

    # set values of the final h5 file
    experiment_index = 0
    for job_index in range(num_jobs):
        print("Collecting job number:", job_index)

        # open the h5 file
        job_h5_file = "%s/job%d/%s" % (out_dir, job_index, h5_file_name)
        job_h5_open = h5py.File(job_h5_file, "r")

        # append to the final h5 file
        for key in job_h5_open.keys():
            job_experiments_num = job_h5_open[key].shape[0]

            if job_h5_open[key].ndim == 1:
                final_h5_open[key][
                    experiment_index : experiment_index + job_experiments_num
                ] = job_h5_open[key]

            elif job_h5_open[key].ndim == 2:
                # keys with stat metrics
                final_h5_open[key][
                    experiment_index : experiment_index + job_experiments_num,
                    :,
                ] = job_h5_open[key]

            elif job_h5_open[key].ndim == 3:
                # keys with maps

                if key.split("_")[0] == "map":
                    final_h5_open[key][
                        experiment_index : experiment_index
                        + job_experiments_num,
                        :,
                        :,
                    ] = job_h5_open[key]

                elif key.split("_")[0] == "refmap":
                    if virtual_exp:
                        bg_indices = list(set(job_h5_open["background_index"]))

                        for bg_index in bg_indices:
                            # if this background hasn't been saved yet, save it
                            # [note, the same background indices may appear in multiple files
                            # depending on the number of jobs the task has been split into]
                            if final_h5_open[key][bg_index, :, :].sum() == 0.0:
                                final_h5_open[key][
                                    bg_index, :, :
                                ] = job_h5_open[key][bg_index, :, :]
                    else:
                        final_h5_open[key][
                            experiment_index : experiment_index
                            + job_experiments_num,
                            :,
                            :,
                        ] = job_h5_open[key]

                else:
                    raise Exception(
                        f"Unexpected dimension = {job0_h5_open[key].ndim} of the following key: {key}"
                    )

        experiment_index += job_experiments_num
        job_h5_open.close()

    final_h5_open.close()


def suspicious_collected_h5_size(
    out_dir, h5_file_name, collected_to_sum_file_size_ths
):
    """
    Check if the size of a collected HDF5 file is suspiciously small compared to the sum of individual job files.

    This function calculates the size of the collected HDF5 file and the sum of sizes of individual job-specific
    HDF5 files. It then compares the ratio of the collected HDF5 size to the sum of individual job file sizes
    against the specified threshold. If the ratio is less than the threshold, it suggests that the collected HDF5
    file might be suspiciously small compared to the individual job files.

    Parameters
    ------------
    out_dir : str
        Path to the directory containing job-specific HDF5 files.
    h5_file_name : str
        Name of the HDF5 file.
    collected_to_sum_file_size_ths : float
    Threshold ratio for comparison. Should be a float value representing the threshold for suspicious size difference.

    Returns:
    bool: True if the collected HDF5 file size is suspiciously small compared to individual job files,
        False otherwise.
    """

    num_jobs = infer_num_jobs(out_dir)
    collected_h5_path = f"{out_dir}/{h5_file_name}"

    # size of collected h5 file in kb
    collected_h5_size = round(os.stat(collected_h5_path).st_size / (1024), 3)

    # sum of sizes of collected job-files
    file_size_cum = 0
    for job_index in range(num_jobs):
        file_path = f"{out_dir}/job{job_index}/{h5_file_name}"
        file_size = round(os.stat(file_path).st_size / (1024), 3)
        file_size_cum += file_size

    if collected_h5_size / file_size_cum < collected_to_sum_file_size_ths:
        return True
    else:
        return False


def clean_directory(out_dir, h5_file_name):
    """
    Clean up job-specific directories by removing individual HDF5 files.

    This function deletes individual HDF5 files within the specified job directories,
    effectively cleaning up all job-related HDF5 files generated during processing.

    Parameters:
    - out_dir (str): Path to the directory containing job-specific directories.
    - h5_file_name (str): Name of the HDF5 file to be removed from each job directory.

    Returns:
    None
    """

    num_jobs = infer_num_jobs(out_dir)

    for job_index in range(num_jobs):
        os.remove("%s/job%d/%s" % (out_dir, job_index, h5_file_name))
