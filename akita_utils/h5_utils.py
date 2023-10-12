import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

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
  
    h5_outfile = h5py.File(f"%s/STATS_OUT.h5" % out_dir, "w")
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
            h5_outfile.create_dataset(
                key, data=seq_coords_df[key].values.astype("S")
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
                "ref_"
                + f"{stat_metric}_h{head_index}_m{model_index}",
                shape=(num_experiments, num_targets),
                dtype="float16",
                compression=None,
            )
            h5_outfile.create_dataset(
                "alt_"
                + f"{stat_metric}_h{head_index}_m{model_index}",
                shape=(num_experiments, num_targets),
                dtype="float16",
                compression=None,
            )
                
    return h5_outfile


def initialize_maps_output_h5_background(
    out_dir, model_file, genome_fasta, seqnn_model, seq_coords_df
):
    """
    Initializes an h5 file to save vectors predicted by Akita.

    Parameters
    ------------
    out_dir : str
        Path to the desired location of the output h5 file.
    model_file : str
        Path to the model file.
    genome_fasta : str
        Path to the genome file (mouse or human).
    seqnn_model : object
        Loaded model.
    seq_coords_df : dataFrame
        Pandas dataframe where each row represents one experiment (so one set of prediction).

    Returns
    ---------
    h5_outfile : h5py object
        An initialized h5 file.
    """
    h5_outfile = h5py.File(f"%s/MAPS_OUT.h5" % out_dir, "w")
    seq_coords_df_dtypes = seq_coords_df.dtypes

    head_index = int(model_file.split("model")[-1][0])
    model_index = int(model_file.split("c0")[0][-1])
    prediction_vector_length = seqnn_model.target_lengths[0]

    num_backgrounds = len(seq_coords_df.background_index.unique())
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
    reference_prediction_matrix = reference_prediction_matrix.astype("float32")
    
    # convert prediction vectors to maps
    map_matrix = ut_dense(prediction_matrix, diagonal_offset)
    ref_map_matrix = ut_dense(reference_prediction_matrix, diagonal_offset)
    
    # getting desired scores
    scores = calculate_scores(stat_metrics, map_matrix, ref_map_matrix)

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
    plot_freq=100
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
            map_target = block_reduce(map_target, (2, 2), np.mean)
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

        # TODO: we can save SCD or other scores in the name of plots
        # we should let user change which metric
        # such that they can sort plots by name

        plt.savefig(
            f"e{experiment_index}_h{head_index}_m{model_index}_t{target_index}.pdf"
        )
        plt.close()


# COLLECTING DATA FROM MULTIPLE HDF5 FILES

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
        for object_name in folder:
            if object_name[:3] == "job":
                index = int(object_name.split("job")[1].split(".")[0])
                if index > highest_index:
                    highest_index = index
    
    num_job = highest_index + 1
    return num_job


def collect_h5(out_dir, h5_file_name="STATS_OUT.h5"):
    """
    Collects statistics from multiple job-specific HDF5 files into a final HDF5 file.
    
    This function collects statistics from individual job-specific HDF5 files stored
    in directories named "jobX" within the specified directory. It combines the data
    into a single HDF5 file, considering different dimensions (1D, 2D, 3D) of the
    statistics matrices.
    
    Parameters
    ------------
    out_dir : str
        Path to the directory containing job-specific HDF5 files.
    h5_file_name : str, optional
        Name of the output HDF5 file. Default is "STATS_OUT.h5".
    
    Returns
    ---------
        None
    """
    num_jobs = infer_num_jobs(out_dir)
    
    # count experiments (number of sites x number of brackground if applies)
    num_experiments = 0
    for job_index in range(num_jobs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, job_index, h5_file_name)
        job_h5_open = h5py.File(job_h5_file, "r")
        num_experiments += len(job_h5_open["chrom"])
        job_h5_open.close()
        
    # initialize final final h5 based on the 0th-job file
    final_h5_file = "%s/%s" % (out_dir, h5_file_name)
    final_h5_open = h5py.File(final_h5_file, "w")

    job0_h5_file = "%s/job0/%s" % (out_dir, h5_file_name)
    job0_h5_open = h5py.File(job0_h5_file, "r")

    for key in job0_h5_open.keys():

        if job0_h5_open[key].ndim == 1:
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
                _, num_targets = job0_h5_open[key].shape

                final_h5_open.create_dataset(
                    key,
                    shape=(
                        num_variants,
                        prediction_vector_length,
                        num_targets,
                    ),
                    dtype=job0_h5_open[key].dtype,
                )
            else:
                (
                    num_backgrounds,
                    prediction_vector_length,
                    num_targets,
                ) = job0_h5_open[key].shape
        
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
            raise Exception(f"Unexpected dimension = {job0_h5_open[key].ndim} of the following key: {key}")

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
                final_h5_open[key][experiment_index : experiment_index + job_experiments_num] = job_h5_open[key]

            elif job_h5_open[key].ndim == 2:
                # keys with stat metrics
                final_h5_open[key][experiment_index : experiment_index + job_experiments_num, :] = job_h5_open[key]
                
            elif job_h5_open[key].ndim == 3:

                if key.split("_")[0] == "map":
                    final_h5_open[key][experiment_index : experiment_index + job_experiments_num, :, :] = job_h5_open[key]

                else:
                    num_backgrounds, _, _ = job_h5_open[key].shape
                    final_h5_open[key][num_backgrounds, :, :] = job_h5_open[key]

        experiment_index += job_experiments_num
        job_h5_open.close()

    final_h5_open.close()

