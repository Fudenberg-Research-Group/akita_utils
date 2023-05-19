import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from akita_utils.utils import ut_dense
from akita_utils.stats_utils import calculate_scores


def initialize_output_h5(
    out_dir, seq_coords_df, stat_metrics, target_ids, head_index, model_index
):
    """
    Initializes an h5 file to save statistical metrics calculated from Akita's predicftions.

    Parameters
    ------------
    out_dir : str
        Path to the desired location of the output h5 file.
    seq_coords_df : dataFrame
        Pandas dataframe where each row represents one experiment (so one set of prediction).
    stat_metrics : list
        List of stratistical metrics that are supposed to be calculated.
    target_ids : list
        List of target indices.
    head_index : int
        Head index used to get a prediction (Mouse: head_index=1; Human: head_index=0).
    model_index : int
        Index of one of 8 models that has been used to make predictions (an index between 0 and 7).

    Returns
    ---------
    h5_outfile : h5py object
        An initialized h5 file.
    """

    num_experiments = len(seq_coords_df)

    h5_outfile = h5py.File(f"%s/stat_summary_h{head_index}_m{model_index}.h5" % out_dir, "w")
    seq_coords_df_dtypes = seq_coords_df.dtypes

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

        for target_index in target_ids:
            h5_outfile.create_dataset(
                f"{stat_metric}_h{head_index}_m{model_index}_t{target_index}",
                shape=(num_experiments,),
                dtype="float16",
                compression=None,
            )

    return scd_out


def write_stat_metrics_to_h5(
    prediction_matrix,
    h5_outfile,
    experiment_index,
    head_index,
    model_index,
    diagonal_offset=2,
    stat_metrics=["SCD"]
):
    """
    Writes to an h5 file saving statistical metrics calculated from Akita's predicftions.

    Parameters
    ------------
    prediction_matrix : numpy matrix
        Matrix collecting Akita's predictions.
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
    plot_dir : str
        Path to the desired location of the output plots (plots will not be saved if plot_dir == None).
    plot_lim_min : float
        Negative minimum and positive maximum values that will be used to plot maps.
    plot_freq : int
        A plot of one out of plot_freq number of predictions is saved.
    """

    # increase dtype
    prediction_matrix = prediction_matrix.astype("float32")
    
    # convert prediction vectors to maps
    map_matrix = ut_dense(prediction_matrix, diagonal_offset)
    
    # getting desired scores
    # TODO: reference_map_matrix may not be None if the experiments are done in the genomic context,
    # it should be adjusted in the future
    scores = calculate_scores(stat_metrics, map_matrix, reference_map_matrix=None)
    
    for score in scores:
        for target_index in range(prediction_matrix.shape[1]):
            h5_outfile[f"{score}_h{head_index}_m{model_index}_t{target_index}"][
                experiment_index
            ] = score[target_ind].astype("float16")


def write_maps_to_h5(
    prediction_matrix,
    h5_outfile,
    experiment_index,
    head_index,
    model_index,
    diagonal_offset=2,
    plot_dir=None
):
    """
    Writes entire maps to an h5 file.

    Parameters
    ------------
    prediction_matrix : numpy matrix
        Matrix collecting Akita's predictions.
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
    plot_dir : str
        Path to the desired location of the output plots (plots will not be saved if plot_dir == None).
    """
    
    # increase dtype
    prediction_matrix = prediction_matrix.astype("float32")
    
    # convert prediction vectors to maps
    map_matrix = ut_dense(prediction_matrix, diagonal_offset)
    
    for target_index in range(prediction_matrix.shape[1]):
        h5_outfile[f"e{experiment_index}_h{head_index}_m{model_index}_t{target_index}"] = map_matrix[:, :, target_index]
        
            
# TODO: this function should be moved somewhere else - where?
            
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
        
        plt.savefig(f"e{experiment_index}_h{head_index}_m{model_index}_t{target_index}.pdf")
        plt.close()
