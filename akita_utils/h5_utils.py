import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.measure import block_reduce
from akita_utils.utils import ut_dense
from akita_utils.stats_utils import insul_diamonds_scores


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
    scd_out : h5py object
        An initialized h5 file.
    """

    num_experiments = len(seq_coords_df)

    scd_out = h5py.File("%s/scd.h5" % out_dir, "w")
    seq_coords_df_dtypes = seq_coords_df.dtypes

    for key in seq_coords_df:
        if seq_coords_df_dtypes[key] is np.dtype("O"):
            scd_out.create_dataset(
                key, data=seq_coords_df[key].values.astype("S")
            )
        else:
            scd_out.create_dataset(key, data=seq_coords_df[key])

    # initialize keys for statistical metrics collection
    for stat_metric in stat_metrics:

        if stat_metric in seq_coords_df.keys():
            raise KeyError("check input tsv for clashing score name")

        for target_index in target_ids:
            scd_out.create_dataset(
                f"{stat_metric}_h{head_index}_m{model_index}_t{target_index}",
                shape=(num_experiments,),
                dtype="float16",
                compression=None,
            )

    return scd_out


def write_to_h5_stats_for_prediction(
    prediction_matrix,
    scd_out,
    experiment_index,
    head_index,
    model_index,
    diagonal_offset=2,
    stat_metrics=["SCD"],
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
    scd_out : h5py object
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

    # saving
    if "SCD" in stat_metrics:
        SCDs = np.sqrt((prediction_matrix**2).sum(axis=0))
        for target_ind in range(prediction_matrix.shape[1]):
            scd_out[f"SCD_h{head_index}_m{model_index}_t{target_ind}"][
                experiment_index
            ] = SCDs[target_ind].astype("float16")

    if np.any((["INS" in i for i in stat_metrics])):
        ref_map = ut_dense(prediction_matrix, diagonal_offset)
        for stat in stat_metrics:
            if "INS" in stat:
                insul_window = int(stat.split("-")[1])
                for target_ind in range(prediction_matrix.shape[1]):
                    scd_out[
                        f"{stat}_h{head_index}_m{model_index}_t{target_ind}"
                    ][experiment_index] = insul_diamonds_scores(
                        ref_map, window=insul_window
                    )[
                        target_ind
                    ].astype(
                        "float16"
                    )

    if (plot_dir is not None) and (np.mod(experiment_index, plot_freq) == 0):
        print("plotting prediction: ", experiment_index)

        ref_map = ut_dense(prediction_matrix, diagonal_offset)
        _, axs = plt.subplots(1, prediction_matrix.shape[-1], figsize=(24, 4))

        for target_index in range(prediction_matrix.shape[-1]):
            ref_map_target = ref_map[..., target_index]
            ref_map_target = block_reduce(ref_map_target, (2, 2), np.mean)
            vmin = min(ref_map_target.min(), ref_map_target.min())
            vmax = max(ref_map_target.max(), ref_map_target.max())
            vmin = min(-plot_lim_min, vmin)
            vmax = max(plot_lim_min, vmax)
            sns.heatmap(
                ref_map_target,
                ax=axs[target_index],
                center=0,
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu_r",
                xticklabels=False,
                yticklabels=False,
            )

        plt.tight_layout()
        plt.savefig("%s/s%d.pdf" % (plot_dir, experiment_index))
        plt.close()
