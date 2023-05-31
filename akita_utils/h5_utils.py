import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from akita_utils.utils import ut_dense
import akita_utils.stats_utils
from akita_utils.stats_utils import insul_diamonds_scores
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


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

        
def initialize_output_h5_v2(
    out_dir,
    stats,
    seq_coords_df,
    target_ids,
    target_labels,
    head_index,
    model_index,
):
    """Initialize an output HDF5 file for SCD stats. fahad's version"""

    num_targets = len(target_ids)
    num_experiments = len(seq_coords_df)
    stats_out = h5py.File(f"{out_dir}/stats_h{head_index}_m{model_index}.h5", "w")
    seq_coords_df_dtypes = seq_coords_df.dtypes

    for key in seq_coords_df:
        if seq_coords_df_dtypes[key] is np.dtype("O"):
            stats_out.create_dataset(key, data=seq_coords_df[key].values.astype("S"))
        else:
            stats_out.create_dataset(key, data=seq_coords_df[key])
            
    # initialize scd stats
    for stat in stats:
        if stat in seq_coords_df.keys():
            raise KeyError("check input tsv for clashing score name")
            
        for target_ind in range(num_targets):
            if "INS" not in stat:
                stats_out.create_dataset(
                    f"{stat}_h{head_index}_m{model_index}_t{target_ind}",
                    shape=(num_experiments,),
                    dtype="float16",
                    compression=None,
                )
            else:
                stats_out.create_dataset(
                    "ref_" + f"{stat}_h{head_index}_m{model_index}_t{target_ind}",
                    shape=(num_experiments,),
                    dtype="float16",
                    compression=None,
                )
                stats_out.create_dataset(
                    "alt_" + f"{stat}_h{head_index}_m{model_index}_t{target_ind}",
                    shape=(num_experiments,),
                    dtype="float16",
                    compression=None,
                )

    log.info(f"Initialized an output HDF5 file for stats ")
    return stats_out


def write_snp_v2(
    ref_preds,
    alt_preds,
    stats_out,
    si,
    head_index,
    model_index,
    diagonal_offset,
    stats=["SCD"],
    plot_dir=None,
    plot_lim=4,
    plot_freq=100,
):
    """Write SNP predictions to HDF. fahad's version"""

    log.info(f"writting SNP predictions for experiment {si}")

    # increase dtype
    ref_preds = ref_preds.astype("float32")
    alt_preds = alt_preds.astype("float32")
            
    if "SCD" in stats:
        # sum of squared diffs
        diff2_preds = (ref_preds - alt_preds) ** 2
        sd2_preds = np.sqrt(diff2_preds.sum(axis=0))
        for target_ind in range(ref_preds.shape[1]):
            stats_out[f"SCD_h{head_index}_m{model_index}_t{target_ind}"][si] = sd2_preds[
                target_ind
            ].astype("float16")
        
    if "SSD" in stats:
        # sum of squared diffs
        ref_ss = (ref_preds**2).sum(axis=0)
        alt_ss = (alt_preds**2).sum(axis=0)
        s2d_preds = np.sqrt(alt_ss) - np.sqrt(ref_ss)
        for target_ind in range(ref_preds.shape[1]):
            stats_out[f"SSD_h{head_index}_m{model_index}_t{target_ind}"][si] = sd2_preds[
                target_ind
            ].astype("float16")
            
    if np.any((["INS" in i for i in stats])):
        ref_map = ut_dense(ref_preds, diagonal_offset)
        alt_map = ut_dense(alt_preds, diagonal_offset)
        for stat in stats:
            if "INS" in stat:
                insul_window = int(stat.split("-")[1])
    
                for target_ind in range(ref_preds.shape[1]):
                    stats_out["ref_" + f"{stat}_h{head_index}_m{model_index}_t{target_ind}"][
                        si
                    ] = akita_utils.stats_utils.insul_diamonds_scores(
                        ref_map, window=insul_window
                    )[
                        target_ind
                    ].astype(
                        "float16"
                    )
                    
                    stats_out["alt_" + f"{stat}_h{head_index}_m{model_index}_t{target_ind}"][
                        si
                    ] = akita_utils.stats_utils.insul_diamonds_scores(
                        alt_map, window=insul_window
                    )[
                        target_ind
                    ].astype(
                        "float16"
                    )
    
        
    if (plot_dir is not None) and (np.mod(si, plot_freq) == 0):
        log.info(f"plotting map for experiment {si}")

        # convert back to dense
        ref_map = ut_dense(ref_preds, diagonal_offset)
        alt_map = ut_dense(alt_preds, diagonal_offset)
        
        _, (ax_ref, ax_alt, ax_diff) = plt.subplots(3, ref_preds.shape[-1], figsize=(21, 6))

        for ti in range(ref_preds.shape[-1]):
            ref_map_ti = ref_map[..., ti]
            alt_map_ti = alt_map[..., ti]
    
            # TEMP: reduce resolution
            ref_map_ti = block_reduce(ref_map_ti, (2, 2), np.mean)
            alt_map_ti = block_reduce(alt_map_ti, (2, 2), np.mean)
            vmin, vmax = (-plot_lim, plot_lim)

            
            sns.heatmap(
                ref_map_ti,
                ax=ax_ref[ti],
                center=0,
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu_r",
                xticklabels=False,
                yticklabels=False,
            )
            sns.heatmap(
                alt_map_ti,
                ax=ax_alt[ti],
                center=0,
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu_r",
                xticklabels=False,
                yticklabels=False,
            )
            sns.heatmap(
                alt_map_ti - ref_map_ti,
                ax=ax_diff[ti],
                center=0,
                cmap="PRGn",
                xticklabels=False,
                yticklabels=False,
            )
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/experiment_{si}.pdf")
        plt.close()
        
def collect_h5(file_name, out_dir, num_procs):
    # count variants
    num_variants = 0
    for pi in range(num_procs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, file_name)
        job_h5_open = h5py.File(job_h5_file, "r")
        num_variants += len(job_h5_open["chrom"])
        job_h5_open.close()

    # initialize final h5
    final_h5_file = "%s/%s" % (out_dir, file_name)
    final_h5_open = h5py.File(final_h5_file, "w")

    # keep dict for string values
    final_strings = {}

    job0_h5_file = "%s/job0/%s" % (out_dir, file_name)
    job0_h5_open = h5py.File(job0_h5_file, "r")
    for key in job0_h5_open.keys():
        if key in ["target_ids", "target_labels"]:
            # copy
            final_h5_open.create_dataset(key, data=job0_h5_open[key])

        elif job0_h5_open[key].dtype.char == "S":
            final_strings[key] = []

        elif job0_h5_open[key].ndim == 1:
            final_h5_open.create_dataset(
                key, shape=(num_variants,), dtype=job0_h5_open[key].dtype
            )

        else:
            num_targets = job0_h5_open[key].shape[1]
            final_h5_open.create_dataset(
                key, shape=(num_variants, num_targets), dtype=job0_h5_open[key].dtype
            )

    job0_h5_open.close()

    # set values
    vi = 0
    for pi in range(num_procs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, file_name)
        job_h5_open = h5py.File(job_h5_file, "r")

        # append to final
        for key in job_h5_open.keys():
            if key in ["target_ids", "target_labels"]:
                # once is enough
                pass

            else:
                if job_h5_open[key].dtype.char == "S":
                    final_strings[key] += list(job_h5_open[key])
                else:
                    job_variants = job_h5_open[key].shape[0]
                    final_h5_open[key][vi : vi + job_variants] = job_h5_open[key]

        vi += job_variants
        job_h5_open.close()

    # create final string datasets
    for key in final_strings:
        final_h5_open.create_dataset(key, data=np.array(final_strings[key], dtype="S"))

    final_h5_open.close()