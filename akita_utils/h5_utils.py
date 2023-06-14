import h5py
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from akita_utils.utils import ut_dense
from skimage.measure import block_reduce
from akita_utils.stats_utils import calculate_scores, calculate_INS
import logging
import seaborn as sns

sns.set(style="ticks", font_scale=1.3)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def initialize_output_h5(
    out_dir, model_file, genome_fasta, seqnn_model, stat_metrics, seq_coords_df
):
    """
    Initializes an h5 file to save statistical metrics calculated from Akita's predicftions.

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
    stat_metrics : list
        List of stratistical metrics that are supposed to be calculated.
    seq_coords_df : dataFrame
        Pandas dataframe where each row represents one experiment (so one set of prediction).

    Returns
    ---------
    h5_outfile : h5py object
        An initialized h5 file.
    """

    h5_outfile = h5py.File(f"{out_dir}/OUT.h5", "w")
    seq_coords_df_dtypes = seq_coords_df.dtypes

    h5_outfile.attrs["date"] = str(date.today())

    metadata_dict = {
        "model_index": int(model_file.split("c0")[0][-1]),
        "head_index": int(model_file.split("model")[-1][0]),
        "genome": genome_fasta.split("/")[-1],
        "seq_length": seqnn_model.seq_length,
        "diagonal_offset": seqnn_model.diagonal_offset,
        "prediction_vector_length": seqnn_model.target_lengths[0],
        "target_crops": seqnn_model.target_crops,
        "num_targets": seqnn_model.num_targets(),
    }
    model_index = metadata_dict["model_index"]
    head_index = metadata_dict["head_index"]

    h5_outfile.attrs.update(metadata_dict)

    num_targets = seqnn_model.num_targets()
    target_ids = [ti for ti in range(num_targets)]

    num_experiments = len(seq_coords_df)

    for key in seq_coords_df:
        if seq_coords_df_dtypes[key] is np.dtype("O"):
            h5_outfile.create_dataset(key, data=seq_coords_df[key].values.astype("S"))
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

    return h5_outfile


def write_stat_metrics_to_h5(
    prediction_matrix,
    reference_map_matrix,
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
    reference_map_matrix : numpy matrix
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
    scores = calculate_scores(stat_metrics, map_matrix, reference_map_matrix)

    for key in scores:
        for target_index in range(prediction_matrix.shape[1]):
            h5_outfile[f"{key}_h{head_index}_m{model_index}_t{target_index}"][
                experiment_index
            ] = scores[key][target_index].astype("float16")


def write_maps_to_h5(
    prediction_matrix,
    h5_outfile,
    experiment_index,
    head_index,
    model_index,
    diagonal_offset=2,
    plot_dir=None,
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
        h5_outfile[
            f"e{experiment_index}_h{head_index}_m{model_index}_t{target_index}"
        ] = map_matrix[:, :, target_index]


# TODO: this function should be moved somewhere else - where?


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


def initialize_output_h5_v2(
    out_dir,
    stats,
    seq_coords_df,
    target_ids,
    target_labels,
    head_index,
    model_index,
):
    """
    Initializes an output HDF5 file for SCD stats.

    Args:
        out_dir (str): Directory to save the output file.
        stats (list): List of statistics to initialize.
        seq_coords_df (DataFrame): DataFrame containing sequence coordinates.
        target_ids (list): List of target IDs.
        target_labels (list): List of target labels.
        head_index (int): Head index.
        model_index (int): Model index.

    Returns:
        h5py.File: Initialized output HDF5 file for stats.
    """

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

    log.info(f"Initialized an output HDF5 file for stats {stats}")
    return stats_out


def write_stats(
    ref_preds,
    alt_preds,
    stats_out,
    experiment_ID,
    head_index,
    model_index,
    diagonal_offset,
    stats=["SCD"],
):
    """
    Writes statistics to HDF file.

    Args:
        ref_preds (ndarray): Reference predictions.
        alt_preds (ndarray): Alternate predictions.
        stats_out (h5py.File): HDF file to write the statistics.
        experiment_ID (int): Experiment ID.
        head_index (int): Head index.
        model_index (int): Model index.
        diagonal_offset (int): Diagonal offset for the maps.
        stats (list, optional): List of statistics to write. Defaults to ["SCD"].
    """

    log.info(f"writting scores for experiment {experiment_ID}")

    # increase dtype
    ref_preds = ref_preds.astype("float32")
    alt_preds = alt_preds.astype("float32")

    if "SCD" in stats:
        # sum of squared diffs
        diff2_preds = (alt_preds - ref_preds) ** 2
        sd2_preds = np.sqrt(diff2_preds.sum(axis=0))
        for target_ind in range(ref_preds.shape[1]):
            stats_out[f"SCD_h{head_index}_m{model_index}_t{target_ind}"][
                experiment_ID
            ] = sd2_preds[target_ind].astype("float16")

    if "SSD" in stats:
        # sum of squared diffs
        ref_ss = (ref_preds**2).sum(axis=0)
        alt_ss = (alt_preds**2).sum(axis=0)
        s2d_preds = np.sqrt(alt_ss) - np.sqrt(ref_ss)
        for target_ind in range(ref_preds.shape[1]):
            stats_out[f"SSD_h{head_index}_m{model_index}_t{target_ind}"][
                experiment_ID
            ] = s2d_preds[target_ind].astype("float16")

    if np.any((["INS" in i for i in stats])):
        ref_map = ut_dense(ref_preds, diagonal_offset)
        alt_map = ut_dense(alt_preds, diagonal_offset)
        for stat in stats:
            if "INS" in stat:
                insul_window = int(stat.split("-")[1])

                for target_ind in range(ref_preds.shape[1]):
                    stats_out[
                        "ref_" + f"{stat}_h{head_index}_m{model_index}_t{target_ind}"
                    ][experiment_ID] = calculate_INS(ref_map, window=insul_window)[
                        target_ind
                    ].astype(
                        "float16"
                    )

                    stats_out[
                        "alt_" + f"{stat}_h{head_index}_m{model_index}_t{target_ind}"
                    ][experiment_ID] = calculate_INS(alt_map, window=insul_window)[
                        target_ind
                    ].astype(
                        "float16"
                    )


def plot_maps(
    ref_preds,
    alt_preds,
    experiment_ID,
    diagonal_offset,
    plot_dir,
    plot_lim=4,
    plot_freq=100,
):
    """
    Plots maps for the given reference and alternate predictions.

    Args:
        ref_preds (ndarray): Reference predictions.
        alt_preds (ndarray): Alternate predictions.
        experiment_ID (int): Experiment ID.
        diagonal_offset (int): Diagonal offset for the maps.
        plot_dir (str): Directory to save the plots.
        plot_lim (int, optional): Limit for the color range in the plots. Defaults to 4.
        plot_freq (int, optional): Frequency of plotting maps. Defaults to 100.
    """
    if np.mod(experiment_ID, plot_freq) == 0:
        log.info(f"plotting map for experiment {experiment_ID}")

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
            ax=ax_ref[ti],  # ref map
            center=0,
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
            xticklabels=False,
            yticklabels=False,
        )
        ax_ref[ti].set_title("Reference Map")

        sns.heatmap(
            alt_map_ti,
            ax=ax_alt[ti],  # alt map
            center=0,
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
            xticklabels=False,
            yticklabels=False,
        )
        ax_alt[ti].set_title("Alternate Map")

        sns.heatmap(
            alt_map_ti - ref_map_ti,  # diff map
            ax=ax_diff[ti],
            center=0,
            cmap="PRGn",
            xticklabels=False,
            yticklabels=False,
        )
        ax_diff[ti].set_title("Difference Map")

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/experiment_{experiment_ID}.pdf")
    plt.close()


def collect_h5(file_name, out_dir, num_procs):
    """
    Collects H5 files from multiple processes and merges them into a final H5 file.

    Args:
        file_name (str): Name of the file to be collected.
        out_dir (str): Directory where the files are located.
        num_procs (int): Number of processes/files to be collected.
    """
    # Count variants
    num_variants = 0
    for process in range(num_procs):
        # Open job
        job_h5_file = "%s/job%d/%s" % (out_dir, process, file_name)
        job_h5_open = h5py.File(job_h5_file, "r")
        num_variants += len(job_h5_open["chrom"])
        job_h5_open.close()

    final_h5_open, final_strings = _initialize_final_h5_file(
        out_dir, file_name, num_variants
    )

    _write_jobs_h5_to_final_h5(
        out_dir, file_name, final_h5_open, final_strings, num_procs
    )


def _initialize_final_h5_file(out_dir, file_name, num_variants):
    """
    Initializes the final H5 file.

    Args:
        out_dir (str): Directory where the file will be located.
        file_name (str): Name of the file to be initialized.
        num_variants (int): Number of variants in the final file.

    Returns:
        final_h5_open (h5py.File): The final H5 file object.
        final_strings (dict): Dictionary to store string values.
    """
    # Initialize final H5 file
    final_h5_file = "%s/%s" % (out_dir, file_name)
    final_h5_open = h5py.File(final_h5_file, "w")

    # Keep dict for string values
    final_strings = {}

    job0_h5_file = "%s/job0/%s" % (out_dir, file_name)
    job0_h5_open = h5py.File(job0_h5_file, "r")
    for key in job0_h5_open.keys():
        if key in ["target_ids", "target_labels"]:
            # Copy
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

    return final_h5_open, final_strings


def _write_jobs_h5_to_final_h5(
    out_dir, file_name, final_h5_open, final_strings, num_procs
):
    """
    Writes the data from individual job H5 files to the final H5 file.

    Args:
        out_dir (str): Directory where the files are located.
        file_name (str): Name of the file being collected.
        final_h5_open (h5py.File): The final H5 file object.
        final_strings (dict): Dictionary containing string values.
        num_procs (int): Number of processes/files being collected.
    """
    # set values
    vi = 0
    for process in range(num_procs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, process, file_name)
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
