import pandas as pd
import h5py
from io import StringIO

from akita_utils.h5_utils import (
    average_over_keys,
    collect_all_keys_with_keywords,
)
from akita_utils.stats_utils import (
    calculate_INS_keywords,
    calculate_INS_by_targets_keywords,
)


def h5_to_df(
    filename,
    stats=["SCD"],
    average=True,
    verbose=False,
    ignore_keys=[],
):
    """
    Load an HDF5 file, extract specified statistics, and create a DataFrame for analysis.

    This function loads an HDF5 file and extracts specified statistics from keys in the file.
    The extracted statistics are processed based on the specified options, such as averaging
    over models and/or targets, and insulation (INS) values are calculated accordingly. The
    function returns a DataFrame containing the processed data.

    Parameters
    ----------
    filename : str
        Name of the HDF5 file to be loaded.
    stats : list of str
        Names of keys storing computed statistics.
    average : bool
        True if stat metrics are supposed to be averaged over targets and models.
    verbose : bool
        True if step-by-step information is desired to be printed.

    Returns
    -------
    pd.DataFrame: A DataFrame containing processed statistics from the HDF5 file.
    """

    hf = h5py.File(filename, "r")
    df_out = pd.DataFrame()

    # adding "alt_" and "ref_" to insulation scores
    extended_stats = []
    for stat in stats:
        if "INS" in stat:
            extended_stats.append("alt_" + stat)
            extended_stats.append("ref_" + stat)
        else:
            extended_stats.append(stat)
    stats = extended_stats

    if average:
        df_out = average_over_keys(hf, df_out, stats)
        if verbose:
            print(
                "Stat metrics have been averaged over models and/or targets"
            )

    else:
        # collecting all columns with stats
        df_out = collect_all_keys_with_keywords(
            hf, df_out, stats, ignore_keys=ignore_keys
        )
        if verbose:
            print(
                "Stat metrics have been NOT averaged over models and/or targets"
            )

    remaining_keys = [
        key
        for key in hf.keys()
        if all(stat not in key for stat in stats)
    ]
    remaining_keys = remaining_keys + ignore_keys
    exact_matches = [key for key in hf.keys() if key in stats]

    for key in remaining_keys:
        if verbose:
            print(f"Remaining h5 file keys: {key}")
        df_out = pd.concat(
            [df_out, pd.Series(hf[key][()], name=key)], axis=1
        )

    for key in exact_matches:
        # it's possible that this stat metric have been already averaged
        if verbose:
            print(
                f"Exact match thus using old pipeline processing for {key}"
            )
        df_out = pd.concat(
            [
                df_out,
                pd.Series(
                    pd.DataFrame(hf[key][()]).mean(axis=1), name=key
                ),
            ],
            axis=1,
        )

    hf.close()

    if average:
        df_out = calculate_INS_keywords(df_out, stats)
    else:
        df_out = calculate_INS_by_targets_keywords(df_out, stats)

    # checking and optionally correcting dtypes
    for key, key_dtype in df_out.dtypes.items():
        if not pd.api.types.is_numeric_dtype(key_dtype):
            df_out[key] = df_out[key].str.decode("utf8").copy()

    # removing index if it had been read
    if "Unnamed: 0" in df_out.columns:
        df_out = df_out.drop(columns=["Unnamed: 0"])

    return df_out


def read_jaspar_to_numpy(
    motif_file="/project/fudenber_735/motifs/pfms/JASPAR2022_CORE_redundant_pfms_jaspar/MA0139.1.jaspar",
    normalize=True,
):
    """
    Read a JASPAR motif file into a numpy array.

    Parameters
    ------------
    motif_file : str, optional
        Path to the JASPAR motif file. Default is a specific path.
    normalize : bool, optional
        If True, normalize the motif matrix by dividing each row by its sum. Default is True.

    Returns
    ---------
    motif : numpy array
        A 2D array of shape (n_positions, 4) representing the motif matrix, where each row corresponds
        to the probabilities of 'A', 'C', 'G', 'T' at that position.
    """
    with open(motif_file, "r") as f:
        motif = []
        for line in f.readlines():
            if ">" in line:
                continue
            else:
                motif.append(
                    line.strip()
                    .replace("[", "")
                    .replace("]", "")
                    .split()
                )
    motif = pd.DataFrame(motif).set_index(0).astype(float).values.T
    if normalize is True:
        motif /= motif.sum(axis=1)[:, None]
    if motif.shape[1] != 4:
        raise ValueError(
            "motif returned should be have n_positions x 4 bases"
        )
    return motif


def read_rmsk(
    rmsk_file="/project/fudenber_735/genomes/mm10/database/rmsk.txt.gz",
):
    """
    Read RepeatMasker annotation file into a pandas DataFrame.

    Parameters
    ------------
    rmsk_file : str, optional
        Path to the RepeatMasker annotation file. Default is a specific path.

    Returns
    ---------
    rmsk : pandas DataFrame
        DataFrame containing RepeatMasker annotation data with columns: 'chrom', 'start', 'end',
        'genoLeft', 'strand', 'repName', 'repClass', 'repFamily', 'repStart', 'repEnd', 'id'.
    """
    rmsk_cols = list(
        pd.read_csv(
            StringIO(
                """bin swScore milliDiv milliDel milliIns genoName genoStart genoEnd genoLeft strand repName repClass repFamily repStart repEnd repLeft id"""
            ),
            sep=" ",
        )
    )

    rmsk = pd.read_table(
        rmsk_file,
        names=rmsk_cols,
    )

    rmsk.rename(
        columns={
            "genoName": "chrom",
            "genoStart": "start",
            "genoEnd": "end",
        },
        inplace=True,
    )

    return rmsk
