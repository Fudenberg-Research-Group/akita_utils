import pandas as pd
import numpy as np


def h5_to_df(
    filename,
    scd_stats=["SCD", "SSD", "INS"],
    drop_duplicates_key="index",
    verbose=False,
):

    """
    Load an h5 file, as generated by scripts in akita_utils/bin/
    as a DataFrame for analysis. Currently this function expects
    that keys with statistics are (N x #model_outputs) matrices
    and adds an average across all model outputs as a dataFrame column.

    Parameters
    ----------
    filename : str
        Name of h5 file to be loaded.

    scd_stats : list of str
        Names of keys storing computed statistics.

    drop_duplicates : str or None
        Attempts to drop duplicated rows based on provided str.
        If None, no dupliate removal attempted.

    Returns
    -------
    df_out : pd.DataFrame

    """

    hf = h5py.File(filename, "r")
    s = []
    for key in hf.keys():
        if key.replace("ref_", "").replace("alt_", "").split("-")[0] in scd_stats:
            if verbose:
                print(key)
            s.append(pd.Series(hf[key][()].mean(axis=1), name=key))
        else:
            s.append(pd.Series(hf[key][()], name=key))

    # adding difference between reference and alternate insulation
    insulation_stats = ["INS-16", "INS-32", "INS-64", "INS-128", "INS-256"]
    for key in insulation_stats:
        if "ref_" + key in hf.keys():
            diff = hf["ref_" + key][()].mean(axis=1) - hf["alt_" + key][()].mean(axis=1)
            s.append(pd.Series(diff, name=key))
    hf.close()

    # generating pandas DataFrame and converting bytestrings
    df_out = pd.concat(s, axis=1)
    for key, key_dtype in df_out.dtypes.items():
        if not pd.api.types.is_numeric_dtype(key_dtype):
            df_out[key] = df_out[key].str.decode("utf8").copy()

    if drop_duplicates_key is not None:
        len_orig = len(df_out)
        if drop_duplicates_key not in df_out.keys():
            raise ValueError("duplicate removal key must be present in dataFrame")
        df_out.drop_duplicates(drop_duplicates_key, inplace=True)
        if verbose:
            print(len_orig - len(df_out), "duplicates removed for ", filename)
        df_out.reset_index(inplace=True, drop=True)

    return df_out


def read_jaspar_to_numpy(
    motif_file="/project/fudenber_735/motifs/pfms/JASPAR2022_CORE_redundant_pfms_jaspar/MA0139.1.jaspar",
    normalize=True,
):
    """
    Read a jaspar pfm to a numpy array that can be used with scan_motif. Default motif is CTCF (MA0139.1)

    Parameters
    ----------
    motif_file : str
        Default CTCF motif file.
    normalize :
        Whether to normalize counts to sum to one for each position in the motif. Default True.

    Returns
    -------
    motif : np.array
        n_positions by 4 bases

    """

    with open(motif_file, "r") as f:
        motif = []
        for line in f.readlines():
            if ">" in line:
                continue
            else:
                motif.append(line.strip().replace("[", "").replace("]", "").split())
    motif = pd.DataFrame(motif).set_index(0).astype(float).values.T
    if normalize is True:
        motif /= motif.sum(axis=1)[:, None]
    if motif.shape[1] != 4:
        raise ValueError("motif returned should be have n_positions x 4 bases")
    return motif