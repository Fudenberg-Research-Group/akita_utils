import pandas as pd
import h5py
from io import StringIO
import numpy as np


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
            key for key in h5_file.keys() if keyword in key and key not in keyword
        ]
        if not keys:
            raise Exception(f"There are no matching keys for the following keyword: {keyword}")
            
        data = pd.DataFrame()
        for key in keys:
            nr_targets = h5_file[key][()].shape[1]
            for target_index in range(nr_targets):
                series = pd.Series(h5_file[key][:,target_index], name=key + f"_t{target_index}")
                data = pd.concat([data, series], axis=1)
            
        # averaging over targets and / or models
        average = data.mean(axis=1)
        df = pd.concat(
            [df, pd.Series(average, name=keyword)], axis=1
        )
    
    return df


def collect_all_keys_with_keywords(h5_file, df, keywords):
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
            key for key in h5_file.keys() if keyword in key and key not in keyword
        ]
        if not keys:
            raise Exception(f"There are no matching keys for the following keyword: {keyword}")
            
        for key in keys:
            nr_targets = h5_file[key][()].shape[1]
            for target_index in range(nr_targets):
                series = pd.Series(h5_file[key][:,target_index], name=key + f"_t{target_index}")
                data = pd.concat([data, series], axis=1)
            
    return data


def calculate_INS(df, keywords, drop=True):
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
            if ("ref_" + key in df.columns and "alt_" + key in df.columns):
                df_out[key] = df_out["alt_" + key] - df_out["ref_" + key]
                if drop:
                    df_out = df_out.drop(columns=["alt_" + key, "ref_" + key])
            else:
                raise Exception(f"alt_ and ref_ columns cannot be found for the following keyword: {keyword}")

        return df_out
    
    else:
        return df


def calculate_INS_by_targets(df, keywords, max_nr_targets=6, max_nr_heads=2, max_nr_models=8, drop=True):
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
                
                        if ("ref_" + key in df.columns and "alt_" + key in df.columns):
                            df_out[key] = df_out["alt_" + key] - df_out["ref_" + key]
                            if drop:
                                df_out = df_out.drop(columns=["alt_" + key, "ref_" + key])
        
        return df_out
    
    else:
        return df


def h5_to_df(
    filename,
    stats=["SCD"],
    average=True,
    verbose=False,
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
            print("Stat metrics have been averaged over models and/or targets")
            
    else:
        # collecting all columns with stats
        df_out = collect_all_keys_with_keywords(hf, df_out, stats)
        if verbose:
            print("Stat metrics have been NOT averaged over models and/or targets")
            
    remaining_keys = [key for key in hf.keys() if all(stat not in key for stat in stats)]
    exact_matches = [key for key in hf.keys() if key in stats]

    for key in remaining_keys:
        if verbose:
            print(f"Remaining h5 file keys: {key}")
        df_out = pd.concat([df_out, pd.Series(hf[key][()], name=key)], axis=1)

    for key in exact_matches:
        # it's possible that this stat metric have been already averaged 
        if verbose:
            print(f"Exact match thus using old pipeline processing for {key}")
        df_out = pd.concat(
            [df_out,
                pd.Series(pd.DataFrame(hf[key][()]).mean(axis=1), name=key)],
            axis=1,
        )
        
    hf.close()

    if average:
        df_out = calculate_INS(df_out, stats)
    else:
        df_out = calculate_INS_by_targets(df_out, stats)

    # checking and optionally correcting dtypes
    for key, key_dtype in df_out.dtypes.items():
        if not pd.api.types.is_numeric_dtype(key_dtype):
            df_out[key] = df_out[key].str.decode("utf8").copy()

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
                motif.append(
                    line.strip().replace("[", "").replace("]", "").split()
                )
    motif = pd.DataFrame(motif).set_index(0).astype(float).values.T
    if normalize is True:
        motif /= motif.sum(axis=1)[:, None]
    if motif.shape[1] != 4:
        raise ValueError("motif returned should be have n_positions x 4 bases")
    return motif


def read_rmsk(
    rmsk_file="/project/fudenber_735/genomes/mm10/database/rmsk.txt.gz",
):

    """reads a data frame containing repeatable elements and renames columns specifying genomic intervals to standard: chrom, start, end, used in thie repo."""

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
        columns={"genoName": "chrom", "genoStart": "start", "genoEnd": "end"},
        inplace=True,
    )

    return rmsk
