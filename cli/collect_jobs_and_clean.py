# This script collects and processes HDF5 files generated from genomic experiments,
# ensuring proper data collection and directory management. It supports both virtual and genomic experiments,
# and provides options for cleaning up the output directory.
#
# Inputs:
# - out_dir: Output directory where the collected HDF5 files and logs will be stored.
#
# Parameters:
# - h5_file_name: Name of the HDF5 files to collect (default: "STATS_OUT.h5").
# - tsv_path: Path to the table with experiment specifications.
# - virtual_experiment: Flag indicating a virtual experiment (default: True).
# - genomic_experiment: Flag indicating a genomic experiment (default: True).
# - collecting_maps: Flag to indicate if maps should be collected (default: True).
# - not_collecting_maps: Flag to indicate if maps should not be collected (default: True).
# - leave_files: Flag to indicate if the output directory should not be cleaned (default: True).
# - not_leave_files: Flag to indicate if the output directory should be cleaned (default: True).
# - collected_to_sum_file_size_ths: Threshold for suspicious size difference (default: 0.8).
#
# Outputs:
# - Collected HDF5 files stored in the specified output directory.
#
# Example command-line usage:
# python collect_h5_files.py -f STATS_OUT.h5 -d experiment_specifications.tsv -g -m -l out_directory


import os
import pandas as pd

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from optparse import OptionParser
from akita_utils.h5_utils import (
    collect_h5,
    suspicious_collected_h5_size,
    clean_directory,
)


def main():
    usage = "usage: %prog [options] <out_dir>"
    parser = OptionParser(usage)
    parser.add_option(
        "-f",
        dest="h5_file_name",
        default="STATS_OUT.h5",
        type="str",
        help="Name of h5 files to collect [Default: %default]",
    )
    parser.add_option(
        "-d",
        dest="tsv_path",
        default=None,
        type="str",
        help="Path to the table with experiments specification [Default: %default]",
    )
    parser.add_option(
        "-v",
        dest="virtual_experiment",
        default=True,
        action="store_true",
        help="True, if virtual experiment [Default: %default]",
    )
    parser.add_option(
        "-g",
        dest="genomic_experiment",
        default=True,
        action="store_true",
        help="True, if genomic experiment [Default: %default]",
    )
    parser.add_option(
        "-m",
        dest="collecting_maps",
        default=True,
        action="store_true",
        help="Collecting maps [Default: %default]",
    )
    parser.add_option(
        "-n",
        dest="not_collecting_maps",
        default=True,
        action="store_true",
        help="Not collecting maps [Default: %default]",
    )
    parser.add_option(
        "-l",
        dest="leave_files",
        default=True,
        action="store_true",
        help="Don't clean the output directory [Default: %default]",
    )
    parser.add_option(
        "-c",
        dest="not_leave_files",
        default=True,
        action="store_true",
        help="Clean the output directory [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="collected_to_sum_file_size_ths",
        default=0.8,
        type="float",
        help="The threshold for suspicious size difference [Default: %default]",
    )

    (options, args) = parser.parse_args()

    if len(args) == 1:
        out_dir = args[0]
    else:
        raise Exception(
            "Too many arguments have been provided. Check the command, please."
        )

    exp_dataframe = pd.read_csv(options.tsv_path, sep="\t")
    if "Unnamed: 0" in exp_dataframe.columns:
        exp_dataframe = exp_dataframe.drop(columns=["Unnamed: 0"])

    if options.not_collecting_maps == True:
        options.collecting_maps = False

    if options.genomic_experiment == True:
        options.virtual_experiment = False

    if options.not_leave_files == True:
        options.leave_files == False

    # By default, stat metrics are collected.
    # Otherwise, h5 files with maps are supposed to be collected with default file name MAPS_OUT.h5.
    if (
        options.collecting_maps == True
        and options.h5_file_name == "STATS_OUT.h5"
    ):
        options.h5_file_name = "MAPS_OUT.h5"

    # data collection
    print("Collecting h5 files...")

    collect_h5(
        out_dir,
        exp_dataframe,
        options.h5_file_name,
        virtual_exp=options.virtual_experiment,
    )

    if not options.leave_files:
        if suspicious_collected_h5_size(
            out_dir,
            options.h5_file_name,
            options.collected_to_sum_file_size_ths,
        ):
            raise Exception(
                "Please, check the collected file. Job-files have not been deleted yet since the sum of their sizes is suspiciously bigger than the size of the collected h5 file."
            )
        else:
            print("Cleaning the directory...")
            clean_directory(out_dir, options.h5_file_name)


if __name__ == "__main__":
    main()
