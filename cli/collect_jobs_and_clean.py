#!/usr/bin/env python

# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

###################################################

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from optparse import OptionParser
from akita_utils.h5_utils import collect_h5, suspicious_collected_h5_size, clean_directory


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
        "-s",
        dest="collecting_stats",
        default=True,
        action="store_false",
        help="Not collecting stat metrics [Default: %default]",
    )
    parser.add_option(
        "-c",
        dest="clean_dir",
        default=True,
        action="store_false",
        help="Don't clean the output directory [Default: %default]",
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
        raise Exception("Too many arguments have been provided. Check the command, please.")

    # By default, stat metrics are collected. 
    # Otherwise, h5 files with maps are supposed to be collected with default file name MAPS_OUT.h5.
    if (options.collecting_stats == False and options.h5_file_name == "STATS_OUT.h5"):
        options.h5_file_name = "MAPS_OUT.h5"

    # data collection
    print("Collecting h5 files...")
    collect_h5(out_dir, options.h5_file_name)

    if options.clean_dir:
        if suspicious_collected_h5_size(out_dir, options.h5_file_name, options.collected_to_sum_file_size_ths):
            raise Exception("Please, check the collected file. Job-files have not been deleted yet since the sum of their sizes is suspiciously bigger than the size of the collected h5 file.")
        else:
            print("Cleaning the directory...")
            clean_directory(out_dir, options.h5_file_name)

if __name__ == "__main__":
    main()
