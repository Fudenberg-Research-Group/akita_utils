#!/usr/bin/env python

# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""
multiGPU-virtual_symmetric_experiment_flanks.py
Derived from akita_motif_scd_multi.py (https://github.com/Fudenberg-Research-Group/akita_utils/blob/main/bin/disrupt_genomic_boundary_ctcfs/akita_motif_scd_multi.py)

Compute scores for motifs in a TSV file, using multiple processes.

Relies on slurm_gf.py to auto-generate slurm jobs.

"""

from optparse import OptionParser

import h5py
import numpy as np

################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <model_file> <tsv_file>"
    parser = OptionParser(usage)

    parser.add_option(
        "-o",
        dest="out_dir",
        default="scd",
        help="Output directory for tables and plots [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    parser.add_option(
        "-f",
        dest="filename",
        default="OUT.h5",
        help="Name of the output files [Default: %default]",
    )
    parser.add_option(
        "-c",
        dest="collected_results_name",
        default="collected.h5",
        help="Name of the files with all the results collected together [Default: %default]",
    )
    (options, args) = parser.parse_args()

    #######################################################
    # collect output

    collect_h5(
        options.collected_results_name,
        options.filename,
        options.out_dir,
        options.processes,
    )


def collect_h5(collected_results_name, file_name, out_dir, num_procs):
    # count variants
    num_variants = 0
    for pi in range(num_procs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, file_name)
        job_h5_open = h5py.File(job_h5_file, "r")
        num_variants += len(job_h5_open["chrom"])
        job_h5_open.close()

    # initialize final h5
    final_h5_file = "%s/%s" % (out_dir, collected_results_name)
    final_h5_open = h5py.File(final_h5_file, "w")

    # keep dict for string values
    final_strings = {}

    job0_h5_file = "%s/job0/%s" % (out_dir, file_name)
    job0_h5_open = h5py.File(job0_h5_file, "r")
    for key in job0_h5_open.keys():

        if key in [
            "experiment_id",
            "chrom",
            "start",
            "end",
            "strand",
            "genomic_SCD",
            "orientation",
            "background_index",
            "flank_bp",
            "spacer_bp",
        ]:
            final_h5_open.create_dataset(
                key, shape=(num_variants,), dtype=job0_h5_open[key].dtype
            )

        elif job0_h5_open[key].dtype.char == "S":
            final_strings[key] = []

        elif job0_h5_open[key].ndim == 1:
            final_h5_open.create_dataset(
                key, shape=(num_variants,), dtype=job0_h5_open[key].dtype
            )

        else:
            num_targets = job0_h5_open[key].shape[1]
            final_h5_open.create_dataset(
                key,
                shape=(num_variants, num_targets),
                dtype=job0_h5_open[key].dtype,
            )

    job0_h5_open.close()

    # set values
    vi = 0
    for pi in range(num_procs):
        print("collecting job", pi)
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, file_name)
        job_h5_open = h5py.File(job_h5_file, "r")

        # append to final
        for key in job_h5_open.keys():

            job_variants = job_h5_open[key].shape[0]

            if key in [
                "experiment_id",
                "chrom",
                "start",
                "end",
                "strand",
                "genomic_SCD",
                "orientation",
                "background_index",
                "flank_bp",
                "spacer_bp",
            ]:
                final_h5_open[key][vi : vi + job_variants] = job_h5_open[key]

            else:
                if job_h5_open[key].dtype.char == "S":
                    final_strings[key] = list(job_h5_open[key])
                else:
                    final_h5_open[key][vi : vi + job_variants] = job_h5_open[
                        key
                    ]

        vi += job_variants
        job_h5_open.close()

    # create final string datasets
    for key in final_strings:
        final_h5_open.create_dataset(
            key, data=np.array(final_strings[key], dtype="S")
        )

    final_h5_open.close()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
