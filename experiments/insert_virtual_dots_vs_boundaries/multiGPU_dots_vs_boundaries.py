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
import os

import pickle

import akita_utils.slurm_gf as slurm
from akita_utils.h5_utils import job_started

################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <model_file> <tsv_file>"
    parser = OptionParser(usage)

    # scd
    parser.add_option(
        "-f",
        dest="genome_fasta",
        default="%s/data/hg19.fa" % os.environ["BASENJIDIR"],
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-m",
        dest="plot_map",
        default=False,
        action="store_true",
        help="Plot contact map for each allele [Default: %default]",
    )
    parser.add_option(
        "-l",
        dest="plot_lim_min",
        default=0.1,
        type="float",
        help="Heatmap plot limit [Default: %default]",
    )
    parser.add_option(
        "--plot-freq",
        dest="plot_freq",
        default=100,
        type="int",
        help="Heatmap plot freq [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="scd",
        help="Output directory for tables and plots [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--stats",
        dest="stats",
        default="SCD",
        type="str",
        help="Comma-separated list of stats to save. [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "--batch-size",
        dest="batch_size",
        default=None,
        type="int",
        help="Specify batch size",
    )

    # insertion-specific options
    parser.add_option(
        "--background-file",
        dest="background_file",
        default="/project/fudenber_735/tensorflow_models/akita/v2/analysis/background_seqs.fa",
        help="file with insertion seqs in fasta format",
    )
    parser.add_option(
        "--save-maps",
        dest="save_maps",
        default=False,
        action="store_true",
        help="Save all the maps in the h5 file(for all inserts, all backgrounds used, and all targets)",
    )

    # multi
    parser.add_option(
        "--cpu",
        dest="cpu",
        default=False,
        action="store_true",
        help="Run without a GPU [Default: %default]",
    )
    parser.add_option(
        "--num_cpus",
        dest="num_cpus",
        default=2,
        type="int",
        help="Number of cpus [Default: %default]",
    )
    parser.add_option(
        "--name",
        dest="name",
        default="exp",
        help="SLURM name prefix [Default: %default]",
    )
    parser.add_option(
        "--max_proc",
        dest="max_proc",
        default=None,
        type="int",
        help="Maximum concurrent processes [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    parser.add_option(
        "-q",
        dest="queue",
        default="gpu",
        help="SLURM queue on which to run the jobs [Default: %default]",
    )
    parser.add_option(
        "-r",
        dest="restart",
        default=False,
        action="store_true",
        help="Restart a partially completed job [Default: %default]",
    )
    parser.add_option(
        "--time",
        dest="time",
        default="01:00:00",
        help="time to run job. [Default: %default]",
    )
    parser.add_option(
        "--gres",
        dest="gres",
        default="gpu",
        help="gpu resources. [Default: %default]",
    )
    parser.add_option(
        "--constraint",
        dest="constraint",
        default="[xeon-6130|xeon-2640v4]",
        help="cpu constraints to avoid the a40 gpus. [Default: %default]",
    )

    (options, args) = parser.parse_args()

    if len(args) != 3:
        parser.error("Must provide parameters and model files and TSV file")
    else:
        params_file = args[0]
        model_file = args[1]
        tsv_file = args[2]

    #######################################################
    # prep work

    # output directory
    if not options.restart:
        if os.path.isdir(options.out_dir):
            # print("Please remove %s" % options.out_dir, file=sys.stderr)
            exit(1)
        os.mkdir(options.out_dir)

    # pickle options
    options_pkl_file = "%s/options.pkl" % options.out_dir
    options_pkl = open(options_pkl_file, "wb")
    pickle.dump(options, options_pkl)
    options_pkl.close()

    #######################################################
    # launch worker threads
    jobs = []
    for pi in range(options.processes):
        if not options.restart or not job_started(options, pi):
            if options.cpu:
                cmd = 'eval "$(conda shell.bash hook)";'
                cmd += "conda activate basenji;"
                cmd += (
                    "module load gcc/8.3.0; module load cudnn/8.0.4.30-11.0;"
                )
            else:
                cmd = 'eval "$(conda shell.bash hook)";'
                cmd += "conda activate basenji-numba2;"
                cmd += (
                    "module load gcc/8.3.0; module load cudnn/8.0.4.30-11.0;"
                )

            cmd += (
                " ${SLURM_SUBMIT_DIR}/virtual_symmetric_experiment_dots_vs_boundaries.py %s %s %d"
                % (
                    options_pkl_file,
                    " ".join(args),
                    pi,
                )
            )

            name = "%s_p%d" % (options.name, pi)
            outf = "%s/job%d.out" % (options.out_dir, pi)
            errf = "%s/job%d.err" % (options.out_dir, pi)

            num_gpu = 1 * (not options.cpu)

            j = slurm.Job(
                cmd,
                name,
                outf,
                errf,
                queue=options.queue,
                gpu=num_gpu,
                gres=options.gres,
                mem=15000,
                time=options.time,
                cpu=options.num_cpus,
                constraint=options.constraint,
            )
            jobs.append(j)

    slurm.multi_run(
        jobs,
        max_proc=options.max_proc,
        verbose=False,
        launch_sleep=10,
        update_sleep=60,
    )


################################################################################
# __main__
################################################################################

if __name__ == "__main__":
    main()
