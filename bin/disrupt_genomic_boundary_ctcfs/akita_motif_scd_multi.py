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
This script launches multiple worker threads to perform motif-specific SNP scoring and analysis. It takes a model directory and a TSV file as input. The script prepares the necessary directories and files, launches worker threads, and checks if a job has completed by verifying the existence of the output file. The worker threads execute the "akita_motif_scd.py" script with the provided options and arguments. The script utilizes SLURM for job management and uses GPU resources if available.

"""

from optparse import OptionParser
import os
import pickle
import sys
import akita_utils.slurm_gf as slurm
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def main():
    usage = "usage: %prog [options] <models_dir> <tsv_file>"
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
        default="motif_scd",
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
        dest="scd_stats",
        default="SCD,SSD",
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
    parser.add_option(
        "--head-index",
        dest="head_index",
        default=None,
        type="int",
        help="Specify head index (0=human 1=mus) ",
    )
    parser.add_option(
        "--model-index",
        dest="model_index",
        default=None,
        type="int",
        help="Specify model index between 0 and 7 ",
    )
    parser.add_option(
        "--mut-method",
        dest="mutation_method",
        default="mask_spans",
        type="str",
        help="Specify mutation method, [Default: %default]",
    )
    parser.add_option(
        "--motif-width",
        dest="motif_width",
        default=None,
        type="int",
        help="motif width",
    )
    parser.add_option(
        "--use-span",
        dest="use_span",
        default=False,
        action="store_true",
        help="specify if using spans",
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
        default="motif_del_stats",
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
        default="1-0:0:0",
        help="time to run job. [Default: %default]",
    )
    parser.add_option(
        "--conda_env",
        dest="conda_env",
        default="basenji-gpu",
        help="name of conda environment to run the script",
    )
    parser.add_option(
        "--gres", dest="gres", default="gpu", help="gpu resources. [Default: %default]"
    )
    parser.add_option(
        "--constraint",
        dest="constraint",
        default="[xeon-6130|xeon-2640v4]",
        help="cpu constraints to avoid the a40 gpus. [Default: %default]",
    )

    (options, args) = parser.parse_args()

    if len(args) != 2:
        parser.error("Must provide model dir and TSV file")
    else:
        models_dir = args[0]
        tsv_file = args[1]

        model_dir = models_dir + "/f" + str(options.model_index) + "c0/train/"
        model_file = model_dir + "model" + str(options.head_index) + "_best.h5"
        params_file = model_dir + "params.json"

        new_args = [params_file, model_file, tsv_file]
        options.name = f"{options.name}_m{options.model_index}_h{options.head_index}"

    # output directory
    options.out_dir = f"{options.out_dir}/motif_expt_model{options.model_index}_head{options.head_index}"

    if not options.restart:
        if os.path.isdir(options.out_dir):
            print("Please remove %s" % options.out_dir, file=sys.stderr)
            log.info(f"Please remove {options.out_dir}")
            exit(1)
        os.mkdir(options.out_dir)

    # pickle options
    options_pkl_file = "%s/options.pkl" % options.out_dir
    options_pkl = open(options_pkl_file, "wb")
    pickle.dump(options, options_pkl)
    options_pkl.close()

    # launch worker threads
    jobs = []
    for pi in range(options.processes):
        if not options.restart or not job_completed(options, pi):
            cmd = 'eval "$(conda shell.bash hook)";'
            cmd += f"conda activate {options.conda_env};"
            cmd += "module load gcc/8.3.0; module load cudnn/8.0.4.30-11.0;"
            cmd += " ${SLURM_SUBMIT_DIR}/akita_motif_scd.py %s %s %d" % (
                options_pkl_file,
                " ".join(new_args),
                pi,
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
        jobs, max_proc=options.max_proc, verbose=True, launch_sleep=10, update_sleep=60
    )


def job_completed(options, pi):
    """Check whether a specific job has generated its
    output file."""
    out_file = "%s/job%d/scd.h5" % (options.out_dir, pi)
    return os.path.isfile(out_file) or os.path.isdir(out_file)


if __name__ == "__main__":
    main()
