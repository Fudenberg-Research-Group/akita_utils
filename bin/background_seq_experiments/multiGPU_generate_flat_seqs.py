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
multiGPU_generate_flat_seqs.py

generate flat seqs following parameters in given TSV file, using multiple processes.
Relies on slurm_gf.py to auto-generate slurm jobs.

"""

from optparse import OptionParser
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pickle
import subprocess
import sys

import h5py
import numpy as np
import akita_utils.slurm_gf as slurm

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <models_dir> <tsv_file>"
    parser = OptionParser(usage)

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
        default=".",
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
        default=0,
        type="int",
        help="Specify model index (from 0 to 7)",
    )
    # parser.add_option('--flat_seq_tsv_file', dest='flat_seq_tsv_file',
    #   default='/home1/kamulege/akita_utils/bin/background_seq_experiments/data/flat_seqs.tsv',           
    #   help='flat_seq tsv file')
    
    parser.add_option('-s', dest='save_seqs',
      default=None,
      help='Save the final seqs in fasta format')  
    
    parser.add_option('--max_iters', dest='max_iters',
      default=10,
      type="int",
      help='maximum iterations')
    
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
        default="flat",
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
        print(args)
        parser.error("Must provide models directory and TSV file")
    else:
        models_dir = args[0]
        tsv_file = args[1]
        
        
        model_dir = models_dir+"/f"+str(options.model_index)+"c0/train/"
        model_file = model_dir+'model'+str(options.head_index)+'_best.h5'
        params_file = model_dir+"params.json" 
        
        new_args = [params_file,model_file,tsv_file]
        options.name = f"{options.name}_m{options.model_index}"

    #######################################################
    # prep work
    
    # output directory
    
    options.out_dir = f"{options.out_dir}/flat_seqs_model{options.model_index}_head{options.head_index}"
    
    if not options.restart:
        if os.path.isdir(options.out_dir):
            log.info(f"File {options.out_dir} already exists, please remove it")
            print(f"File {options.out_dir} already exists, please remove it")
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
        if not options.restart or not job_completed(options, pi):
            if options.cpu:
                cmd = 'eval "$(conda shell.bash hook)";'
                cmd += "conda activate basenji-gpu;"
                cmd += "module load gcc/8.3.0; module load cudnn/8.0.4.30-11.0;"
            else:
                cmd = 'eval "$(conda shell.bash hook)";'
                cmd += "conda activate basenji-gpu;"      #changed
                # cmd += "conda activate basenji;"      #changed
                cmd += "module load gcc/8.3.0; module load cudnn/8.0.4.30-11.0;"

            cmd += " ${SLURM_SUBMIT_DIR}/generate_flat_seqs.py %s %s %d" % (
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
                mem=25000,
                time=options.time,
                cpu=options.num_cpus,
                constraint=options.constraint,
            )
            jobs.append(j)
    
    slurm.multi_run(
        jobs, max_proc=options.max_proc, verbose=False, launch_sleep=10, update_sleep=60
    )


def job_completed(options, pi):
    """Check whether a specific job has generated its
    output file."""
    out_file = "%s/job%d/scd.h5" % (options.out_dir, pi)
    return os.path.isfile(out_file) or os.path.isdir(out_file)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()