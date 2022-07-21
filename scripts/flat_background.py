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
from __future__ import print_function

from optparse import OptionParser
import json
import os
import pdb
import pickle
import random
import sys
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pysam
from skimage.measure import block_reduce
import seaborn as sns
sns.set(style='ticks', font_scale=1.3)

import tensorflow as tf
if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#  tf.config.experimental.set_memory_growth(gpu, True)
print(gpus)

from basenji import seqnn
from basenji import stream
from basenji import dna_io

#from basenji import vcf as bvcf

import akita_utils
from io import StringIO
'''
creating flat maps/seqs

'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
  parser = OptionParser(usage)
  parser.add_option('-f', dest='genome_fasta',
      default=None,
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-l', dest='plot_lim_min',
      default=0.1, type='float',
      help='Heatmap plot limit [Default: %default]')
  parser.add_option('--plot-freq', dest='plot_freq',
      default=100, type='int',
      help='Heatmap plot freq [Default: %default]')
  parser.add_option('-m', dest='plot_map',
      default=False, action='store_true',
      help='Plot contact map for each allele [Default: %default]')
  parser.add_option('-o',dest='out_dir',
      default='scd',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--stats', dest='scd_stats',
      default='SCD',
      help='Comma-separated list of stats to save. [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('--batch-size', dest='batch_size',
      default=None, type='int',
      help='Specify batch size')
  parser.add_option('--head-index', dest='head_index',
      default=0, type='int',
      help='Specify head index (0=human 1=mus) ')

  parser.add_option('--mut-method', dest='mutation_method',
      default='mask', type='str',
      help='specify mutation method')

  parser.add_option('--use-span', dest='use_span',
      default=False,  action='store_true',
      help='specify if using spans')

  parser.add_option('-s', dest='save_seqs',
      default=None,
      help='Save the final seqs in fasta format')  
    
  parser.add_option('--motif-width', dest='motif_width',
      default=18, type='int',
      help='motif width')

  parser.add_option('--h5', dest='h5_dirs',
      default=None,              
      help='h5_dirs')

  parser.add_option('--shuffle_k', dest='shuffle_k',
      default=8, type='int',
      help='basepairs considered for shuffling')

  parser.add_option('--ctcf-thresh', dest='ctcf_thresh',
      default=8, type='int',
      help='maximum alowable ctcf motifs in a flat seq')

  parser.add_option('--scores-thresh', dest='scores_thresh',
      default=5500, type='int',
      help='maximum alowable score for a flat seq')

  parser.add_option('--scores-pixelwise-thresh', dest='scores_pixelwise_thresh',
      default=0.04, type='float',
      help='maximum alowable pixel score for a flat seq')
    
  parser.add_option('--chrom-data', dest='chrom_data',
      default=None,              
      help='crom_data_directory')

  (options, args) = parser.parse_args()

  if len(args) == 3:
    # single worker
    params_file = args[0]
    model_file = args[1]
    motif_file = args[2]

  elif len(args) == 5:
    # multi worker
    options_pkl_file = args[0]
    params_file = args[1]
    model_file = args[2]
    motif_file = args[3]
    worker_index = int(args[4])

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

    # update output directory
    options.out_dir = '%s/job%d' % (options.out_dir, worker_index)

  else:
    parser.error('Must provide parameters and model files and QTL VCF file')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)
  if options.plot_map:
    plot_dir = options.out_dir
  else:
    plot_dir = None

  options.shifts = [int(shift) for shift in options.shifts.split(',')]
  options.scd_stats = options.scd_stats.split(',')

  random.seed(44)

  #################################################################
  # read parameters and targets

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_train = params['train']
  params_model = params['model']

  if options.batch_size is None:
    batch_size = params_train['batch_size']
  else: 
    batch_size = options.batch_size
  print(batch_size)
  mutation_method = options.mutation_method
  # if not mutation_method in ['mask','permute']:
  #   raise ValueError('undefined mutation method:', mutation_method)
  motif_width = options.motif_width
  use_span = options.use_span
  if options.use_span:
    print('using SPANS')

  if options.targets_file is not None:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
    target_ids = targets_df.identifier
    target_labels = targets_df.description

  #################################################################
  # setup model

  # load model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file, head_i = options.head_index)
  seqnn_model.build_ensemble(options.rc, options.shifts)
  seq_length = int(params_model['seq_length'])

  # dummy target info
  if options.targets_file is None:
    num_targets = seqnn_model.num_targets()
    target_ids = ['t%d' % ti for ti in range(num_targets)]
    target_labels = ['']*len(target_ids)

  #################################################################
  # fetching chromosome data

  chromsizes = bioframe.read_chromsizes(options.chrom_data)
  dframe = pd.DataFrame(chromsizes)
  dframe['end'] = dframe['length']+ 1310720
  dframe = dframe.reset_index()
  dframe.rename(columns = {'index' : 'chrom', 'length':'start'}, inplace = True)
  df = bioframe.frac_gc(dframe, bioframe.load_fasta(options.genome_fasta), return_input=True)

  #################################################################
  # Generating a sample for down stream analysis

  super_set = []
  error = 0.01

  for gc in np.percentile(df['GC'].dropna().values, np.linspace(1,99,50)):
    for i in range(df.shape[0]):
        if gc-error <= df['GC'].values[i] <= gc+error:
            super_set += [i]
            break

  super_set = list(set(super_set)) 
  sample_set = super_set[2:3]
  new_dataframe = df.iloc[[ind for ind in set(sample_set)]]
  #################################################################

  # create flat sequences
    
  flat_seqs = akita_utils.create_flat_seqs(seqnn_model=seqnn_model, 
                                        genome_fasta=options.genome_fasta, 
                                        seq_length=seq_length, 
                                        dataframe=new_dataframe, 
                                        max_iters = max_iters, 
                                        batch_size = batch_size, 
                                        shuffle_k = options.shuffle_k, 
                                        ctcf_thresh = options.ctcf_thresh, 
                                        scores_thresh = options.scores_thresh, 
                                        scores_pixelwise_thresh = options.scores_pixelwise_thresh)    
    
    
  #################################################################
  # save flat sequences in fasta format if requested
    
  if options.save_seqs is not None:  
      with open(f'{options.out_dir}/backround_seqs.fa','w') as f:
        for i in range(len(flat_seqs)):
            f.write('>shuffled_chr'+str(i)+'_score'+str(int(flat_seqs[i][2]))+'_pixelwise'+str(int(flat_seqs[i][3]*1000))+'\n')
            f.write(dna_io.hot1_dna(flat_seqs[i][0])+'\n')  

    
  #################################################################
  # plot flat sequences
    
  try:
    target_crop = params_model['trunk'][-2]['cropping']
  except:
    target_crop = params_model['target_crop']
  
  target_map_size = seq_length//2048  - target_crop*2 
  hic_diags = params_model['diagonal_offset']  

  if plot_dir is not None:
    target_ind = 0
    vlim = 1.5

    for i in range(len(flat_seqs)):
        flat_pred = flat_seqs[i][1]

        plt.subplot(3,4, i+1)
        im = plt.matshow(
                akita_utils.from_upper_triu(  flat_pred[:,target_ind], target_map_size,hic_diags),
                 vmin=-1*vlim,vmax=vlim, fignum=False,cmap='RdBu_r')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('tot '+
                str(np.round(flat_seqs[i][2],0).astype(int))+'\n pixel '+
                str(np.round(flat_seqs[i][3],4)) ) 

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/flat seqs.pdf')
    plt.close()
    

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()