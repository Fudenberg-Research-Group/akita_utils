### akita utilities 

import bioframe 
import pandas as pd
import numpy as np 
import tensorflow as tf
from basenji import dna_io


### numeric utilites

from scipy.stats import spearmanr, pearsonr
def absmaxND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)

import scipy.signal
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def set_diag(arr, x, i=0, copy=False):
    if copy:
        arr = arr.copy()
    start = max(i, -arr.shape[1] * i)
    stop = max(0, (arr.shape[1] - i)) * arr.shape[1]
    step = arr.shape[1] + 1
    arr.flat[start:stop:step] = x
    return arr

### model i/o 
def from_upper_triu(vector_repr, matrix_len, num_diags):
    z = np.zeros((matrix_len,matrix_len))
    triu_tup = np.triu_indices(matrix_len,num_diags)
    z[triu_tup] = vector_repr
    for i in range(-num_diags+1,num_diags):
        set_diag(z, np.nan, i)
    return z + z.T


### score i/o
import h5py

def h5_to_df(filename):
    scd_out = h5py.File(filename,'r')
    s = []
    scd_stats = ['SCD','SSD','INS-16','INS-32','INS-64','INS-128','INS-256']
    for key in scd_out.keys():
        if key.replace('ref_','').replace('alt_','') in scd_stats:
            s.append(pd.Series(scd_out[key][()].mean(axis=1) , name=key) ) 
        else:
            s.append(pd.Series(scd_out[key][()],name=key)  )
            #print(len(scd_out[key][()]))

    ins_stats= ['INS-16','INS-32','INS-64','INS-128','INS-256']
    for key in ins_stats: 
        if "ref_"+key in scd_out.keys():
            diff = scd_out["ref_"+key][()].mean(axis=1)- scd_out["alt_"+key][()].mean(axis=1)
            s.append( pd.Series(diff, name=key) ) 

    seq_coords_df = pd.concat(s,axis=1)
    for key in ['chrom','strand_2']:#'rownames','strand','chrom','TF']:
        seq_coords_df[key] = seq_coords_df[key].str.decode('utf8').copy()
    scd_out.close()
    
    len_orig = len(seq_coords_df)
    seq_coords_df.drop_duplicates('index',inplace=True)
    print('orig', len_orig, 'filt', len(seq_coords_df))
    print(len_orig-len(seq_coords_df), 'duplicates removed')
    seq_coords_df.rename(columns={'index':'mut_index'}, inplace=True)
    seq_coords_df.reset_index(inplace=True, drop=True)
    return seq_coords_df



### sequence handling

def dna_rc(seq):
  return seq.translate(str.maketrans("ATCGatcg","TAGCtagc"))[::-1]


def permute_seq_k(seq_1hot, k=2):
    if np.mod(k,2) != 0: raise ValueError('current implementation only works for multiples of 2')
    seq_1hot_perm = np.zeros(np.shape(seq_1hot)).astype(int)
    perm_inds = k*np.random.permutation( np.arange(len(seq_1hot)// k ))
    for i in range(k):
        seq_1hot_perm[i::k]= seq_1hot[perm_inds+i,:].copy()
    return seq_1hot_perm



### motif handling 
def scan_motif(seq_1hot, motif , strand=None):
    if motif.shape[-1] != 4: raise ValueError("motif should be n_postions x 4 bases, A=0, C=1, G=2, T=3")
    if seq_1hot.shape[-1] != 4: raise ValueError("seq_1hot should be n_postions x 4 bases, A=0, C=1, G=2, T=3")
    scan_forward = tf.nn.conv1d(  
                         np.expand_dims(seq_1hot,0).astype(float),
                         np.expand_dims(motif,-1).astype(float),
                         stride=1, padding='SAME').numpy()[0]
    if strand == 'forward':
        return scan_forward
    scan_reverse = tf.nn.conv1d(  
                         np.expand_dims(seq_1hot,0).astype(float),
                         np.expand_dims(dna_io.hot1_rc(motif),-1).astype(float),
                         stride=1, padding='SAME').numpy()[0]
    if strand == 'reverse':
        return scan_reverse
    return np.maximum(scan_forward,scan_reverse).flatten()

def read_jaspar_to_numpy(motif_file, normalize=True):
    ## read jaspar pfm 
    with open(motif_file,'r') as f:
        motif = []
        for line in f.readlines():
            if '>' in line: 
                continue
            else: 
                motif.append(line.strip().replace('[','').replace(']','').split())
    motif = pd.DataFrame(motif).set_index(0).astype(float).values.T
    if normalize==True:
        motif /= motif.sum(axis=1)[:,None]
    if motif.shape[1] !=4: raise ValueError('motif returned should be have n_positions x 4 bases')
    return motif

