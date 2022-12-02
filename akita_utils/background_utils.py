import pandas as pd
import numpy as np
import pysam
import time
from .io import read_jaspar_to_numpy
from .dna_utils import dna_1hot, permute_seq_k, scan_motif

def create_flat_seqs(
    seqnn_model,
    genome_fasta,
    seq_length,
    dataframe,
    max_iters=1,
    batch_size=6,
    shuffle_k=8,
    ctcf_thresh=8,
    scores_thresh=5500,
    scores_pixelwise_thresh=0.04,
    masking=False
):
    """This function creates flat sequences by permutating experimental sequences

    Args:
        seqnn_model : model used to make predictions
        genome_fasta : _description_
        seq_length (int): length of the sequence
        dataframe : dataframe of experimental sequences
        max_iters (int, optional): maximum iterations in making permutations. Defaults to 1.
        batch_size (int, optional): batch size used in model predictions. Defaults to 6.
        shuffle_k (int, optional): basepairs to shuffle while permutating. Defaults to 8.
        ctcf_thresh (int, optional): minimum number of ctcf motiifs allowed in a flat sequence. Defaults to 8.
        scores_thresh (int, optional): score used to determine how much structure exists in the output. Defaults to 5500.
        scores_pixelwise_thresh (float, optional): pixelwise score to determine structure in output. Defaults to 0.04.
        masking (bool, optional): Option to determine whether to mask or not before permutation. Defaults to False.

    Returns:
        flat_seqs : list of flat sequences
    """
    flat_seqs = []
    num_seqs = dataframe.shape[0]

    # mot = ">CCAsyAGrkGGCr\n0.0000\t1.0000\t0.0000\t0.0000\n0.0000\t1.0000\t0.0000\t0.0000\n1.0000\t0.0000\t0.0000\t0.0000\n0.0000\t0.5000\t0.5000\t0.0000\n0.0000\t0.5000\t0.0000\t0.5000\n1.0000\t0.0000\t0.0000\t0.0000\n0.0000\t0.0000\t1.0000\t0.0000\n0.5000\t0.0000\t0.5000\t0.0000\n0.0000\t0.0000\t0.5000\t0.5000\n0.0000\t0.0000\t1.0000\t0.0000\n0.0000\t0.0000\t1.0000\t0.0000\n0.0000\t1.0000\t0.0000\t0.0000\n0.5000\t0.0000\t0.5000\t0.0000"
    # motif = pd.read_csv(
    #     StringIO(mot), sep="\t", header=0, names=["A", "C", "G", "T"]
    # ).values
    # motif_window = int(np.ceil(len(motif) / 2))

    motif = read_jaspar_to_numpy()
    motif_window = int(np.ceil(len(motif) / 2))

    CTCF_MOTIF_SHUF = np.array([12, 0, 1, 11, 10, 3, 2, 8, 9, 4, 5, 7, 6]) # manual shuffle procedure 

    for ind in range(num_seqs):
        try:
            chrom, start, end, gc = dataframe.iloc[ind][["chrom", "start", "end", "GC"]]
        except:
            chrom, start, end, strand = dataframe.iloc[ind][["chrom", "start", "end", "strand"]]
            print('The dataframe used doesnot have GC content')
            
        genome_open = pysam.Fastafile(genome_fasta)
        seq = genome_open.fetch(chrom, start, end).upper()
        seq_1hot = dna_1hot(seq)

        t0 = time.time()
        num_iters = 0
        while num_iters < max_iters:
            print("ind", ind, ", iter ", num_iters, ", for", chrom, start, end)
            # print(len(flat_seqs))

            seq_1hot_batch = []
            for i in range(batch_size):
                seq_1hot_mut = permute_seq_k(seq_1hot, k=shuffle_k)
                if masking is True:
                    s = scan_motif(seq_1hot_mut, motif)
                    for i in np.where(s > ctcf_thresh)[0]:
                        # seq_1hot_mut[i-motif_window:i+motif_window] = permute_seq_k(seq_1hot_mut[i-motif_window:i+motif_window], k=2)
                        seq_1hot_mut[
                            i - motif_window + 1 : i + motif_window
                        ] = seq_1hot_mut[i - motif_window + 1 : i + motif_window][CTCF_MOTIF_SHUF]
                    seq_1hot_batch.append(seq_1hot_mut)
                else:
                    seq_1hot_batch.append(seq_1hot_mut)
            seq_1hot_batch = np.array(seq_1hot_batch)

            pred = seqnn_model.predict(seq_1hot_batch, batch_size=batch_size)
            scores = np.sum(pred**2, axis=-1).sum(axis=-1)
            scores_pixelwise = np.max(pred**2, axis=-1).max(axis=-1)
            t1 = time.time()
            
            if np.any([
                    (np.min(scores) < scores_thresh)
                    , (np.min(scores_pixelwise) < scores_pixelwise_thresh)]
            ):
                
                best_ind = np.argmin(scores_pixelwise)
                best_seq = seq_1hot_batch[best_ind]
                best_pred = pred[best_ind]
                best_score, best_score_pixelwise = (
                    scores[best_ind],
                    scores_pixelwise[best_ind],
                )
                num_iters = max_iters
                print(
                    "success: best seq, thresh",
                    np.min(scores),
                    " pixelwise",
                    np.min(scores_pixelwise),
                    "time",
                    t1 - t0
                )

            num_iters += 1
            if num_iters >= max_iters:
                print(f"max iters exceeded, final time {t1 - t0}")
                
                if gc :
                    flat_seqs.append([
                    best_seq,
                    best_pred,
                    best_score,
                    best_score_pixelwise,
                    t1 - t0,
                    gc
                ])
                else:
                    flat_seqs.append([
                    best_seq,
                    best_pred,
                    best_score,
                    best_score_pixelwise,
                    t1 - t0
                ])
                # raise ValueError('cannot generate flat sequence for', chrom, start, end)

    return flat_seqs


def custom_calculate_scores(    seqnn_model,
                                genome_fasta,
                                seq_length,
                                dataframe,
                                max_iters=1,
                                batch_size=6,
                                shuffle_k=8,
                                ctcf_thresh=8,
                                scores_thresh=5500,
                                scores_pixelwise_thresh=0.04,
                                success_scores = 0,
                                masking=False,
                            ):
    """_summary_

    Args:
        seqnn_model : model used to make predictions
        genome_fasta : _description_
        seq_length (int): length of the sequence
        dataframe : dataframe of experimental sequences
        max_iters (int, optional): maximum iterations in making permutations. Defaults to 1.
        batch_size (int, optional): batch size used in model predictions. Defaults to 6.
        shuffle_k (int, optional): basepairs to shuffle while permutating. Defaults to 8.
        ctcf_thresh (int, optional): minimum number of ctcf motiifs allowed in a flat sequence. Defaults to 8.
        scores_thresh (int, optional): score used to determine how much structure exists in the output. Defaults to 5500.
        scores_pixelwise_thresh (float, optional): pixelwise score to determine structure in output. Defaults to 0.04.
        success_scores (int, optional): _description_. Defaults to 0.
        masking (bool, optional): Option to determine whether to mask or not before permutation. Defaults to False.

    Returns:
        scores_set : list containing scores
    """
    scores_set = []
    num_seqs = dataframe.shape[0]

    # mot = ">CCAsyAGrkGGCr\n0.0000\t1.0000\t0.0000\t0.0000\n0.0000\t1.0000\t0.0000\t0.0000\n1.0000\t0.0000\t0.0000\t0.0000\n0.0000\t0.5000\t0.5000\t0.0000\n0.0000\t0.5000\t0.0000\t0.5000\n1.0000\t0.0000\t0.0000\t0.0000\n0.0000\t0.0000\t1.0000\t0.0000\n0.5000\t0.0000\t0.5000\t0.0000\n0.0000\t0.0000\t0.5000\t0.5000\n0.0000\t0.0000\t1.0000\t0.0000\n0.0000\t0.0000\t1.0000\t0.0000\n0.0000\t1.0000\t0.0000\t0.0000\n0.5000\t0.0000\t0.5000\t0.0000"
    # motif = pd.read_csv(StringIO(mot), sep="\t", header=0, names=["A", "C", "G", "T"]).values
    
    motif = read_jaspar_to_numpy()
    motif_window = int(np.ceil(len(motif) / 2))
    CTCF_MOTIF_SHUF = np.array([12, 0, 1, 11, 10, 3, 2, 8, 9, 4, 5, 7, 6, 13])

    for ind in range(num_seqs):
        chrom, start, end, gc = dataframe.iloc[ind][["chrom", "start", "end", "GC"]]
        genome_open = pysam.Fastafile(genome_fasta)
        seq = genome_open.fetch(chrom, start, end).upper()
        seq_1hot = dna_1hot(seq)

        num_iters = 0
        while num_iters < max_iters:
            print("ind",ind,", iter ",num_iters,",k ",shuffle_k,", for", chrom, start, end,)
            seq_1hot_batch = []
            for i in range(batch_size):
                seq_1hot_mut = permute_seq_k(seq_1hot, k=shuffle_k)
                if masking == True:
                    s = scan_motif(seq_1hot_mut, motif)
                    for i in np.where(s > ctcf_thresh)[0]:
                        if len(seq_1hot_mut[i-motif_window:i+motif_window]) == len(CTCF_MOTIF_SHUF):
                            seq_1hot_mut[i-motif_window:i+motif_window] = permute_seq_k(seq_1hot_mut[i-motif_window:i+motif_window], k=2)
                        
                        # seq_1hot_mut[i-motif_window:i+motif_window] = seq_1hot_mut[i-motif_window:i+motif_window][CTCF_MOTIF_SHUF]
                    seq_1hot_batch.append(seq_1hot_mut)
                else:    
                    seq_1hot_batch.append(seq_1hot_mut)

            seq_1hot_batch = np.array(seq_1hot_batch)
            pred = seqnn_model.predict(seq_1hot_batch, batch_size=batch_size)
            scores = np.sum(pred**2, axis=-1).sum(axis=-1)
            scores_pixelwise = np.max(pred**2, axis=-1).max(axis=-1)
            
            if success_scores == 1:
                if np.any([
                    (np.min(scores) < scores_thresh)
                    , (np.min(scores_pixelwise) < scores_pixelwise_thresh)]
                    ):
                    print(
                        "*** success: best seq, thresh",
                        np.min(scores),
                        " pixelwise",
                        np.min(scores_pixelwise),
                        "***",
                    )
            else:
                    scores_set += [scores]
                
            num_iters += 1
            if num_iters >= max_iters:
                print("max iters exceeded")

    return scores_set
