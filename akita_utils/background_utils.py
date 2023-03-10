import numpy as np
import pandas as pd
import pysam

from .dna_utils import dna_1hot, permute_seq_k
from .stats_utils import insul_diamonds_scores


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def create_flat_seqs_gen(
    seqnn_model,
    genome_fasta,
    dataframe,
    max_iters=10,
    batch_size=6
):
    """This function creates flat sequences by permutating experimental sequences

    Args:
        seqnn_model : model used to make predictions
        genome_fasta(str) : path to fasta file
        dataframe : dataframe of experimental sequences' parameters
        max_iters (int, optional): maximum iterations in making permutations. Defaults to 1.
        batch_size (int, optional): batch size used in model predictions. Defaults to 6.

    Returns:
        flat_seqs : list of flat sequences
    """
    flat_seqs = []
    num_seqs = dataframe.shape[0]
    genome_open = pysam.Fastafile(genome_fasta)
    for ind in range(num_seqs):
        locus_specification, shuffle_k, ctcf_thresh, scores_thresh,scores_pixelwise_thresh = dataframe.iloc[ind][["locus_specification","shuffle_parameter","ctcf_selection_threshold","map_score_threshold",'scores_pixelwise_thresh']]
        chrom, start, end = locus_specification.split(",")
        seq = genome_open.fetch(chrom, int(start), int(end)).upper()
        seq_1hot = dna_1hot(seq)
        num_iters = 0
        while num_iters < max_iters:
            seq_1hot_batch = _seq_batch_generator_shuffled_seqs(seq_1hot, shuffle_k, batch_size)
            predictions = seqnn_model.predict(seq_1hot_batch, batch_size=batch_size)
            scores, scores_pixelwise, custom_scores = _calculate_scores_from_predictions(predictions)

            # log.info(f"\nSCD_scores {scores} - pixelwise {scores_pixelwise} - custom {custom_scores}\n")
            for score_ind,score in enumerate(scores):
                if scores[score_ind]< scores_thresh and scores_pixelwise[score_ind]< scores_pixelwise_thresh and custom_scores[score_ind]>20:
                    num_iters = max_iters
                    best_ind = score_ind
                    best_seq = seq_1hot_batch[best_ind]
                    best_pred = predictions[best_ind]
                    best_score, best_score_pixelwise, best_custom_score = (scores[best_ind],scores_pixelwise[best_ind],custom_scores[best_ind])
                    log.info(f"success: best seq, map score {best_score}, pixelwise {best_score_pixelwise}, custom {best_custom_score}")
                    flat_seqs.append([best_seq,best_pred,best_score,best_score_pixelwise])

            num_iters += 1
            if num_iters == max_iters:
                log.info(f"******* last iteration *******\nscores: {scores} \nscores pixelwise: {scores_pixelwise} \ncustom scores: {custom_scores}")
            
    return flat_seqs


def _seq_batch_generator_shuffled_seqs(seq_1hot, shuffle_k, batch_size):
    seq_1hot_batch = []
    for _ in range(batch_size):
        seq_1hot_batch.append(permute_seq_k(seq_1hot, k=shuffle_k))
    return np.array(seq_1hot_batch)


def _calculate_scores_from_predictions(predictions):
    scores = []
    scores_pixelwise =[]
    custom_score =[]
    mae = []
    for seq_num in range(predictions.shape[0]):    
        ref_preds=predictions[seq_num,:,:]
        scores_pixelwise += [np.max(ref_preds, axis=-1)]
        mae += [np.mean(abs(ref_preds - ref_preds.mean(axis=0)), axis=0)]
        scores += [np.sqrt((ref_preds**2).sum(axis=0))] # SCD
        std = np.std(ref_preds, axis=0)
        mean = np.mean(ref_preds, axis=0)
        custom_score +=  [3/mean + 2/std] # remove this score?!!!  maybe not
        
    return np.max(scores,axis=1), np.max(scores_pixelwise, axis=1), np.min(custom_score,axis=1)
