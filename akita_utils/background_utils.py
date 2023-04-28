import numpy as np
import pandas as pd
import pysam
from .dna_utils import dna_1hot, permute_seq_k
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

        locus_specification, shuffle_k, ctcf_thresh, scores_thresh,scores_pixelwise_thresh = dataframe.iloc[ind][["locus_specification","shuffle_parameter","ctcf_detection_threshold","map_score_threshold",'scores_pixelwise_thresh']]
        chrom, start, end = locus_specification.split(",")
        seq = genome_open.fetch(chrom, int(start), int(end)).upper()
        seq_1hot = dna_1hot(seq)
        num_iters = 0
        while num_iters < max_iters:
            seq_1hot_batch = _seq_batch_generator_shuffled_seqs(seq_1hot, shuffle_k, batch_size)
            predictions = seqnn_model.predict(seq_1hot_batch, batch_size=batch_size)
            num_iters, scores, scores_pixelwise, custom_scores = _check_and_append_satisfying_seqs(seq_1hot_batch=seq_1hot_batch, 
                                                                                                   predictions=predictions, 
                                                                                                   scores_thresh=scores_thresh,
                                                                                                   scores_pixelwise_thresh=scores_pixelwise_thresh, 
                                                                                                   custom_scores_thresh=40, 
                                                                                                   flat_seqs_storage_list=flat_seqs, 
                                                                                                   num_iters=num_iters, 
                                                                                                   max_iters=max_iters
                                                                                                  )

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
    """
    Calculates SCD scores, MPS scores, and custom scores for each sequence in `predictions`.

    Args:
        predictions (numpy.ndarray): A numpy array of shape `(num_sequences, sequence_length, num_classes)`
            containing the predicted probabilities for each class for each position in each sequence.

    Returns:
        tuple: A tuple containing three numpy arrays: `scores` (maximum_of_SCD scores) of shape `(num_sequences,)`,
            `scores_pixelwise` (maximum_of_MPS scores) of shape `(num_sequences,)`, and `custom_scores` (minimum_of_CS scores)
            of shape `(num_sequences,)`.
    """
    scores = []
    scores_pixelwise =[]
    custom_score =[]
    for seq_num in range(predictions.shape[0]):
        ref_preds=predictions[seq_num,:,:]
        std = np.std(ref_preds, axis=0)
        mean = np.mean(ref_preds, axis=0)
        
        scores_pixelwise += [np.max(np.max(np.abs(ref_preds), axis=0))] # maximum_of_MPS
        scores += [np.max(np.sqrt((ref_preds**2).sum(axis=0)))] # maximum_of_SCD
        custom_score +=  [np.min(3/mean + 2/std)] # minimum_of_CS
        
    return scores, scores_pixelwise, custom_score


def _check_and_append_satisfying_seqs(seq_1hot_batch, predictions, scores_thresh, scores_pixelwise_thresh, custom_scores_thresh, flat_seqs_storage_list, num_iters, max_iters):
    """
    Check which sequences in a batch of predictions satisfy certain score thresholds.

    Args:
        seq_1hot_batch (list): A list of one-hot encoded sequences.
        predictions (list): A list of predictions to calculate scores for.
        scores_thresh (float): A threshold for the SCD score.
        scores_pixelwise_thresh (float): A threshold for the pixelwise score (MPS).
        custom_scores_thresh (float): A threshold for a custom score (CS).
        flat_seqs_storage_list (list): A list to store flat seq data in.
        num_iters (int): The current number of iterations.
        max_iters (int): The maximum number of iterations allowed.

    Returns:
        int: The updated number of iterations (max_iters) if the predictions satisfy the conditions, and the satisfying sequences are appended to flat_seqs_storage_list.
        OR,
        int: The current number of iterations (num_iters) if the predictions do not satisfy the conditions.
    """
    scores, scores_pixelwise, custom_scores = _calculate_scores_from_predictions(predictions)

    for score_ind, score in enumerate(scores):
        if scores[score_ind] < scores_thresh and scores_pixelwise[score_ind] < scores_pixelwise_thresh and custom_scores[score_ind] > custom_scores_thresh:
            num_iters = max_iters
            best_ind = score_ind
            best_seq = seq_1hot_batch[best_ind]
            best_pred = predictions[best_ind]
            best_score, best_score_pixelwise, best_custom_score = (scores[best_ind], scores_pixelwise[best_ind], custom_scores[best_ind])
            log.info(f"success: best seq, map score(SCD) {best_score}, pixelwise(MPS) {best_score_pixelwise}, custom(CS) {best_custom_score}")
            flat_seqs_storage_list.append([best_seq, best_pred, best_score, best_score_pixelwise, best_custom_score])

    return num_iters, scores, scores_pixelwise, custom_scores
                   