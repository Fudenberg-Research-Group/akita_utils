import numpy as np
import pandas as pd
import pysam
from akita_utils.dna_utils import dna_1hot, permute_seq_k


def create_flat_seqs_gen(
    seqnn_model,
    genome_fasta,
    dataframe,
    max_iters=25,
    batch_size=8
):
    """This function creates flat sequences by permutating experimental sequences

    Args:
        seqnn_model : model used to make predictions
        genome_fasta(str) : path to fasta file
        dataframe : dataframe of experimental sequences' parameters
        max_iters (int, optional): maximum iterations in making permutations. Defaults to 25.
        batch_size (int, optional): batch size used in model predictions. Defaults to 6.

    Returns:
        flat_seqs : list of flat sequences
    """
    max_iters = int(max_iters) # fix this from above
    flat_seqs = []
    num_seqs = dataframe.shape[0]
    genome_open = pysam.Fastafile(genome_fasta)
    for ind in range(num_seqs):
        print(f"Working on seq number {ind}")
        chrom, start, end, shuffle_k, ctcf_thresh, scores_thresh = dataframe.iloc[ind][["chrom", "start", "end","shuffle_parameter","ctcf_detection_threshold","map_score_threshold"]]
        seq = genome_open.fetch(chrom, int(start), int(end)).upper()
        seq_1hot = dna_1hot(seq)
        num_iters = 0
        while num_iters < max_iters:
            print(f"\t - iteration {num_iters}")
            seq_1hot_batch = _seq_batch_generator_shuffled_seqs(seq_1hot, shuffle_k, batch_size)
            predictions = seqnn_model.predict(seq_1hot_batch, batch_size=batch_size)
            num_iters, scores = _check_and_append_satisfying_seqs(seq_1hot_batch=seq_1hot_batch, 
                                                                                                   predictions=predictions, 
                                                                                                   scores_thresh=scores_thresh,
                                                                                                   flat_seqs_storage_list=flat_seqs, 
                                                                                                   num_iters=num_iters, 
                                                                                                   max_iters=max_iters
                                                                                                  )
            num_iters += 1
            
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
        tuple: A tuple containing three numpy arrays: `scores` (maximum_of_SCD scores) of shape `(num_sequences,)`.
    """
    scores = []
    for seq_num in range(predictions.shape[0]):
        ref_preds=predictions[seq_num,:,:]
        ref_preds = ref_preds.astype("float32")

        # there is no division by 2, since it's calculated based on the prediction vector, not a map
        scores += [np.max(np.sqrt((ref_preds**2).sum(axis=0)))] # maximum_of_SCD
        
    return scores


def _check_and_append_satisfying_seqs(seq_1hot_batch, predictions, scores_thresh, flat_seqs_storage_list, num_iters, max_iters):
    """
    Check which sequences in a batch of predictions satisfy certain score thresholds.

    Args:
        seq_1hot_batch (list): A list of one-hot encoded sequences.
        predictions (list): A list of predictions to calculate scores for.
        scores_thresh (float): A threshold for the SCD score.
        flat_seqs_storage_list (list): A list to store flat seq data in.
        num_iters (int): The current number of iterations.
        max_iters (int): The maximum number of iterations allowed.

    Returns:
        int: The updated number of iterations (max_iters) if the predictions satisfy the conditions, and the satisfying sequences are appended to flat_seqs_storage_list.
        OR,
        int: The current number of iterations (num_iters) if the predictions do not satisfy the conditions.
    """
    scores = _calculate_scores_from_predictions(predictions)

    for score_ind, score in enumerate(scores):
        if scores[score_ind] < scores_thresh:
            num_iters = max_iters
            best_ind = score_ind
            best_seq = seq_1hot_batch[best_ind]
            best_pred = predictions[best_ind]
            best_score = scores[best_ind]
            flat_seqs_storage_list.append([best_seq, best_pred, best_score])

    return num_iters, scores

