import numpy as np
import pysam
from .dna_utils import dna_1hot, permute_seq_k


def create_flat_seqs_gen(
    seqnn_model,
    genome_fasta,
    dataframe,
    max_iters=10,
    batch_size=6,
):
    """This function creates flat sequences by permutating experimental sequences

    Args:
        seqnn_model : model used to make predictions
        genome_fasta : _description_
        dataframe : dataframe of experimental sequences
        max_iters (int, optional): maximum iterations in making permutations. Defaults to 1.
        batch_size (int, optional): batch size used in model predictions. Defaults to 6.

    Returns:
        flat_seqs : list of flat sequences
    """
    flat_seqs = []
    num_seqs = dataframe.shape[0]
    genome_open = pysam.Fastafile(genome_fasta)
    for ind in range(num_seqs):
        (
            locus_specification,
            shuffle_k,
            ctcf_thresh,
            scores_thresh,
            scores_pixelwise_thresh,
        ) = dataframe.iloc[ind][
            [
                "locus_specification",
                "shuffle_parameter",
                "ctcf_detection_threshold",
                "map_score_threshold",
                "scores_pixelwise_thresh",
            ]
        ]
        chrom, start, end = locus_specification.split(",")
        seq = genome_open.fetch(chrom, int(start), int(end)).upper()
        seq_1hot = dna_1hot(seq)
        num_iters = 0
        while num_iters < max_iters:
            seq_1hot_batch = _seq_batch_generator_flat_maps(
                seq_1hot, shuffle_k, batch_size
            )
            pred = seqnn_model.predict(seq_1hot_batch, batch_size=batch_size)
            scores = np.sum(pred**2, axis=-1).sum(axis=-1)
            scores_pixelwise = np.max(pred**2, axis=-1).max(axis=-1)

            if np.all(
                [
                    (np.min(scores) < scores_thresh),
                    (np.min(scores_pixelwise) < scores_pixelwise_thresh),
                ]
            ):
                num_iters = max_iters
                best_ind = np.argmin(scores_pixelwise)
                best_seq = seq_1hot_batch[best_ind]
                best_pred = pred[best_ind]
                best_score, best_score_pixelwise = (
                    scores[best_ind],
                    scores_pixelwise[best_ind],
                )
                print(
                    "success: best seq, thresh",
                    np.min(scores),
                    " pixelwise",
                    np.min(scores_pixelwise),
                )
                flat_seqs.append(
                    [best_seq, best_pred, best_score, best_score_pixelwise]
                )
            num_iters += 1

    return flat_seqs


def _seq_batch_generator_flat_maps(seq_1hot, shuffle_k, batch_size):
    seq_1hot_batch = []
    for i in range(batch_size):
        seq_1hot_batch.append(permute_seq_k(seq_1hot, k=shuffle_k))
    return np.array(seq_1hot_batch)
