import akita_utils.dna_utils
import numpy as np

class Locus:
    def __init__(self, allowed_classes):
        self.contents = []
        self.allowed_classes = allowed_classes

    def insert(self, object):
        if isinstance(object, tuple(self.allowed_classes)):
            self.contents.append(object)
        else:
            raise ValueError(f"Invalid object. Only objects of classes {self.allowed_classes} are allowed.")

    def remove(self, object):
        self.contents.remove(object)

    def __str__(self):
        return f"Locus containing: {str(self.contents)}"


class Insertion:
    def __init__(self, name, chrom, start, end, flanks, strand):
        self.name = name
        self.chrom = chrom
        self.start = start
        self.end = end
        self.flanks = flanks
        self.strand = strand

    def replace_flanks(self, other_insertion):
        if isinstance(other_insertion, CTCF):
            self.flanks = other_insertion.flanks
            self.left_flank_chrom = other_insertion.left_flank_chrom
            self.left_flank_start = other_insertion.left_flank_start
            self.left_flank_end = other_insertion.left_flank_end
            self.right_flank_chrom = other_insertion.right_flank_chrom
            self.right_flank_start = other_insertion.right_flank_start
            self.right_flank_end = other_insertion.right_flank_end
        else:
            raise ValueError("Invalid object. Only CTCF objects are allowed.")

    def __str__(self):
        return "Insert: " + str(self.name)


class CTCF(Insertion):
    def __init__(self, name, chrom, start, end, flanks, strand):
        self.name = name
        self.chrom = chrom
        self.start = start
        self.end = end
        self.strand = strand
        self.flanks = flanks
        self.left_flank_length = flanks[0]
        self.right_flank_length = flanks[1]
        self.left_flank_chrom = chrom
        self.left_flank_start = start - flanks[0]
        self.left_flank_end = start - 1
        self.right_flank_chrom = chrom
        self.right_flank_start = end + 1
        self.right_flank_end = end + flanks[1]

    def __str__(self):
        return f"CTCF: {self.name} ({self.chrom}:{self.start}-{self.end})"


class Gene(Insertion):
    def __init__(self, name, chrom, start, end, strand):
        self.name = name
        self.chrom = chrom
        self.start = start
        self.end = end
        self.strand = strand

    def __str__(self):
        return f"Gene: {self.name} ({self.chrom}:{self.start}-{self.end})"


def create_insertions_sequences(locus, genome_open):
    sequences = []
    for element in locus.contents:
        if isinstance(element, CTCF):
            left_flank_sequence = akita_utils.dna_utils.dna_1hot(genome_open[element.left_flank_chrom])
            ctcf_sequence = akita_utils.dna_utils.dna_1hot(genome_open[element.chrom])
            right_flank_sequence = akita_utils.dna_utils.dna_1hot(genome_open[element.right_flank_chrom])
            seq_1hot_insertion = np.concatenate((left_flank_sequence, ctcf_sequence, right_flank_sequence), axis=0)
        else:
            seq_1hot_insertion = akita_utils.dna_utils.dna_1hot(genome_open[element.chrom])
        if element.strand == "-":
            seq_1hot_insertion = akita_utils.dna_utils.hot1_rc(seq_1hot_insertion)
        sequences.append(seq_1hot_insertion)

    return sequences



# def create_insertions_sequences(locus, genome_open):
#     sequences = []
#     for element in locus.contents:
#         if isinstance(element, CTCF):
#             left_flank_sequence = akita_utils.dna_utils.dna_1hot(genome_open.fetch(element.left_flank_chrom,element.left_flank_start-1,element.left_flank_end))
#             ctcf_sequence = akita_utils.dna_utils.dna_1hot(genome_open.fetch(element.chrom,element.start-1,element.end))
#             right_flank_sequence = akita_utils.dna_utils.dna_1hot(genome_open.fetch(element.right_flank_chrom,element.right_flank_start-1,element.right_flank_end))
#             seq_1hot_insertion = np.concatenate((left_flank_sequence, ctcf_sequence, right_flank_sequence), axis=0)
#         else:
#             seq_1hot_insertion = akita_utils.dna_utils.dna_1hot(genome_open.fetch(element.chrom,element.start-1,element.end))
#         if element.strand == "-":
#             seq_1hot_insertion = akita_utils.dna_utils.hot1_rc(seq_1hot_insertion)
#         sequences.append(seq_1hot_insertion)

#     return sequences


# ctcf1 = CTCF("strong_ctcf", "chr1", 100, 200, [10, 15])
# ctcf2 = CTCF("weak_ctcf",  "chr1", 25, 80, [10, 15])
# ctcf1.replace_flanks(ctcf2)

# gene1 = Gene("fenemw", "chr1", 150, 250)

# output_locus = Locus([CTCF, Gene])

# output_locus.insert(gene1)
# output_locus.insert(ctcf1)

# print(output_locus)


# sequences = create_insertions_sequences(output_locus, genome_open)

# print(sequences)
