import akita_utils.dna_utils
import numpy as np


import pysam
def genome_open():
    return pysam.Fastafile("/project/fudenber_735/genomes/mm10/mm10.fa") 


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class LOCUS:
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
    def __init__(self, name, chrom, start, end, strand):
        self.name = name
        self.chrom = chrom
        self.start = start
        self.end = end
        self.strand = strand

    def replace_flanks(self, other_insertion):
        if isinstance(other_insertion, INSERT):
            self.flanks = other_insertion.flanks
            self.left_flank_chrom = other_insertion.left_flank_chrom
            self.left_flank_start = other_insertion.left_flank_start
            self.left_flank_end = other_insertion.left_flank_end
            self.right_flank_chrom = other_insertion.right_flank_chrom
            self.right_flank_start = other_insertion.right_flank_start
            self.right_flank_end = other_insertion.right_flank_end
        else:
            raise ValueError("Invalid object. Only INSERT objects are allowed.")

    def __str__(self):
        return "Insert: " + str(self.name)


class INSERT(Insertion):
    def __init__(self, name, chrom, start, end, flanks, strand):
        super().__init__(name, chrom, start, end, strand)
        self.flanks = flanks
        self.left_flank_length = flanks[0]
        self.right_flank_length = flanks[1]
        self.left_flank_chrom = chrom
        self.right_flank_chrom = chrom
        self.left_flank_start = None
        self.left_flank_end = None
        self.right_flank_start = None
        self.right_flank_end = None
        
        if self.left_flank_length not in [0,1] and self.right_flank_length not in [0,1]:
            self.left_flank_start = start - flanks[0]
            self.left_flank_end = start - 1
            self.right_flank_start = end + 1
            self.right_flank_end = end + flanks[1]
        
        if self.right_flank_length == 1:
            self.right_flank_start = self.right_flank_end = end + 1
            
        if self.left_flank_length == 1:
            self.left_flank_start = self.left_flank_end = start - 1
 

    def __str__(self):
        return f"INSERT: {self.name} {self.chrom}:{self.start}-{self.end} \n {self.flanks} \n {self.left_flank_length} \n {self.right_flank_length} \n {self.left_flank_chrom} \n {self.right_flank_chrom} \n {self.left_flank_start} \n {self.left_flank_end} \n {self.right_flank_start} \n {self.right_flank_end}"



def create_insertions_sequences(locus, genome_open):
    sequences = []
    for element in locus.contents:
        if element.left_flank_start : 
            left_flank_sequence = akita_utils.dna_utils.dna_1hot(genome_open.fetch(element.left_flank_chrom,element.left_flank_start-1,element.left_flank_end))
        else:
            left_flank_sequence = akita_utils.dna_utils.dna_1hot(genome_open.fetch(element.left_flank_chrom,element.start,element.start))

        insert_sequence = akita_utils.dna_utils.dna_1hot(genome_open.fetch(element.chrom,element.start,element.end))

        if element.right_flank_end :
            right_flank_sequence = akita_utils.dna_utils.dna_1hot(genome_open.fetch(element.right_flank_chrom,element.right_flank_start,element.right_flank_end+1))
        else:
            right_flank_sequence = akita_utils.dna_utils.dna_1hot(genome_open.fetch(element.right_flank_chrom,element.end,element.end))

        seq_1hot_insertion = np.concatenate((left_flank_sequence, insert_sequence, right_flank_sequence), axis=0)

        if element.strand == "-":
            seq_1hot_insertion = akita_utils.dna_utils.hot1_rc(seq_1hot_insertion)
            
        sequences.append(seq_1hot_insertion)

    return sequences


# ctcf1 = INSERT("strong_ctcf", "chr1", 100, 200, [10, 15],"-")
# ctcf2 = INSERT("weak_ctcf",  "chr1", 25, 80, [10, 15],"+")
# ctcf1.replace_flanks(ctcf2)

# gene1 = Gene("fenemw", "chr1", 150, 250,"-")

# output_locus = Locus([INSERT])

# output_locus.insert(gene1)
# output_locus.insert(ctcf1)

# logger.info(output_locus)
