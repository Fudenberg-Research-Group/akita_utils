'''
Module Description:

The ToyGenomeOpen module provides a simple implementation of a toy genome representation. It includes the ToyGenomeOpen class, which allows fetching subsequences from the genome and retrieving the length of chromosomes.

Class Description - ToyGenomeOpen:
The ToyGenomeOpen class represents a toy genome. It is designed to store and retrieve genomic information in a simplified manner. The class allows users to fetch subsequences from specific chromosomes and retrieve the lengths of chromosomes.

Functionality:

    - Initialization: The ToyGenomeOpen class is initialized with a dictionary of chromosome names as keys and their respective sequences as values.
    - Fetching Subsequences: The fetch method enables users to retrieve a subsequence from a specific chromosome by providing the chromosome name, start position, and end position. It returns the subsequence as a string. If the specified chromosome is not found, a ValueError is raised.
    - Retrieving Chromosome Length: The get_reference_length method allows users to obtain the length of a specific chromosome by providing its name. It returns the length as an integer. If the specified chromosome is not found, a ValueError is raised.
'''

class ToyGenomeOpen:
    def __init__(self, genome_data):
        """
        Initialize the ToyGenomeOpen class with genome data.

        Args:
            genome_data (dict): A dictionary containing chromosome names as keys and their respective sequences as values.
        """
        self.genome_data = genome_data

    def fetch(self, chrom, start, end):
        """
        Fetch a subsequence from the specified chromosome.

        Args:
            chrom (str): The name of the chromosome.
            start (int): The start position of the subsequence (inclusive).
            end (int): The end position of the subsequence (exclusive).

        Returns:
            str: The subsequence of the specified chromosome.

        Raises:
            ValueError: If the chromosome is not found in the genome data.
        """
        chromosome = self.genome_data.get(chrom)
        if chromosome:
            return chromosome[start:end]
        else:
            raise ValueError(f"Chromosome {chrom} not found")

    def get_reference_length(self, chrom):
        """
        Get the length of the specified chromosome.

        Args:
            chrom (str): The name of the chromosome.

        Returns:
            int: The length of the specified chromosome.

        Raises:
            ValueError: If the chromosome is not found in the genome data.
        """
        chromosome = self.genome_data.get(chrom)
        if chromosome:
            return len(chromosome)
        else:
            raise ValueError(f"Chromosome {chrom} not found")


if __name__ == "__main__":
    # Example usage
    genome_data = {
        "chr1": "AGCTCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
        "chr2": "GATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT",
    }

    toy_genome = ToyGenomeOpen(genome_data)

    subsequence = toy_genome.fetch("chr1", 5, 10)
    print(subsequence)  # Output: "GATCG"

    length = toy_genome.get_reference_length("chr2")
    print(length)  # Output: 71
