import math
import argparse
import codecs
from collections import defaultdict
import random
import importlib
from BigramTrainer import BigramTrainer
import operator
import numpy as np


"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):
    
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # A hashmap holding the bigram counts.
        self.bigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        # Important that it is named self.logProb for the --check flag to work
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0

        self.end_tokens = ['.', ',', '!', '\'m', ';']


    def calculate_pair_probability (self, i1, i2):
        pair_prob = self.bigram_count[i1][i2] / self.unigram_count[i1]
        return pair_prob


    def calculate_bigram_probability (self):
        for i1, v1 in self.bigram_count.items():
            for i2, _ in v1.items():
                self.bigram_prob[i1][i2] = math.log(self.calculate_pair_probability(i1, i2))


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        bigram_trainer = BigramTrainer()
        bigram_trainer.process_files(filename)

        self.index = bigram_trainer.index
        self.word = bigram_trainer.word
        self.unigram_count = bigram_trainer.unigram_count
        self.unique_words = bigram_trainer.unique_words
        self.total_words = bigram_trainer.total_words
        self.bigram_count = bigram_trainer.bigram_count
        self.calculate_bigram_probability()

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and following the distribution
        of the language model.
        """ 

        generated_text = []
        generated_text.append(w)

        w_index = self.index.get(w)

        if w_index is None: 
            print('In valid start word. No text can be generated.')
        else:
            for _ in range(n-1):
                probs = self.bigram_prob.get(w_index)

                # If all probs are zeros (no successor), pick random word in the corpus
                if probs is None:
                    w_index = np.random.choice(list(self.word), 1, replace=False)[0]

                else:
                    # Step 1: Generate a random number
                    rand_num = np.random.uniform()
                    lower_bound = 0
                    upper_bound = 0

                    # Step 2: Check which word the number represents by checking each interval
                    #         interval = lowerbound + probability
                    for identifier in probs:
                        lower_bound = upper_bound
                        upper_bound = upper_bound + np.exp(probs[identifier])
                        if rand_num <= upper_bound and rand_num > lower_bound:
                            w_index = identifier
                            break  

                w_to_add = self.word[w_index]
                if w_to_add not in self.end_tokens:
                    generated_text.append(' ')
                generated_text.append(self.word[w_index])

            print(''.join(generated_text))


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start,arguments.number_of_words)

if __name__ == "__main__":
    main()
