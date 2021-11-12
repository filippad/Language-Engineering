#  -*- coding: utf-8 -*-
import math
import argparse
import nltk
import codecs
from collections import defaultdict
import json
import requests
import numpy as np

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.

Task 3: 

test guardian_guardian: 6.65
test austen_austen: 5.77
test guardian_austen: 6.40
test austen_guardian: 9.78

"""

class BigramTester(object):
    def __init__(self):
        """
        This class reads a language model file and a test file, and computes
        the entropy of the latter. 
        """
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(lambda: defaultdict(int))

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
        self.pre_processed_word = '--'

    def map_section_2 (self, line):
        parts = line.strip().split(' ')
        return int(parts[0]), parts[1], int(parts[2])

    
    def map_section_3 (self, line):
        parts = line.strip().split(' ')
        return int(parts[0]), int(parts[1]), float(parts[2])


    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                
                for _ in range(self.unique_words):
                    identifier, word, uni_count = self.map_section_2(f.readline())
                    self.word[identifier] = word
                    self.index[word] = identifier
                    self.unigram_count[identifier] = uni_count

                while True:
                    line = f.readline()
                    if line.strip() == '-1':
                        break
                    i1, i2, prob = self.map_section_3(line)
                    self.bigram_prob[i1][i2] = prob
                
                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False


    def calculate_linear_interpolation (self, i1, i2):
        uni_prob = self.unigram_count[i2] / self.total_words
        lambda1_term = np.exp(self.bigram_prob[i1][i2])
        if self.bigram_prob[i1][i2] == 0:
            lambda1_term = 0
        return self.lambda1 * lambda1_term + self.lambda2 * uni_prob + self.lambda3

    def compute_entropy_cumulatively(self, word):

        self.test_words_processed = self.test_words_processed + 1
        this_logProb = 0

        # Process the first word in the corpus, lambda1 term = 0
        if self.test_words_processed == 1:   
            word_index = self.index.get(word)
            if word_index is None:
                uni_count = 0
            else: 
                uni_count = self.unigram_count[word_index]
            uni_prob = uni_count/self.total_words
            this_logProb = math.log(self.lambda2*uni_prob + self.lambda3)

            if uni_count == 0:
                self.pre_processed_word = -1    
            else: 
                self.pre_processed_word = word_index

        else: 
            # If the word is not in the trained model, lambda1 and lambda2 terms = 0
            if self.index.get(word) is None:
                this_logProb = math.log(self.lambda3)
                self.pre_processed_word = -1

            else:
                i= self.index[word]
                prob = self.calculate_linear_interpolation(self.pre_processed_word, i)
                this_logProb = math.log(prob)
                self.pre_processed_word = i
        self.logProb = self.logProb + this_logProb 

    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(test_filename, 'r', 'utf-8') as f:
                self.tokens = nltk.word_tokenize(f.read().lower()) # Important that it is named self.tokens for the --check flag to work
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)
                self.logProb = - self.logProb/len(self.tokens)
            return True
        except IOError:
            print('Error reading testfile')
            return False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--test_corpus', '-t', type=str, required=True, help='test corpus')
    parser.add_argument('--check', action='store_true', help='check if your alignment is correct')

    arguments = parser.parse_args()

    bigram_tester = BigramTester()
    bigram_tester.read_model(arguments.file)
    bigram_tester.process_test_file(arguments.test_corpus)
    if arguments.check:
        results  = bigram_tester.logProb

        payload = json.dumps({
            'model': open(arguments.file, 'r').read(),
            'tokens': bigram_tester.tokens,
            'result': results
        })
        response = requests.post(
            'https://language-engineering.herokuapp.com/lab2_tester',
            data=payload,
            headers={'content-type': 'application/json'}
        )
        response_data = response.json()
        if response_data['correct']:
            print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb))
            print('Success! Your results are correct')
        else:
            print('Your results:')
            print('Estimated entropy: {0:.2f}'.format(bigram_tester.logProb))
            print("The server's results:\n Entropy: {0:.2f}".format(response_data['result']))

    else:
        print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb))

if __name__ == "__main__":
    main()
