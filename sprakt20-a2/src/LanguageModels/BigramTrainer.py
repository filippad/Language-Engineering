#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs
import json
import requests

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""


class BigramTrainer(object):
    """
    This class constructs a bigram language model from a corpus.
    """

    def process_files(self, f):
        """
        Processes the file @code{f}.
        """
        print(f)
        with codecs.open(f, 'r', 'utf-8') as text_file:
            text = reader = str(text_file.read()).lower()
        try :
            self.tokens = nltk.word_tokenize(text) # Important that it is named self.tokens for the --check flag to work
        except LookupError :
            nltk.download('punkt')
            self.tokens = nltk.word_tokenize(text)
        for token in self.tokens:
            self.process_token(token)


    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram and
        bigram counts.

        :param token: The current word to be processed.
        """

        # Always increase the number of total words in the corpus
        self.total_words = self.total_words + 1

        # Check if token is a new entry
        if self.index.get(token) is None:
            token_index = self.unique_words
            self.index[token] = token_index
            self.word[token_index] = token
            self.unique_words = self.unique_words + 1
        else:
            token_index = self.index[token]

        # Update unigram count
        count = self.unigram_count[token_index]
        self.unigram_count[token_index] = count + 1

        # Update bigram count
        if self.last_index > -1:
            count = self.bigram_count[self.last_index][token_index]
            self.bigram_count[self.last_index][token_index] = count + 1

        self.last_index = token_index


    def stats(self):
        """
        Creates a list of rows to print of the language model.

        """
        rows_to_print = []

        # First row
        first_row = []
        first_row.append(self.unique_words)
        first_row.append(self.total_words)
        rows_to_print.append(' '.join(map(str, first_row)))

        # Print unigram counts
        for identifier in self.word:
            this_row = []
            this_row.append(identifier)
            this_row.append(self.word[identifier])
            this_row.append(self.unigram_count[identifier])
            rows_to_print.append(' '.join(map(str, this_row)))

        # Print bigram probabilities
        for i1, v1 in self.bigram_count.items():
            for i2, _ in v1.items():
                this_row = []
                this_row.append(i1)
                this_row.append(i2)
                this_row.append("{:.15f}".format(math.log(self.calculate_bigram_prob(i1, i2))))
                rows_to_print.append(' '.join(map(str, this_row)))

        # End row
        rows_to_print.append(str(-1))

        return rows_to_print

    
    def calculate_bigram_prob (self, i1, i2, lamda1=0.99, lamda3=1e-6):
        bigram_prob = self.bigram_count[i1][i2] / self.unigram_count[i1]
        return bigram_prob


    def __init__(self):
        """
        <p>Constructor. Processes the file <code>f</code> and builds a language model
        from it.</p>

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        """
        The bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway).
        """
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # The identifier of the previous word processed.
        self.last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        self.laplace_smoothing = False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')
    parser.add_argument('--check', action='store_true', help='check if your alignment is correct')

    arguments = parser.parse_args()

    bigram_trainer = BigramTrainer()

    bigram_trainer.process_files(arguments.file)

    if arguments.check:
        results  = bigram_trainer.stats()
        payload = json.dumps({
            'tokens': bigram_trainer.tokens,
            'result': results
        })
        response = requests.post(
            'https://language-engineering.herokuapp.com/lab2_trainer',
            data=payload,
            headers={'content-type': 'application/json'}
        )
        response_data = response.json()
        if response_data['correct']:
            print('Success! Your results are correct')
            #for row in results: print(row)
        else:
            print('Your results:\n')
            #for row in results: print(row)
            print("The server's results:\n")
            #for row in response_data['result']: print(row)
    else:
        stats = bigram_trainer.stats()
        if arguments.destination:
            with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
                for row in stats: f.write(row + '\n')
        else:
            for row in stats: print(row)


if __name__ == "__main__":
    main()
