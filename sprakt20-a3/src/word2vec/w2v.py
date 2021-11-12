import os
import time
import argparse
import string
import re
from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

from tqdm import tqdm


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2020 by Dmytro Kalpakchi.
"""


class Word2Vec(object):
    def __init__(self, filenames, dimension=300, window_size=2, nsample=10,
                 learning_rate=0.025, epochs=5, use_corrected=True, use_lr_scheduling=True):
        """
        Constructs a new instance.
        
        :param      filenames:      A list of filenames to be used as the training material
        :param      dimension:      The dimensionality of the word embeddings
        :param      window_size:    The size of the context window
        :param      nsample:        The number of negative samples to be chosen
        :param      learning_rate:  The learning rate
        :param      epochs:         A number of epochs
        :param      use_corrected:  An indicator of whether a corrected unigram distribution should be used
        """
        self.__pad_word = '<pad>'
        self.__sources = filenames
        self.__H = dimension
        self.__lws = window_size
        self.__rws = window_size
        self.__C = self.__lws + self.__rws
        self.__init_lr = learning_rate
        self.__lr = learning_rate
        self.__nsample = nsample
        self.__epochs = epochs
        self.__nbrs = None
        self.__use_corrected = use_corrected
        self.__use_lr_scheduling = use_lr_scheduling
        self.__w2i = {}
        self.__i2w = {}
        self.unigram = []
        self.corrected_unigram = []
        self.word_index = -1
        self.__V = 0
        self.num_words = 0


    def init_params(self, W, w2i, i2w):
        self.__W = W
        self.__w2i = w2i
        self.__i2w = i2w


    @property
    def vocab_size(self):
        return self.__V
        

    def clean_line(self, line):
        """
        The function takes a line from the text file as a string,
        removes all the punctuation and digits from it and returns
        all words in the cleaned line as a list
        
        :param      line:  The line
        :type       line:  str
        """
        line = re.sub('[^A-Za-z ]+', "", line).strip()
        line = " ".join(line.split())
        line = line.split(' ')
        return line


    def text_gen(self):
        """
        A generator function providing one cleaned line at a time

        This function reads every file from the source files line by
        line and returns a special kind of iterator, called
        generator, returning one cleaned line a time.

        If you are unfamiliar with Python's generators, please read
        more following these links:
        - https://docs.python.org/3/howto/functional.html#generators
        - https://wiki.python.org/moin/Generators
        """
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)


    def get_word_index (self, word):
        if self.__w2i.get(word) is None:     
            self.word_index = self.word_index + 1
            self.__w2i[word] = self.word_index
            self.__i2w[self.word_index] = word   
            self.__V = self.__V + 1
        return self.__w2i[word]
            

    def get_context(self, sent, i):
        """
        Returns the context of the word `sent[i]` as a list of word indices
        
        :param      sent:  The sentence
        :type       sent:  list
        :param      i:     Index of the focus word in the sentence
        :type       i:     int
        """

        context_words = []

        for ci in range(1, self.__lws + 1):
            ind = i - ci
            if  ind >= 0 and ind < len(sent):
                cw = sent[ind]
                wi = self.get_word_index(cw)
                context_words.append(wi)

        for ci in range(1, self.__rws + 1):
            ind = i + ci
            if  ind >= 0 and ind < len(sent):
                cw = sent[ind]
                wi = self.get_word_index(cw)
                context_words.append(wi)

        return context_words


    def skipgram_data(self):
        """
        A function preparing data for a skipgram word2vec model in 3 stages:
        1) Build the maps between words and indexes and vice versa
        2) Calculate the unigram distribution and corrected unigram distribution
           (the latter according to Mikolov's article)
        3) Return a tuple containing two lists:
            a) list of focus words
            b) list of respective context words
        """

        focus_words = []
        context_words = []
        unigram = defaultdict(lambda:0)
        corrected_unigram = defaultdict(lambda:0)

        for row in self.text_gen():
            if '' not in row:
                for i, word in enumerate(row):

                    # Build the maps between words and indexes and vice versa
                    wi = self.get_word_index(word)
                    focus_words.append(wi)
                    context_words.append(self.get_context(row, i))

                    # Unigram count
                    unigram[wi] = unigram[wi] + 1
                    self.num_words = self.num_words + 1

        # Calculate unigram and corrected unigram distribution
        unigram = {k: v / self.num_words for k, v in unigram.items()}
        corrected_unigram = {k: v**0.75 for k, v in unigram.items()}
        self.unigram = [unigram[k] for k in sorted(unigram.keys(), reverse=False)]
        self.corrected_unigram = [corrected_unigram[k] for k in sorted(corrected_unigram.keys(), reverse=False)]

        # Normalize probability
        self.corrected_unigram = [float(i)/sum(self.corrected_unigram) for i in self.corrected_unigram]

        #return list(context_mapping.keys()), list(context_mapping.values())
        return focus_words, context_words


    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    
    def get_alpha(self, start_alpha, total_words, n_processed, n_epochs):
        '''
            Calculate the learning rate
        '''
        alpha = start_alpha * (1 - n_processed/(1 + n_epochs * total_words))
        if alpha < start_alpha * 1e-4:
            alpha = start_alpha * 1e-4
        return alpha


    def negative_sampling(self, number, xb, pos):
        """
        Sample a `number` of negatives examples with the words in `xb` and `pos` words being
        in the taboo list, i.e. those should be replaced if sampled.
        
        :param      number:     The number of negative examples to be sampled
        :type       number:     int
        :param      xb:         The index of the current focus word
        :type       xb:         int
        :param      pos:        The index of the current positive example
        :type       pos:        int
        """
        
        all_words = np.arange(self.__V)
        sampling = []
        to_add = xb 
        
        if self.__use_corrected:
            u_to_use = self.corrected_unigram
        else:
            u_to_use = self.unigram

        for _ in range(number):
            while to_add == xb or to_add == pos:
                to_add = np.random.choice(all_words, p=u_to_use)
            sampling.append(to_add)

        return sampling

    
    def get_delta_focus(self, focus_word, context_words, alpha):
        pos_sum = np.zeros((1, self.__H))
        neg_sum = np.zeros((1, self.__H))
        neg_samplings = {}

        for context_word in context_words:

            # Sum over all positive samples
            dot_prod= np.dot(self.__W[focus_word], self.__U[context_word].T)
            pos_sum = pos_sum + self.__U[context_word] * (self.sigmoid(dot_prod) - 1)

            # Update context vector for this context word
            self.__U[context_word] = self.__U[context_word] - alpha * (self.sigmoid(dot_prod) - 1) * self.__W[focus_word]

            # Generate negative samples
            # if neg_samplings.get((focus_word, context_word)) is None:
            #     neg_words = self.negative_sampling(self.__nsample, focus_word, context_word)
            #     neg_samplings[(focus_word, context_word)] = neg_words
            # else:
            #     neg_words = neg_samplings.get((focus_word, context_word))

            neg_words = self.negative_sampling(self.__nsample, focus_word, context_word)

            # Sum over all negative samples
            for neg_word in neg_words:
                dot_prod= np.dot(self.__W[focus_word], self.__U[neg_word].T)
                neg_sum = neg_sum + self.__U[neg_word] * self.sigmoid(dot_prod)

                # Update context vector for this negative word
                sigm = 1 - self.sigmoid(np.dot(self.__W[focus_word], -self.__U[neg_word].T))
                self.__U[neg_word] = self.__U[neg_word] - alpha * self.__W[focus_word] * sigm

        return pos_sum + neg_sum


    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        focus_words, context_words_list = self.skipgram_data()
        N = len(focus_words)
        print("Dataset contains {} datapoints".format(N))

        alpha_start = self.__lr
        alpha = self.__lr

        # Initialize focus and context vectors for each word in the vocabulary
        self.__W = np.random.uniform(low=-0.5, high=0.5, size=(self.__V, self.__H)) # focus vectors
        self.__U = np.random.uniform(low=-0.5, high=0.5, size=(self.__V, self.__H)) # context vectors

        for _ in range(self.__epochs):

            # Run through the training datapoints
            for i in tqdm(range(N)):
                focus_word = focus_words[i]
                context_words = context_words_list[i]
                alpha = self.get_alpha(alpha_start, N, i + 1, self.__epochs)

                # Compute the gradients for the focus word, update context vetors for all its context words
                # as well as n_context_words * n_sample number of negative samples
                delta_focus = self.get_delta_focus(focus_word, context_words, alpha)

                # Update focus vector for focus word
                self.__W[focus_word] = self.__W[focus_word] - alpha * delta_focus


    def find_nearest(self, words, metric, k=5):
        """
        Function returning k nearest neighbors with distances for each word in `words`
        
        We suggest using nearest neighbors implementation from scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
        carefully their documentation regarding the parameters passed to the algorithm.
    
        To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
        "Harry" and "Potter" using some distance metric `m`. 
        For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.
        The output of the function would then be the following list of lists of tuples (LLT)
        (all words and distances are just example values):
    
        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
         [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
        
        The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
        list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
        The tuples are sorted either by descending similarity or by ascending distance.
        
        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """
        results = []
        values = self.__W
        neigh = NearestNeighbors(n_neighbors=k, metric='cosine')
        neigh.fit(values)

        for word in words:
            word_index = self.__w2i[word]
            a = values[word_index].reshape(1, -1)
            nbs = neigh.kneighbors(a, return_distance=True)

            # Generate results
            dists = nbs[0].flatten()
            n_inds = nbs[1].flatten()
            result = []
            for i, n_ind in enumerate(n_inds):
                di = round(dists[i], 2)
                neigh_word = self.__i2w[n_ind]
                result.append((neigh_word, di))
            results.append(result)

        return results


    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        if self.__use_corrected:
            filename = 'w2v_corrected.txt'
        else:
            filename = "w2v.txt"

        with open(filename, 'w') as f:
            W = self.__W
            f.write("{} {}\n".format(self.__V, self.__H))
            for i, word in self.__i2w.items():
                f.write(str(word) + " " + " ".join(map(lambda x: "{0:.6f}".format(x), W[i,:])) + "\n")


    @classmethod
    def load(cls, fname):
        """
        Load the word2vec model from a file `fname`
        """
        w2v = None
        try:
            with open(fname, 'r') as f:
                V, H = (int(a) for a in next(f).split())
                w2v = cls([], dimension=H)

                W, i2w, w2i = np.zeros((V, H)), {}, {}
                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    w2i[word] = i
                    i2w[i] = word
                    W[i] = list(map(float, parts[1:]))

                w2v.init_params(W, w2i, i2w)
        except:
            print("Error: failing to load the model to the file")
        return w2v


    def interact(self):
        """
        Interactive mode allowing a user to enter a number of space-separated words and
        get nearest 5 nearest neighbors for every word in the vector space
        """
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text, 'cosine')

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')


    def train_and_persist(self):
        """
        Main function call to train word embeddings and being able to input
        example words interactively
        """
        self.train()
        self.write_to_file()
        self.interact()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-t', '--text', default='harry_potter_1.txt',
                        help='Comma-separated source text files to be trained on')
    parser.add_argument('-s', '--save', default='w2v_corrected.txt', help='Filename where word vectors are saved')
    parser.add_argument('-d', '--dimension', default=50, help='Dimensionality of word vectors')
    parser.add_argument('-ws', '--window-size', default=2, help='Context window size')
    parser.add_argument('-neg', '--negative_sample', default=10, help='Number of negative samples')
    parser.add_argument('-lr', '--learning-rate', default=0.025, help='Initial learning rate')
    parser.add_argument('-e', '--epochs', default=5, help='Number of epochs')
    parser.add_argument('-uc', '--use-corrected', action='store_true', default=True,
                        help="""An indicator of whether to use a corrected unigram distribution
                                for negative sampling""")
    parser.add_argument('-ulrs', '--use-learning-rate-scheduling', action='store_true', default=True,
                        help="An indicator of whether using the learning rate scheduling")
    args = parser.parse_args()

    args.use_corrected = False
    if args.use_corrected == False:
        args.save = 'w2v.txt'

    if os.path.exists(args.save):
        w2v = Word2Vec.load(args.save)
        if w2v:
            w2v.interact()
    else:
        w2v = Word2Vec(
            args.text.split(','), dimension=args.dimension, window_size=args.window_size,
            nsample=args.negative_sample, learning_rate=args.learning_rate, epochs=args.epochs,
            use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
        )
        w2v.train_and_persist()
