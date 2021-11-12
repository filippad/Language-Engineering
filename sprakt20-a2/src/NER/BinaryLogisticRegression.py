from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    #LEARNING_RATE = 0.01  # The learning rate.
    LEARNING_RATE = 1  # The learning rate.
    CONVERGENCE_MARGIN = 0.0005  # The convergence criterion.
    MAX_ITERATIONS = 1e6 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1000 # Minibatch size (only for minibatch gradient descent)

    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y).reshape(-1,1)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.random.uniform(0, 1, self.FEATURES)

    # ----------------------------------------------------------------------


    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / (1 + np.exp(-z))


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """
        x = self.x[datapoint]
        if label == 1:
            return self.sigmoid(np.dot(x, self.theta.T))
        return 1 - self.sigmoid(np.dot(x, self.theta.T))


    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """
        
        for f in range(self.FEATURES):
            # Calculate cumulative sum
            sum = 0
            for i in range(self.DATAPOINTS):
                x = self.x[i]
                label = self.y[i]
                diff = self.sigmoid(np.dot(x, self.theta.T)) - label
                sum = sum + x[f] * diff
            
            # Update this gradient of this feature
            self.gradient[f] = sum / self.DATAPOINTS


    def compute_gradient_minibatch(self, X, Y):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """
        # for f in range(self.FEATURES):
        #     diff = self.sigmoid(np.dot(X, self.theta.T)) - Y
        #     grad = np.dot(X[:,f].T, diff)[0] / X.shape[0]
        #     self.gradient[f] = grad

        for f in range(self.FEATURES):
            # Calculate cumulative sum
            sum = 0
            for i in range(len(Y)):
                x = X[i]
                label = Y[i]
                diff = self.sigmoid(np.dot(x, self.theta.T)) - label
                sum = sum + x[f] * diff
            
            # Update this gradient of this feature
            self.gradient[f] = sum / len(Y)


    def compute_gradient(self, index):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """

        x = self.x[index].flatten()
        label = self.y[index].flatten()[0]
        for f in range(self.FEATURES):
            diff = self.sigmoid(np.dot(x, self.theta.T)) - label       
            self.gradient[f] = x[f] * diff


    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        iters = 0
        while iters < self.MAX_ITERATIONS:
            # Randomly choose one datapoint (index) and compute gradient
            i = np.random.randint(0, high=self.DATAPOINTS, size=1)
            self.compute_gradient(i)

            # Update weights
            self.theta = self.theta - self.LEARNING_RATE * self.gradient

            # Track convergence
            if (iters % 1000 == 0):
                self.update_plot(np.sum(np.square(self.gradient)))

            iters = iters + 1
            #print('iter :', iters, ' - grads: ', self.gradient)


    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_plot(self.FEATURES)
        # print(self.x.shape)

        iters = 0
        while np.sum(np.square(self.gradient)) > self.CONVERGENCE_MARGIN:
            # Shuffle the dataset
            indices = np.arange(self.DATAPOINTS)
            np.random.shuffle(indices)
            self.x = self.x[indices,:]
            self.y = self.y[indices]

            # Devide into minibatches anf train these one by one
            for i in np.arange(1, self.DATAPOINTS/self.MINIBATCH_SIZE, 1):
                start = int((i-1) * self.MINIBATCH_SIZE)
                end = int(start+self.MINIBATCH_SIZE)
                X = self.x[start:end, :]
                Y = self.y[start:end]

                self.compute_gradient_minibatch(X, Y)

                 # Track convergence
                # if (iters % 50 == 0):
                #     self.update_plot(np.sum(np.square(self.gradient)))
                
                # Update weights
                self.theta = self.theta - self.LEARNING_RATE * self.gradient
                iters = iters + 1
                print('iteration :', iters, 'gradients: ', np.sum(np.square(self.gradient)))


    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)
       
        iters = 0
        while np.sum(np.square(self.gradient)) > self.CONVERGENCE_MARGIN:

            self.compute_gradient_for_all()

            # Update weights
            self.theta = self.theta - self.LEARNING_RATE * self.gradient

            # Track convergence
            if (iters % 3 == 0):
                self.update_plot(np.sum(np.square(self.gradient)))

            iters = iters + 1
            #print('iteration :', iters, 'gradients: ', self.gradient)


    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:')

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='') 
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))


    def print_result(self):
        print('theta: ', ' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print('gradient: ', ' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    #b.fit()
    b.stochastic_fit()
    b.print_result()
    #b.classify_datapoints(x, y)


if __name__ == '__main__':
    main()
