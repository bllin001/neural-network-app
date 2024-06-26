# Description: Implementation of Perceptron Algorithm
# Author: Brian Llinas
# Last Modified: 2020-11-30

#=========================================== Import Libraries ============================================#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#=========================================== Activation Function ============================================#

def activation_function(z, activation=None):
    if activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    elif activation == 'tanh':
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    elif activation == 'relu':
        return np.where(z > 0, z, 0)
    elif activation == 'leaky_relu':
        return np.where(z > 0, z, 0.01 * z)
    elif activation == 'elu':
        return np.where(z > 0, z, 0.01 * (np.exp(z) - 1))
    elif activation == 'softmax':
        return np.exp(z) / np.sum(np.exp(z), axis=0)
    elif activation == 'binary_step':
        return np.where(z > 0, 1, 0)   

#=========================================== Perceptron ============================================#

class Perceptron:

    # Initialize the model
    def __init__(self, epochs=1, learning_rate=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.activation_func = activation_function
        self.w = None

    # Fit the model
    def fit(self, X, y, w, activation='binary_step'):
        self.w = w
        self.loss = []
        self.iteration = []
        self.activation = activation

        # Loop through all epochs in X
        for epoch in range(self.epochs):

            total_error = 0 # Initialize total error for each epoch

            # Loop through all rows in X
            for i in range(X.shape[0]):
                w_initial = self.w # Save the initial weights
                z = np.dot(X[i], self.w) # Calculate the dot product between the weights and the inputs
                y_hat = self.activation_func(z, self.activation)
                error = y[i] - y_hat # Calculate the error

                # Update the weights
                if error != 0:
                    update = self.learning_rate * error
                    self.w += update * X[i]
                    total_error += np.abs(error)
                else:
                    update = 0
                    self.w = self.w

                # Append w_initial, X[i], y[i], y_hat, error, w
                self.iteration.append([epoch + 1, i, np.round(w_initial,3), X[i][1], X[i][2], y[i], np.round(y_hat,3), np.round(error,3), np.round(self.w,3)])
            # Append the total error for each epoch    
            self.loss.append([epoch + 1, total_error])
        
        # Convert the iteration and loss to a dataframe
        self.iteration = pd.DataFrame(self.iteration, columns=['Epoch', 'Row', 'Initial_Weights', 'X1', 'X2', 'y', 'y_hat', 'Error', 'Weights'])
        self.loss = pd.DataFrame(self.loss, columns=['Epoch', 'Total_Loss'])
        self.w = np.round(self.w,3)
        
    # Predict the output
    def predict(self, X):
        z = np.dot(X, self.w)
        y_hat = self.activation_func(z, self.activation)
        return y_hat

