# Description: Class to hold data for the Perceptron
# Author: Brian Llinas
# Last Modified: 2020-11-30

#=========================================== Import Libraries ============================================#

import numpy as np

#=========================================== Data ============================================#

#=========================================== Data ============================================#

class data:
    def __init__(self, logic=None):
        self.logic = logic
        self.X = []
        self.y = []

        if logic == 'AND':
            self.X = [[0,0],
                [1,0],
                [0,1],
                [1,1]]

            self.X = np.array(self.X)

            self.y = [0,0,0,1]
        elif logic == 'OR':
            self.X = [[0,0],
                [1,0],
                [0,1],
                [1,1]]

            self.X = np.array(self.X)    

            self.y = [0,1,1,1]
        elif logic == 'XOR':
            self.X = [[0,0],
                [1,0],
                [0,1],
                [1,1]]

            self.X = np.array(self.X)    

            self.y = [0,1,1,0]        
