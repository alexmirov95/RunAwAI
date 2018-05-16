#
# activation.py
#

import math

class ActivationFunction():
    """
    """
    
    def __init__(self, function='binary', parameter1=1, parameter2=0):
        self.function = function
        self.p1 = parameter1
        self.p2 = parameter2
        
    def evaluate(self, value):
        """
        """
        if self.function == "binary":
            return self.binary(value)
        
        elif self.function == "tanh":
            return self.tanh(value)
        
        elif self.function == "identity":
            return self.identity(value)

        elif self.function == "sigmoid":
            return self.sigmoid(value)

        return self.identity(value)
    
    def binary(self, value):
        """
        """
        if value < self.p1:
            return 0 + self.p2

        return self.p1 + self.p2

    def identity(self, value):
        """
        """
        return value

    def tanh(self, value):
        """
        """
        return math.tanh(value)

    def sigmoid(self, value):
        return 1 / (1 + math.exp(-value))
