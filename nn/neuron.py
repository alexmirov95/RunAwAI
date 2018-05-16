#
# neuron.py
#

import abc
from activation import ActivationFunction

class NeuralNetworkNeuron(abc.ABC):
    """
    """
    @abc.abstractmethod
    def __init__(self):
        self.function = None
        # connections into this neuron
        self.connections = {} # Map of NeuralNetworkNeurons: weight
        self.inputs = {} # Map of NeuralNetworkNeurons: weight*activation


    def activate(self):
        total = 0
        for value in self.inputs:
            total += self.inputs[value]
        return self.function.evaluate(total)


    def add_connection(self, neuron, weight):
        self.connections[neuron] = weight


    def remove_connection(self, neuron):
        neuron.connection[self] = 0


class NeuralNetworkInputNeuron(NeuralNetworkNeuron):
    def __init__(self, label, node_id):
        self.function = ActivationFunction('identity')
        self.connections = {}
        self.inputs = 0
        self.label = label
        self.node_id = node_id


    def activate(self):
        return self.function.evaluate(self.inputs)


    def __str__(self):
        return self.label


class NeuralNetworkOutputNeuron(NeuralNetworkNeuron):
    def __init__(self, label, node_id, function='identity'):
        self.function = ActivationFunction(function)
        self.connections = None
        self.inputs = {}
        self.label = label
        self.node_id = node_id

        
    def __str__(self):
        return self.label


class NeuralNetworkHiddenNeuron(NeuralNetworkNeuron):
    def __init__(self, function_type, node_id, label=None):
        self.function = ActivationFunction(function_type)
        self.connections = {}
        self.inputs = {}
        self.label = label
        self.node_id = node_id


    def __str__(self):
        if self.label:
            return self.label

        return "Hidden Neuron"
