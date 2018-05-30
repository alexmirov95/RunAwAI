#
# nn_training.py
#
# Example of how we can train neural networks evolutionarily. This particular
# implementation trains a neural network that is able to determine whether or
# not a particular point is above or below the graph of a line.

import sys
sys.path.insert(0, './../')

import math
import random

from ga import Individual
from ga import EvolutionaryController
from nn import NeuralNetwork
from nn import (
    NeuralNetworkInputNeuron,
    NeuralNetworkOutputNeuron,
    NeuralNetworkHiddenNeuron,
)

LEARNING_CONSTANT = 1
TEST_SIZE = 100
MIN_TEST_VALUE = -15
MAX_TEST_VALUE = 15
HIDDEN_NET_SIZE = 10

class TestFunction():
    # Choose a test function to test the NN against
    def __init__(self, function):
        self.function = function

    def evaluate(self, p):
        if self.function == 'quadratic':
            return self.quadratic(p)
        elif self.function == 'cubic':
            return self.cubic(p)
        elif self.function == 'sine':
            return self.sine(p)
        return self.identity(p)

    def identity(self, p):
        return p

    def quadratic(self, p):
        return p * p

    def cubic(self, p):
        return p * p

    def sine(self, p):
        return math.sin(p)


class NN(Individual):
    def individual_print(self):
        pass

    def breed_parents(self, parent_tuple, c, breeding_factor):
        # Use NEAT algorithm for breeding
        return parent_tuple[0][0]

    def mutate(self, mutation_constant):
        # mutate both weight and topology
        # Higher mutation constant manipulates more weights
        # Higher mutation constant adds more neurons
        pass


class NNRegression(NN):
    def __init__(self, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def fitness_function(self):
        pass

    def test(self):
        pass


class NNClassification(NN):
    def __init__(self, hidden_size=HIDDEN_NET_SIZE, hfunc_type='identity',
                 test_function='quadratic'):
        self.nn = NeuralNetwork(LEARNING_CONSTANT)
        self.input_neuron_x = self.nn.add_input(None, 'x-coord')
        self.input_neuron_y = self.nn.add_input(None, 'y-coord')
        self.output_neuron = self.nn.add_output('above f(x)', function='binary')
        self.hidden_size = hidden_size
        self.hidden_layer = []
        self.hfunc = hfunc_type
        self.test_function = TestFunction(test_function)

    def fitness_function(self):
        # TODO: should we peanalize network size
        # Generate cases and build overall loss, which we would like to minimize
        test_cases = []
        return_cases = []
        for i in range(TEST_SIZE):
            x_coord = random.uniform(MIN_TEST_VALUE, MAX_TEST_VALUE)
            y_coord = random.uniform(
                self.test_function.evaluate(MIN_TEST_VALUE),
                self.test_function.evaluate(MAX_TEST_VALUE))

            if y_coord > self.test_function.evaluate(x_coord):
                above = True
            else:
                above = False

            test_cases.append(
                [
                    {
                        self.input_neuron_x: x_coord,
                        self.input_neuron_y: y_coord
                    },
                    above
                ]
            )

        total_error = 0
        for case in test_cases:
            r_value = self.nn.feed_forward(case[0])
            if r_value == 1 and case[1] != True:
                total_error += 1
            elif r_value == 0 and case[1] != False:
                total_error += 1

        return total_error / len(test_cases)

    def generate_random(self):
        # Randomly assign weights to the neurons
        for i in range(self.hidden_size):
            new_hidden = NeuralNetworkHiddenNeuron(self.hfunc)
            self.input_neuron_x.add_connection(new_hidden, random.uniform(-1, 1))
            self.input_neuron_y.add_connection(new_hidden, random.uniform(-1, 1))
            self.nn.add_hidden({self.output_neuron: random.uniform(-1, 1)},
                               function=self.hfunc)
            self.hidden_layer.append(new_hidden)

        # Randomly connect some of the hidden neurons (50%)
        # TODO: make sure we do not create loopy graphs, only add downstream
        # connections
        for i in range(self.hidden_size - 1):
            # Add the ability to tweak the ratio of random connections
            if random.randint(0, 1) is 1:
                rand = i
                while rand is i:
                    rand = random.randint(0, self.hidden_size - 1)

                self.hidden_layer[i].add_connection(self.hidden_layer[rand],
                                                    random.uniform(-1, 1))

    def test(self):
        pass


if __name__ == "__main__":
    function = input("Select a function type: ")
    nn = NNClassification(test_function=function)

    trial = EvolutionaryController(NNClassification)
    trial.run_evolution(15, 0, 1)
