# Originally written by Kyle Chickering, Taaha Chaudhry, Xin Jin, Alex Mirov and
# Simon Wu for ECS 170 @ UC Davis, Spring 2018. 
#
# neat.py
#
# Implementation of the NEAT architecuture for neuroevolution. NEAT was created
# by Kenneth Stanley and Risto Miikkulainen @ University of Texas, Austin.
#

EXCESS_GENOME_COMPATIBILITY_CONSTANT = 1
DIJOINT_GENOME_COMPATIBILITY_CONSTANT = 1
WEIGHT_GENOME_COMPATIBILITY_CONSTANT = 1

from nn.activation import ActivationFunction

class NEATNet(object):

    def __init__(self):
        self.genome = Genome()
        self.inputs = []
        self.outputs = []
        self.hidden = []

    def structural_mutatation(self):
        pass

    def non_structural_mutation(self):
        pass

    def breed(self, mate):
        pass

    def feed_forward(self):
        pass

    def back_propogate(self):
        pass


class NEATNeuron(object):

    def __init__(self, n_type, label=None, function='binary', p1=1, p2=1):
        self.function = ActivationFunction(
            function=function, parameter1=p1, parameter2=p2)
        self.inputs = []
        self.value = 0
        self.connections = {}
        self.n_type = n_type
        
        if (self.n_type == 'input' or self.n_type == 'output') and not label:
            raise Exception("Input and output neurons require a label")
        
        self.label = label

    def reset_neuron(self):
        self.inputs = []
        self.value = 0
        
    def evaluate(self):
        return self.function.evaluate(self.value)


class Genome(object):

    def __init__(self):
        self.innovation_number = 0

    def add_gene(self):
        pass

    def calculate_compatibility(self, genome,
                                c1=EXCESS_GENOME_COMPATIBILITY_CONSTANT,
                                c2=DISJOINT_GENOME_COMPATIBILITY_CONSTANT,
                                c3=WEIGHT_GENOME_COMPATIBILITY_CONSTANT):
        # return the delta function of the two genomes
        pass
