# Originally written by Kyle Chickering, Taaha Chaudhry, Xin Jin, Alex Mirov and
# Simon Wu for ECS 170 @ UC Davis, Spring 2018.
#

"""
nn.py

Implements a non-recurrent, NEAT compatiable neural network. NEAT was created by
Kenneth Stanley and Risto Miikkulainen @ University of Texas, Austin.
"""

import math
import queue
import random

from neuron import (
    NeuralNetworkInputNeuron,
    NeuralNetworkOutputNeuron,
    NeuralNetworkHiddenNeuron
)

class NEATEvolutionController(object):
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def step_generation(self):
        innovation_number = 0
        genome = []

        
class NeuralNetwork(object):
    """
    Implementaion of a semi-topologically limited neural network. NeuralNetwork
    creates three layer neural networks, where the hidden layer can be any
    topology.

    The network is implemented using a graph, and computes feed-forward values
    using a breadth-first algorithm. Since implementing the NEAT algorithm for
    neural networks requires an explicit network, traditional implicit matrix 
    implementations do not suffice for our purposes.

    We still need to run benchmarking tests to see how the graph implementation
    of a neural network compares to the traditional implicitly defined nueral
    networks.
    """
    # TODO: Support a bias neuron
    def __init__(self, input_layer_size, output_layer_size,
                 hidden_function_type='tanh', learning_constant=1,
                 struct_mut_con_rate=0.5, struct_mut_new_rate=0.5,
                 n_struct_mut_rate=0.001):
        """
        Pass learning constant for this neural network.

        input_layer_size: An integer for the number of inputs to this network.
        output_layer_size: An integer for the number of outputs to this network.
        input_labels: An array of string labels for the inputs.
        output_labels: An array of string labels for the outputs.
        learning_constant: The learning constant for this network
        struct_mut_con_rate: The rate at which structural connection mutations
          occur. This is a floating point value, where a 0 is never and a 1 is 
          every time the network is mutated.
        struct_mut_new_rate: The rate at which new nodes are added in a 
          mutation. This is a floating point value, where a 0 is never and a 1 
          is every time the network is mutated.
        n_struct_mut_rate: The rate at which non_structural (weight) mutations
          occur. This is a floating point value, where a 0 is no weights change
          and a 1 is every single weight is changed during each mutation.
        return: None
        """
        self.inputs = []
        self.input_size = input_layer_size
        self.outputs = []
        self.output_size = output_layer_size
        self.hidden = []
        self.const = learning_constant
        self.function = hidden_function_type

        # Genome stuff
        self.genome = {} #(nodeA_id, nodeB_id, innovation, disabled)
        self.innovation_number = 0

        self.total_connections = 0
        self.total_nodes = 0

        # Constants
        self.struct_mut_con_rate = struct_mut_con_rate
        self.struct_mut_new_rate = struct_mut_new_rate
        self.n_struct_mut_rate = n_struct_mut_rate


    def incn(self):
        """
        INCrement Nodes. This is a simple wrapper to increment the network node
        count. This method should not be called except by other class methods.

        return: Incremented total node number
        """
        return self.total_nodes += 1

        
    def build_network(self, input_labels, output_labels):
        """
        Builds a network from the input and output labels. Used for creating an
        initial neural network from scratch.
        
        inputs_labels:
        output_labels:
        return: None
        """
        for i in range(self.output_size):
            self.outputs.append(NeuralNetworkOutputNeuron(output_labels[i - 1],
                                                          self.incn()))
            
        for i in range(self.input_size):
            self.inputs.append(NeuralNetworkInputNeuron(input_labels[i - 1],
                                                        self.incn()))
            
            for output in self.outputs:
                self.inputs[i].add_connection(output, random.uniform(-1,1))
                self.genome[(self.inputs[i].node_id, output.node_id)]  = (self.innovation_number, False)
                self.innovation_number += 1
                self.total_connections += 1

        return


    def feed_forward(self, inputs):
        """
        Feed forward values are calculated by doing a breadth-first traversal
        of the neural network. At each neuron, we calculate the activation value
        and propgate the values all the way to the output layers.

        Calculates the values of the inputs given a list of inputs.

        inputs: An array of values for the inputs in order of their labels.
          that are being input to the neural net.
        return: A dict of {output: value} where each of the outputs of the
          network is associated with the value that it ended with.
        """
        # TODO: Verify that the inputs are valid
        self.network_verify()
        
        outputs = {}
        open_list = queue.Queue()
        for i in range(len(self.inputs) - 1):
            open_list.put(self.inputs[i])
            self.inputs[i].inputs = inputs[i]

        while not open_list.empty():
            current_node = open_list.get()
            activation = current_node.activate()

            if isinstance(current_node, NeuralNetworkOutputNeuron):
                outputs[current_node] = activation

            else:
                for connection in current_node.connections:
                    connection.inputs[current_node] = current_node.connections[
                        connection]*activation*self.const
                    open_list.put(connection)

        return outputs


    def back_propogate(self):
        """
        Modify the neural network using back-propogation
        """
        # TODO: Implement back-propogation algorithm
        pass


    def check_unique_label(self, label):
        """
        Check that a label is not already used as an input, output or hidden
        label.

        label: The label to check for uniqueness.
        return True if the label is unique, false otherwise.
        """
        for i in self.inputs + self.outputs + self.hidden:
            if i.label == label:
                return False

        return True


    def structural_mutation(self, global_innovation, generation_genome):
        """
        Structural mutations are adding connections and adding nodes. Nodes and
        connections are added with probability struct_mut_new_rate and
        struct_mut_con_rate respectivly.
        
        return: None
        """
        # Add a connection or a node
        if random.uniform(0, 1) < self.struct_mut_con_rate:
            # If we can add a connection, then do so
            pass

        if random.uniform(0, 1) < self.struct_mut_new_rate:
            connected_nodes = self.inputs + self.hidden
            node = connected_nodes[random.randint(0, len(connected_nodes) - 1)]
            coice = random.choice(list(node.connections.keys()))
            new_hidden = NeuralNetworkHiddenNeuron(self.function, self.incn())
            new_connections = {}
            
            for connection in node.connections.keys():
                new_hidden.add_connection(connection, node.connections[connection])
                
            node.connections = {new_hidden: 1}

            self.hidden.append(new_hidden)

            # Must update both global and local genomes
            #

            valid_keys = []
            for key in self.genome.keys():
                if key[0] == node.node_id:
                    valid_keys.append(key)
                    
            for gene in valid_keys:
                if gene[0] == node.node_id:
                    key = (new_hidden.node_id, node.node_id)
                    if key in generation_genome.keys():
                        new_gene = (generation_genome[key], False)
                    else:
                        new_gene = (global_innovation, False)
                        global_innovation += 1
                        generation_genome[key] = new_gene

                        generation_genome[gene] = (generation_genome[gene][0],
                                                   True) 

                    self.genome[gene] = (self.genome[gene][0], True)


        return global_innovation


    def non_structural_mutation(self):
        """
        Adds non-structural mutations to the network at a rate propertional to
        the n_struct_mut_rate. Non-structural mutation means that the weights
        between nodes are permuted.

        return: None
        """
        manipulations = math.ceil(self.n_struct_mut_rate*self.total_connections)
        connected_nodes = self.inputs + self.hidden
        for i in range(manipulations):
            node = connected_nodes[random.randint(0, len(connected_nodes) - 1)]
            choice = random.choice(list(node.connections.keys()))
            node.connections[choice] += random.uniform(-1 - 
                                                       node.connections[choice],
                                                       1 - 
                                                       node.connections[choice])

        return


    def breed(self, mate, this_fitness, mate_fitness):
        """
        Breeds two neural networks and returns their child.
        """
        child = NeuralNetwork(self.input_size, self.output_size,
                              hidden_function_type=self.funciton,
                              learning_constant=self.const,
                              struct_mut_con_rate=self.struct_mut_con_rate,
                              struct_mut_new_rate=self.struct_mut_new_rate,
                              n_struct_mut_rate=self.n_struct_mut_rate)

        child_genome = {}
        more_fit = None
        less_fit = None
        
        if this_fitness > mate_fitness:
            more_fit = self
            less_fit = mate
        else:
            more_fit = mate
            less_fit = self

        shared = {}
        disjoint_more = {}
        disjoint_less = {}

        for key in more_fit.keys():
            if key in less_fit.keys():
                shared.append(key)
            else:
                disjoint_more.append(key)

        for key in less_fit.keys():
            if key not in more_fit.keys():
                disjoint_less.append(key)

        # sort the three lists by their innovation numbers and combine

        return child


    def network_verify(self):
        """
        Verify that the network is valid, in other words it has no loops.
        Note that loopy networks are allowed if we make the move to time
        dependent recurrent neural networks.

        return: True if network is valid, False if it is not.
        """
        open_list = queue.Queue()
        visited = []
        for i in self.inputs:
            open_list.put(i)

        while not open_list.empty():
            current_node = open_list.get()
            if current_node in visited:
                return False
            
            visited.append(current_node)

            if not isinstance(current_node, NeuralNetworkOutputNeuron):
                for connection in current_node.connections:
                    open_list.put(connection)

        return True


if __name__ == "__main__":
    nn = NeuralNetwork(3, 2, struct_mut_new_rate=1)
    nn.build_network(['a', 'b', 'c'], ['p', 'q'])
    g_innov = nn.innovation_number
    g_genome = nn.genome
    
    a = nn.feed_forward([1, 2, 3])
    
    for out in a:
        print(a[out])
        
    g_innov = nn.structural_mutation(g_innov, g_genome)

    print()
    a = nn.feed_forward([1, 2, 3])
    
    for out in a:
        print(a[out])

    print(nn.genome)
    
