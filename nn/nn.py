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
        self.input_labels = []
        
        self.outputs = []
        self.output_size = output_layer_size
        self.output_labels = []
        
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
        self.total_nodes += 1
        return self.total_nodes

        
    def build_network(self, input_labels, output_labels):
        """
        Builds a network from the input and output labels. Used for creating an
        initial neural network from scratch.
        
        inputs_labels:
        output_labels:
        return: None
        """
        self.input_labels = input_labels
        self.output_labels = output_labels
        
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


    def build_network_from_genome(self, genome, input_labels, output_labels):
        """
        Builds a network from a supplied genome.
        """
        self.build_network(input_labels, output_labels)

        connection_dict = {}
        
        for connected in self.inputs + self.hidden:
            if connected.node_id not in connection_dict.keys():
                connection_dict[connected.node_id] = [connected]
                
            for connection in connected.connections:
                connection_dict[connected.node_id].append(connection)
        
        for key in genome.keys():
            if key[0] not in connection_dict.keys():
                hidden = NeuralNetworkHiddenNeuron(self.function, key[0])
                
                connection_dict[key[0]] = [hidden]
            
            if key[1] not in [node.node_id for node in connection_dict[key[0]]]:
                hidden = None
                # node does not exist yet
                if key[1] not in connection_dict.keys():
                    hidden = NeuralNetworkHiddenNeuron(self.function, key[1])
                else:
                    hidden = connection_dict[key[1]][0]

                connection_dict[key[0]][0].add_connection(hidden, 1)
                connection_dict[key[0]].append(hidden)


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
                connection_id = (new_hidden.node_id, connection.node_id)
                new_hidden.add_connection(connection,
                                          node.connections[connection])

                valid_keys = []
                for key in self.genome.keys():
                    if key[1] == connection.node_id:
                        valid_keys.append(key)

                for gene in valid_keys:
                    
                    if connection_id in generation_genome.keys():
                        new_gene = (generation_genome[key], False)
                    else:
                        new_gene = (global_innovation, False)
                        global_innovation += 1
                        generation_genome[connection_id] = new_gene
                        generation_genome[gene] = (generation_genome[gene][0],
                                                   True)

                    self.genome[gene] = (self.genome[gene][0], True)
                    
            node.connections = {new_hidden: 1}

            self.hidden.append(new_hidden)

            # Must update both global and local genomes

            valid_keys = []
            for key in self.genome.keys():
                if key[0] == node.node_id:
                    valid_keys.append(key)
                    
            for gene in valid_keys:
                if gene[0] == node.node_id:
                    key = (node.node_id, new_hidden.node_id)
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

        mate:
        this_fitness:
        mate_fitness:
        return: The offspring of the two parent individuals
        """
        child = NeuralNetwork(self.input_size, self.output_size,
                              hidden_function_type=self.function,
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

        shared = []
        disjoint_more = []
        disjoint_less = []

        for key in more_fit.genome.keys():
            if key in less_fit.genome.keys():
                shared.append(key)
            else:
                disjoint_more.append(key)

        for key in less_fit.genome.keys():
            if key not in more_fit.genome.keys():
                disjoint_less.append(key)

        for key in shared:
            child_genome[key] = more_fit.genome[key]

        for key in disjoint_more:
            child_genome[key] = more_fit.genome[key]

        for key in disjoint_less:
            child_genome[key] = less_fit.genome[key]

        child.build_network_from_genome(child_genome,
                                        more_fit.input_labels,
                                        more_fit.output_labels)

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


    def fitness(self, time_alive, time_frame):
        # Compute funciton keeping in mind time alive, time allowed and
        # penalize individuals that are overly complex.
        return 1


    def print(self):
        pass

    
if __name__ == "__main__":
    import sys
    sys.path.insert(0, './../')
    from ga.individual import Individual
    from ga.evolution import EvolutionaryController
        
    class NEATNN(Individual):

        def __init__(self):
            self.network = None
            
        def individual_print(self):
            return ""
    
    
        def fitness_function(self):
            fitness = 0
            results = [list(self.network.feed_forward(a).values())[0] for a in [[0,0], [0, 1], [1, 0], [1, 1]]]
            actual = [0, 1, 1, 0]
            for i in range(len(actual) - 1):
                fitness += abs(actual[i] - results[i])

            return fitness
    
    
        def breed_parents(self, parent_tuple, child, reproduction_constant):
            child.network = parent_tuple[0][1].network.breed(
                parent_tuple[1][1].network,
                parent_tuple[0][0],
                parent_tuple[1][0]
            )

            return child
    
    
        def mutate(self, mutation_constant):
            self.network.structural_mutation(self.network.innovation_number,
                                     self.network.genome)
    
    
        def generate_random(self):
            self.network = NeuralNetwork(2, 1, struct_mut_new_rate=1)
            self.network.build_network(['a', 'b'], ['o'])

    
    EC = EvolutionaryController(NEATNN)
    EC.run_evolution(15, 1, 1, printing='full', generation_limit=100)
    
    # Back-propogation to do live learning

    # nn1 = NeuralNetwork(3, 2, struct_mut_new_rate=1)
    # nn2 = NeuralNetwork(3, 2, struct_mut_new_rate=1)
    # nn1.build_network(['a', 'b', 'c'], ['p', 'q'])
    # nn2.build_network(['a', 'b', 'c'], ['p', 'q'])
    # 
    # nn1.structural_mutation(nn1.innovation_number, nn1.genome)
    # 
    # nnc = nn1.breed(nn2, 1, 2)
    # 
    # a = nnc.feed_forward([1, 2, 3])
    # for out in a:
    #     print(a[out])
    
    # nn = NeuralNetwork(3, 2, struct_mut_new_rate=1)
    # nn.build_network(['a', 'b', 'c'], ['p', 'q'])
    # g_innov = nn.innovation_number
    # g_genome = nn.genome
    # 
    # a = nn.feed_forward([1, 2, 3])
    # 
    # for out in a:
    #     print(a[out])
    #     
    # g_innov = nn.structural_mutation(g_innov, g_genome)
    # 
    # print()
    # a = nn.feed_forward([1, 2, 3])
    # 
    # for out in a:
    #     print(a[out])
    # 
    # print(nn.genome)

    # nn = NeuralNetwork(30, 30, struct_mut_new_rate=1)
    # nn.build_network(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    #                   'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    #                   't', 'u', 'v', 'w', 'x', 'y', 'z', 'aa', 'ab', 'ac',
    #                   'ad'],
    #                  ['ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg', 'bh', 'bi',
    #                   'bj', 'bk', 'bl', 'bm', 'bn', 'bo', 'bp', 'bq', 'br', 'bs',
    #                   'bt', 'bu', 'bv', 'bw', 'bx', 'by', 'bz', 'baa', 'bab', 'bac',
    #                   'bad'])
    # g_innov = nn.innovation_number
    # g_genome = nn.genome
    # 
    # a = nn.feed_forward([1 for i in range(29)])
    # 
    # for out in a:
    #     print(a[out])
    #     
    # g_innov = nn.structural_mutation(g_innov, g_genome)
    # 
    # print()
    # a = nn.feed_forward([1 for i in range(29)])
    # 
    # for out in a:
    #     print(a[out])
    # 
    # print(nn.genome)
    
