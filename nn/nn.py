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

from activation import ActivationFunction

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


    def build_network_from_genome(self, genome, input_labels, output_labels,
                                  connection_weights):
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

                connection_dict[key[0]][0].add_connection(hidden,
                                                          connection_weights[key])
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
        # TODO: Mutate in connections
        # if random.uniform(0, 1) <= self.struct_mut_con_rate:
        #     # If we can add a connection, then do so
        #     connectable_nodes = self.inputs + self.hidden
        #     node = connected_nodes[random.randint(0, len(connected_nodes) - 1)]
        #     choice = random.choice(self.hidden + self.outputs)
        # 
        #     if choice not in list(node.connections.keys()):
        #         node.add_connection(choice, random.uniform(-1, 1))
        # 
        #         # Add the new connection to the genome
        #         connection_id = (node.node_id, choice.node_id)
        #         if connection_id in generation.genome.keys():
        #             new_gene = (generation_genome[key], False)
        #         else:
        #             new_gene = (global_innovation, False)
        #             global_innovation += 1
        #             generation_genome[connection_id] = new_gene
        # 
       #         self.genome[connection_id] = new_gene

        if random.uniform(0, 1) <= self.struct_mut_new_rate:
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

        connection_weights = {} # (a, b): weight
        for node in mate.inputs + mate.hidden:
            for connection in node.connections.keys():
                connection_weights[(node.node_id,
                                    connection.node_id)] = node.connections[connection]
        for node in self.inputs + self.hidden:
            for connection in node.connections.keys():
                connection_weights[(node.node_id,
                                    connection.node_id)] = node.connections[connection]


        child.build_network_from_genome(child_genome,
                                        more_fit.input_labels,
                                        more_fit.output_labels,
                                        connection_weights)

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


class NeuralNetwork2(object):
    """
    Second generation implementation of NeuralNetwork. This version uses
    a much more efficient data structure representation which makes it much
    faster and less confusing.
    """
    # TODO: Use Guassian Distributions instead of uniform distributions 
    def __init__(self, input_size, output_size, input_labels, output_labels,
                 output_function_type='tanh', hidden_function_type='tanh',
                 learning_constant=1, struct_mut_con_rate=0.5,
                 struct_mut_new_rate=0.5, n_struct_mut_rate=0.001):
        """
        """
        self.connection_list = {}
        # id : {type: , label: , connections: {id: {weight: , innovation: , enabled: }} , function: }

        self.input_size = input_size
        self.input_labels = input_labels
        self.output_size = output_size
        self.output_labels = output_labels

        self.h_function = hidden_function_type
        self.o_function = output_function_type

        self.genome = {} # Best data stucture for a genome?
        self.innovation_number = 0
        self.node_id = 0

        self.struct_mut_con_rate = struct_mut_con_rate
        self.struct_mut_new_rate = struct_mut_new_rate
        self.n_struct_mut_rate = n_struct_mut_rate
        self.n_struct_mut_minimizer = 0.25

        self.learning_constant=learning_constant

    def inci(self):
        """
        """
        innov = self.innovation_number
        self.innovation_number += 1
        return innov

        
    def build_network(self, generation_genome,
                      generation_innov):
        """
        Builds a network with all of the inputs connected to all of the outputs.
        """
        for i in range(self.output_size):
            label = self.output_labels[i]

            self.connection_list[self.node_id] = {
                'type': 'output',
                'label': label,
                'connections': None,
                'function': ActivationFunction(function=self.o_function)
            }
            self.node_id += 1

        for i in range(self.input_size):
            label = self.input_labels[i]

            connections = {}
            for node in self.connection_list.keys():
                if self.connection_list[node]['type'] == 'output':
                    connections[node] = {
                        'weight': random.uniform(-1, 1),
                        'innovation': self.inci(),
                        'enabled': True
                    }

                    if (self.node_id, node) in generation_genome.keys():
                        gene = generation_genome[(self.node_id, node)]
                        self.genome[(self.node_id, node)] = gene
                    else:
                        gene = (generation_innov, True)
                        generation_genome[(self.node_id, node)] = gene
                        self.genome[(self.node_id, node)] = gene
                        generation_innov += 1
                
            self.connection_list[self.node_id] = {
                'type': 'input',
                'label': label,
                'connections': connections,
                'function': ActivationFunction(function='identity')
            }

            self.node_id += 1

        return (generation_genome, generation_innov)


    def feed_forward(self, input_dict):
        """
        """
        inputs = []
        ff_list = self.connection_list
        for node in self.connection_list.keys():
            ff_list[node]['value'] = 0
            
            if ff_list[node]['type'] == 'input':
                ff_list[node]['value'] = input_dict[ff_list[node]['label']]
                inputs.append(node)

        outputs = {}
        open_list = queue.Queue()
        seen = []
        
        for i in inputs:
            open_list.put(i)

        while not open_list.empty():
            curr_id = open_list.get()
            
            if curr_id not in seen:
                current = ff_list[curr_id]
                activation = current['function'].evaluate(current['value'])
                
                if current['type'] == 'output':
                    if current['label']:
                        outputs[current['label']] = activation
                    else:
                        outputs[curr_id] = activation
                
                else:
                    for con in current['connections']:
                        curr_con = current['connections'][con]
                        if curr_con['enabled']:
                            open_list.put(con)
                            ff_list[con]['value'] += activation*curr_con['weight']

                seen.append(curr_id)
                
        return outputs


    def back_propogate(self):
        pass


    def structural_mutation(self, generation_genome, generation_innov):
        """
        Add connections if we can and add new nodes
        """
        if random.uniform(0, 1) <= self.struct_mut_new_rate:
            choice = random.choice(list(self.connection_list.keys()))

            # Don't choose an output node
            while self.connection_list[choice]['type'] == 'output':
                choice = random.choice(list(self.connection_list.keys()))

            self.connection_list[self.node_id] = {
                'type': 'hidden',
                'label': None,
                'connections': None,
                'function': ActivationFunction(self.h_function)
            }

            # TODO: Choice has no connections and is assigning None to the new hidden
            if not self.connection_list[choice]['connections']:
                print(choice)
                print(self.connection_list)
                print("Error")
                exit(1)
                
            self.connection_list[self.node_id]['connections'] = self.connection_list[choice]['connections']
            # remove the connection from the genome

            # TODO: Re-factor with a check-genome method
            for c in self.connection_list[choice]['connections']:
                gene = (choice, c)
                if gene in generation_genome.keys():
                    innov_number = generation_genome[gene][0]
                else:
                    generation_innov += 1
                    innov_number = generation_innov
                    generation_genome[gene] = (innov_number, False)
                
                self.genome[gene] = (innov_number, False)

                gene = (self.node_id, c)
                if gene in generation_genome.keys():
                    innov_number = generation_genome[gene][0]
                else:
                    generation_innov += 1
                    innov_number = generation_innov
                    generation_genome[gene] = (innov_number, True)
                
                self.genome[gene] = (innov_number, True)

            gene = (choice, self.node_id)
            if gene in generation_genome.keys():
                innov_number = generation_genome[gene][0]
            else:
                generation_innov += 1
                innov_number = generation_innov
                generation_genome[gene] = (innov_number, True)
                
            self.genome[(choice, self.node_id)] = (innov_number, True)
            
            if (choice, self.node_id) in generation_genome.keys():
                pass
            
            self.connection_list[choice]['connections'] = {
                self.node_id: {
                    'weight' : 1,
                    'innovation': 0,
                    'enabled' : True,
                    }
                }

            self.node_id += 1

        return (generation_genome, generation_innov)


        # Only add "Up-stream" connections BFS through, stop on the node we are
        # adding a connection to, then only add connections to nodes we haven't
        # seen yet


    def non_structural_mutation(self):
        """
        Manipulate the weights of some of the connections. We do not add
        connections to the genome.
        """
        for n_id in self.connection_list:
            connections = self.connection_list[n_id]['connections']
            if connections:
                for connec in connections:
                    if random.uniform(0, 1) <= self.n_struct_mut_rate:
                        connections[connec]['weight'] += random.uniform(
                            -1 - connections[connec]['weight'],
                            1 - connections[connec]['weight']
                        )*self.n_struct_mut_minimizer

        return


    def breed(self, mate, fitness_this, fitness_mate, generation_genome,
              generation_innov):
        more_fit = less_fit = None

        if fitness_this > fitness_mate:
            more_fit = self
            less_fit = mate
        else:
            more_fit = mate
            less_fit = self

        child = NeuralNetwork2(self.input_size, self.output_size,
                               self.input_labels, self.output_labels,
                              output_function_type=self.o_function,
                              hidden_function_type=self.h_function,
                              learning_constant=self.learning_constant,
                              struct_mut_con_rate=self.struct_mut_con_rate,
                              struct_mut_new_rate=self.struct_mut_new_rate,
                              n_struct_mut_rate=self.n_struct_mut_rate)

        child.build_network(generation_genome, generation_innov)

        child_genome = child.genome

        for g in less_fit.genome:
            if g in more_fit.genome:
                gene = more_fit.genome[g]
            else:
                gene = less_fit.genome[g]

            if gene[1]:
                if g in more_fit.genome:
                    weight = more_fit.connection_list[g[0]]['connections'][g[1]]['weight']
                else:
                    weight = less_fit.connection_list[g[0]]['connections'][g[1]]['weight']
                if g in generation_genome:
                    new_gene = generation_genome[g]
                else:
                    new_gene = (generation_innov, True)
                    generation_innov += 1
                    generation_genome[g] = new_gene

                child_genome[g] = new_gene

                if g[0] not in child.connection_list:
                    child.connection_list[g[0]] = {
                        'type': 'hidden',
                        'label': None,
                        'connections': {},
                        'function': ActivationFunction(child.h_function)
                    }

                if g[1] not in child.connection_list[g[0]]['connections']:
                    child.connection_list[g[0]]['connections'][g[1]] = None

                child.connection_list[g[0]]['connections'][g[1]] = {
                    'weight': weight,
                    'innovation': new_gene[0],
                    'enabled': True
                }

        for g in more_fit.genome:
            if g not in child_genome:
                gene = more_fit.genome[g]

            if gene[1]:
                if g not in child_genome:
                    weight = more_fit.connection_list[g[0]]['connections'][g[1]]['weight']
                
                if g in generation_genome:
                    new_gene = generation_genome[g]
                else:
                    new_gene = (generation_innov, True)
                    generation_innov += 1
                    generation_genome[g] = new_gene

                child_genome[g] = new_gene

                if g[0] not in child.connection_list:
                    child.connection_list[g[0]] = {
                        'type': 'hidden',
                        'label': None,
                        'connections': {},
                        'function': ActivationFunction(child.h_function)
                    }

                if g[1] not in child.connection_list[g[0]]['connections']:
                    child.connection_list[g[0]]['connections'][g[1]] = None

                child.connection_list[g[0]]['connections'][g[1]] = {
                    'weight': weight,
                    'innovation': new_gene[0],
                    'enabled': True
                }

        return (generation_genome, generation_innov, child)


    def save_nn(self):
        pass

    
if __name__ == "__main__":
    import sys
    sys.path.insert(0, './../')
    from ga.individual import Individual
    from ga.neat_evolution import EvolutionaryController

    nn = NeuralNetwork2(2, 2, ['a', 'b'], ['c', 'd'],
                        n_struct_mut_rate=1, struct_mut_new_rate=1)
    gg = {}
    gi = 0
    (gg, gi) = nn.build_network( gg, gi)
    print(nn.feed_forward({'a': 1, 'b': 2}))
    nn.non_structural_mutation()
    (gg, gi) = nn.structural_mutation(gg, gi)
    print(nn.feed_forward({'a': 1, 'b': 2}))

    nn2 = NeuralNetwork2(2, 2, ['a', 'b'], ['c', 'd'],
                        n_struct_mut_rate=1, struct_mut_new_rate=1)
    (gg, gi) = nn2.build_network(gg, gi)
    for i in range(10):
        nn.non_structural_mutation()
        (gg, gi) = nn2.structural_mutation(gg, gi)
        nn.non_structural_mutation()
        (gg, gi) = nn2.structural_mutation(gg, gi)

    new_gg = {}
    new_gi = 0
    (new_gg, new_gi, child) = nn.breed(nn2, 1, 2, new_gg, new_gi)
    print(child.feed_forward({'a': 1, 'b': 2}))

    class NEATNN2(object):
        def __init__(self):
            self.network = None


        def individual_print(self):
            return ""

        
        def fitness_function(self):
            fitness = 0
            results = [list(self.network.feed_forward(a).values())[0] for a in [{'a': 0,'b': 0}, {'a': 0, 'b':1}, {'a':1, 'b':0}, {'a': 1, 'b':1}]]
            actual = [0, 1, 1, 0]
            for i in range(len(actual) - 1):
                fitness += abs(actual[i] - results[i])

            return fitness


        def breed_parents(self, parent_tuple, child, reproduction_constant, generation_genome, generation_innov):
            (generation_genome, generation_innov,child.network) = parent_tuple[0][1].network.breed(
                parent_tuple[1][1].network,
                parent_tuple[0][0],
                parent_tuple[1][0], generation_genome, generation_innov
            )

            return (generation_genome, generation_innov)

        
        def mutate(self, mutation_constant, generation_genome, generation_innov):
            (generation_genome, generation_innov) = self.network.structural_mutation(generation_genome, generation_innov)
            self.network.non_structural_mutation()

            return (generation_genome, generation_innov)

        
        def generate_random(self, generation_genome, generation_innov):
            self.network = NeuralNetwork2(2, 1, ['a', 'b'], ['o'],
                                          struct_mut_new_rate=1,
                                          struct_mut_con_rate=1,
                                          n_struct_mut_rate=1)
            
            return self.network.build_network(generation_genome, generation_innov)
    

    EC = EvolutionaryController(NEATNN2)
    EC.run_evolution(15, 1, 1, printing='full', generation_limit=100)
    
    # class NEATNN(Individual):
    # 
    #     def __init__(self):
    #         self.network = None
    #         
    #     def individual_print(self):
    #         return ""
    # 
    # 
    #     def fitness_function(self):
    #         fitness = 0
    #         results = [list(self.network.feed_forward(a).values())[0] for a in [[0,0], [0, 1], [1, 0], [1, 1]]]
    #         actual = [0, 1, 1, 0]
    #         for i in range(len(actual) - 1):
    #             fitness += abs(actual[i] - results[i])
    # 
    #         return fitness
    # 
    # 
    #     def breed_parents(self, parent_tuple, child, reproduction_constant):
    #         child.network = parent_tuple[0][1].network.breed(
    #             parent_tuple[1][1].network,
    #             parent_tuple[0][0],
    #             parent_tuple[1][0]
    #         )
    # 
    #         return child
    # 
    # 
    #     def mutate(self, mutation_constant):
    #         self.network.structural_mutation(self.network.innovation_number,
    #                                  self.network.genome)
    #         self.network.non_structural_mutation()
    # 
    # 
    #     def generate_random(self):
    #         self.network = NeuralNetwork(2, 1, struct_mut_new_rate=1, struct_mut_con_rate=1, n_struct_mut_rate=1)
    #         self.network.build_network(['a', 'b'], ['o'])
    # 
    # 
    # EC = EvolutionaryController(NEATNN)
    # EC.run_evolution(15, 1, 1, printing='full', generation_limit=100)
