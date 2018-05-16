# Originally written by Kyle Chickering, Taaha Chaudhry, Xin Jin, Alex Mirov and
# Simon Wu for ECS 170 @ UC Davis, Spring 2018.
#

"""
nn3.py

Implements a non-recurrent, explicit and topologically unconstrained neural
network.
"""

MAX_NETWORK_SIZE = 1000

import math
import copy
import queue
import random

import sys
sys.path.insert(0, './../')

from nn.activation import ActivationFunction

class NeuralNetwork(object):
    """ Build a neural network with a static number of inputs and outputs """

    def __init__(self, input_size, output_size, input_labels, output_labels,
                 output_function_type='tanh', hidden_function_type='tanh',
                 learning_constant=1, struct_mut_con_rate=0.5,
                 struct_mut_new_rate=0.5, struct_mut_rm_rate=0.5,
                 n_struct_mut_rate=0.001):
        """ Configures network parameters for this neural network. 

        input_size: The number of inputs to the neural network.
        output_size: The number of outputs from the neural network.
        input_labels: A list of input_size consisting of labels for the input
          nodes.
        output_labels: A list of output_size consisting of labels for the output
          nodes.
        output_function_type: The activation function type for the output layer
          neurons. Avaliable function types can be found in nn/activation.py.
        hidden_function_type: The activation function type for the hidden layer
          neurons. Avaliable function types can be found in nn/activation.py.
        learning_constant:
        struct_mut_con_rate: A number between 0 and 1 specifing the rate at
          which new connections form in the network. A 0 is for new connections
          never being built, and a 1 guarenting a new connection if a new
          connection is possible.
        struct_mut_new_rate: A number between 0 and 1 specifing the rate at 
          which new neurons are added to the network. A 0 adds no new nodes to
          the network, a 1 gaurentees a new node is added at each structural
          mutation.
        struct_mut_rm_rate: This parameter is not currently implemented. 
          Specifies the rate at which nodes can be spontaniously removed from
          the network. A number between 0 and 1. 0 is no removal, 1 is a 
          gaurenteed removal each structural mutation.
        n_struct_mut_rate:

        return: None """
        
        self.g_dict = {}

        self.input_size = input_size
        self.input_labels = input_labels
        self.output_size = output_size
        self.output_labels = output_labels

        self.h_function = hidden_function_type
        self.o_function = output_function_type

        self.node_id = 0

        self.struct_mut_con_rate = struct_mut_con_rate
        self.struct_mut_new_rate = struct_mut_new_rate
        self.struct_mut_rm_rate = struct_mut_rm_rate
        self.n_struct_mut_rate = n_struct_mut_rate
        self.n_struct_mut_minimizer = 0.5

        self.learning_constant=learning_constant

        return


    def print_connections(self):
        """ Prints all of the connections in the neural network.

        return: None """
        
        cons = {}
        for node in self.g_dict:
            cons[node] = []
            if self.g_dict[node]['connections']:
                for connection in self.g_dict[node]['connections'].keys():
                    if self.g_dict[node]['connections'][connection]:
                        cons[node].append(connection)

        print(cons)
        return

    
    def bfs_reorder(self):
        """ Walks through the graph and modifies neuron IDs to be in a breadth
        first ordering. This method modifies the network in place.

        return: None """
        
        r_map = {}
        o_map = {}
        n_id = 0
        o_id = MAX_NETWORK_SIZE
        n_g_dict = {}

        inputs = []
        for nn in self.g_dict.keys():
            if self.g_dict[nn]['type'] == 'input':
                inputs.append(nn)

        o_list = queue.Queue()
        seen = []

        for ii in inputs:
            o_list.put(ii)

        while not o_list.empty():
            c_id = o_list.get()

            if c_id not in seen:
                if self.g_dict[c_id]['type'] == 'output':
                    r_map[o_id] = c_id
                    o_map[c_id] = o_id
                    n_g_dict[o_id] = {
                        'type': self.g_dict[c_id]['type'],
                        'label': self.g_dict[c_id]['label'],
                        'connections': None,
                        'function': self.g_dict[c_id]['function']
                    }
                    g_id = o_id
                    o_id += 1
                else:
                    r_map[n_id] = c_id
                    o_map[c_id] = n_id
                    n_g_dict[n_id] = {
                        'type': self.g_dict[c_id]['type'],
                        'label': self.g_dict[c_id]['label'],
                        'connections': None,
                        'function': self.g_dict[c_id]['function']
                    }
                    g_id = n_id
                    n_id += 1

                if self.g_dict[c_id]['connections']:
                    for con in self.g_dict[c_id]['connections']:
                        o_list.put(con)

                seen.append(c_id)
                
                if n_id >= MAX_NETWORK_SIZE - self.output_size:
                    raise Exception("Network has exceeded max size")

        for nn in n_g_dict:
            n_cons = {}
            if self.g_dict[r_map[nn]]['connections']:
                for con in self.g_dict[r_map[nn]]['connections']:
                    n_cons[o_map[con]] = self.g_dict[r_map[nn]]['connections'][con]
                
                n_g_dict[nn]['connections'] = n_cons
            else:
                n_g_dict[nn]['connections'] = None

        self.g_dict = n_g_dict

        return


    def build_network(self):
        """ Builds the initial network from the parameters specified in 
        __init__. Adds the inputs and outputs, labels them, and builds a fully
        connected neural network. 
       
        return: None """
        
        for ii in range(self.output_size):
            label = self.output_labels[ii]

            self.g_dict[self.node_id] = {
                'type': 'output',
                'label': label,
                'connections': None,
                'function': ActivationFunction(function=self.o_function)
            }

            self.node_id += 1
        
        for ii in range(self.input_size):
            label = self.input_labels[ii]
            cons = {}

            for nn in self.g_dict.keys():
                if self.g_dict[nn]['type'] == 'output':
                    cons[nn] = {
                        'weight': random.gauss(0, 1),
                        'enabled': True
                    }

            self.g_dict[self.node_id] = {
                'type': 'input',
                'label': label,
                'connections': cons,
                'function': ActivationFunction(function='identity')
            }

            self.node_id += 1

        return


    def feed_forward(self, i_dict):
        """ Feed values forward through the neural network. Uses a breadth-first
        flood through the graph to calculate the output values for the network's
        output neurons. 
        
        i_dict: A dictionary containing input labels and the associated input 
          value for each label. These are the input values that will be fed 
          forward.

        return: A dictionary of output labels and their associated feed-forward
          result. """
        
        inputs = []
        ff_dict = self.g_dict
        for nn in self.g_dict.keys():
            ff_dict[nn]['value'] = 0

            if ff_dict[nn]['type'] == 'input':
                ff_dict[nn]['value'] = i_dict[ff_dict[nn]['label']]
                inputs.append(nn)

        outputs = {}
        o_list = queue.Queue()
        seen = []

        for ii in inputs:
            o_list.put(ii)

        while not o_list.empty():
            cur_id = o_list.get()

            if cur_id not in seen:
                cur = ff_dict[cur_id]
                activation = cur['function'].evaluate(cur['value'])

                if cur['type'] == 'output':
                    if cur['label']:
                        outputs[cur['label']] = activation
                    else:
                        outputs[cur_id] = activation

                else:
                    for con in cur['connections']:
                        c_con = cur['connections'][con]
                        if c_con['enabled']:
                            o_list.put(con)
                            ff_dict[con]['value'] += activation*c_con['weight']

                seen.append(cur_id)

        return outputs


    def structural_mutation(self):
        """ Adds structural mutations to the neural networks. Structural 
        mutations include adding new connections to the network and adding new
        nodes to the network. Eventually structural mutations may also include
        removing nodes from the network as well. Stuctural mutations occur in
        place and will modify the underlying network.

        return: None """
        
        # Add connections if connections are possible to add
        if random.uniform(0, 1) <= self.struct_mut_con_rate:
            depths = self.dfs()
            node = random.choice(list(self.g_dict.keys()))

            while self.g_dict[node]['type'] == 'output':
                node = random.choice(list(self.g_dict.keys()))

            choices = list(self.g_dict.keys())
            choice = random.choice(choices)

            while self.g_dict[choice]['type'] == 'output' or \
                  depths[node] >= depths[choice]:
                choice = random.choice(choices)
                choices.remove(choice)
                if len(choices) == 0:
                    break
            
            if len(choices) <= 0 or depths[node] >= depths[choice]:
                # Totally connected
                pass
            elif node != choice:
                if self.g_dict[node]['connections']:
                    if not choice in self.g_dict[node]['connections'].keys():
                        self.g_dict[node]['connections'][choice] = {
                            'weight': random.gauss(0, 1),
                            'enabled': True
                        }

        # Add new neurons to the neural network
        if random.uniform(0, 1) <= self.struct_mut_new_rate:
            choice = random.choice(list(self.g_dict.keys()))

            while self.g_dict[choice]['type'] == 'output':
                choice = random.choice(list(self.g_dict.keys()))

            n_con = random.choice(list(self.g_dict[choice]['connections'].keys()))

            n_id = self.node_id

            self.g_dict[n_id] = {
                'type': 'hidden',
                'label': None,
                'connections': {
                    n_con: self.g_dict[choice]['connections'][n_con]
                },
                'function': ActivationFunction(self.h_function)
            }

            self.g_dict[choice]['connections'].pop(n_con)
            self.g_dict[choice]['connections'][n_id] = {
                'weight' : 1,
                'innovation': 0,
                'enabled' : True,
            }

            self.node_id += 1
        
        return


    def non_structural_mutation(self):
        """ Adds non-structural mutations to the network. Non-structural 
        mutations are manipulations of the weights between neurons. This is 
        achieved by adding values from a guassian distribution centered at 0 to
        the existing weight at a rate proportional to the n_struct_mut_rate.
        
        return: None """
        
        for n_id in self.g_dict:
            cons = self.g_dict[n_id]['connections']
            if cons:
                for con in cons:
                    if random.uniform(0, 1) <= self.n_struct_mut_rate:
                        cons[con]['weight'] += random.gauss(0, 1
                    )

        return


    def breed(self, mate, fit_me, fit_mate):
        """ Breed this neural network with a mate to produce a neural network
        offspring. This is not the most efficient evolutionary breeding 
        algorithm. This algorithm clones the more fit parent, and splices in 
        missing connections from the less fit parent, as long as those 
        connections do not create cycles in the neural network.
        
        mate: The neural network that this network will breed with.
        fit_me: The fitness of this network.
        fit_mate: The fitness of the mate network.

        return: A new child neural network bred from the two parents. """
        
        m_fit = l_fit = None

        if fit_mate < fit_me:
            m_fit = self
            l_fit = mate
        else:
            m_fit = mate
            l_fit = self

        child = NeuralNetwork(self.input_size, self.output_size,
                            self.input_labels, self.output_labels,
                            output_function_type=self.o_function,
                            hidden_function_type=self.h_function,
                            learning_constant=self.learning_constant,
                            struct_mut_con_rate=self.struct_mut_con_rate,
                            struct_mut_new_rate=self.struct_mut_new_rate,
                            n_struct_mut_rate=self.n_struct_mut_rate)

        m_fit.bfs_reorder()
        l_fit.bfs_reorder()

        m_fit_depths = m_fit.dfs()
        l_fit_depths = l_fit.dfs()

        child.g_dict = copy.deepcopy(m_fit.g_dict)

        for nn in l_fit.g_dict.keys():
            if nn in child.g_dict.keys() and l_fit.g_dict[nn]['connections']:
                for con in l_fit.g_dict[nn]['connections']:
                    if con in child.g_dict[nn]['connections']:
                        child.g_dict[nn]['connections'][con]['weight'] += l_fit.g_dict[nn]['connections'][con]['weight']
                        child.g_dict[nn]['connections'][con]['weight'] /= 2
                    else:
                        if con in child.g_dict.keys() and \
                           child.g_dict[con]['connections'] and \
                           nn in child.g_dict[con]['connections'].keys():
                            child.g_dict[con]['connections'][nn]['weight'] -= l_fit.g_dict[nn]['connections'][con]['weight']
                            child.g_dict[con]['connections'][nn]['weight'] /= 2
                        else:

                            if (con in m_fit_depths.keys() and \
                                nn in m_fit_depths.keys() and \
                                m_fit_depths[con] > m_fit_depths[nn] and \
                                l_fit_depths[con] > l_fit_depths[nn]) and \
                                (con in m_fit_depths.keys() and \
                                 m_fit_depths[con] > l_fit_depths[nn]) or \
                                (con not in m_fit_depths.keys() and \
                                 l_fit_depths[con] > l_fit_depths[nn]):
                                
                                child.g_dict[nn]['connections'][con] = l_fit.g_dict[nn]['connections'][con]
            else:
                if l_fit.g_dict[nn]['connections']:
                    n_cons = {}
                    for con in l_fit.g_dict[nn]['connections'].keys():
                        if (not con in child.g_dict.keys() or \
                           not child.g_dict[con]['connections'] or \
                           nn not in child.g_dict[con]['connections'].keys()):
                            
                            n_cons[con] = l_fit.g_dict[nn]['connections'][con]
                    
                    child.g_dict[nn] = {
                        'type': None,
                        'label': None,
                        'connections': n_cons,
                        'funciton': None
                    }

        o_count = 0
        for nn in child.g_dict.keys():
            if not child.g_dict[nn]['connections']:
                child.g_dict[nn]['type'] = 'output'
                child.g_dict[nn]['label'] = child.output_labels[o_count]
                o_count += 1
                child.g_dict[nn]['function'] = ActivationFunction(
                    function=child.o_function
                )
                continue

            if nn < child.input_size:
                child.g_dict[nn]['type'] = 'input'
                child.g_dict[nn]['label'] = child.input_labels[nn]
                child.g_dict[nn]['function'] = ActivationFunction(
                    function='identity'
                )
                continue

            child.g_dict[nn]['type'] = 'hidden'
            child.g_dict[nn]['function'] = ActivationFunction(
                function=child.h_function
            )

        # Set child node count to the correct number
        child.node_id = len(child.g_dict.keys())
        
        return child


    def dfs(self):
        """ Depth first search through the neural network. This method does not
        account for cyclic networks, and the recursion will hit the recursive
        limit. Generates a mapping of the node ids and their depth. Calls the
        _dfs method recursivly.

        return: A dictionary containing node ids and their depths in the neural
          network. """
        
        depths = {}
        
        for con in self.g_dict.keys():
            if self.g_dict[con]['type'] == 'input':
                depths = self._dfs(con, depths, 0)

        return depths


    def _dfs(self, node, depths, current_depth):
        """ Recursive call for depth first search. Calls itself on each of the
        connections to the current node. 
        
        node: The current node that we are looking at.
        depths: The depth dictionary.
        current_depth: The current depth of the node we are currently looking 
          at.

        return: The updated depths dictionary. """
        
        if node not in depths:
            depths[node] = current_depth
        else:
            if depths[node] < current_depth:
                depths[node] = current_depth

        if not self.g_dict[node]['connections']:
            return depths

        if self.g_dict[node]['type'] == 'output':
            return depths

        for con in self.g_dict[node]['connections'].keys():
            if self.g_dict[node]['connections'][con]['enabled']:
                self._dfs(con, depths, current_depth + 1)

        return depths


    def detect_cycle(self):
        """ Detects cycles in the neural network by recursivly calling 
        _detect_cycle. """
        
        for con in self.g_dict.keys():
            if self.g_dict[con]['type'] == 'input':
                if self._detect_cycle(con, [con]):
                    return True

        return False


    def _detect_cycle(self, node, callers):
        """ Recursivly examines the network looking for cycles. """
        
        if not self.g_dict[node]['connections']:
            return False

        if self.g_dict[node]['type'] == 'ouput':
            return True

        for con in self.g_dict[node]['connections'].keys():
            if self.g_dict[node]['connections'][con]['enabled']:
                if con in callers:
                    return True
                else:
                    callers.append(con)
                    if self._detect_cycle(con, callers):
                        return True
                    
                    del callers[-1]

        return False
