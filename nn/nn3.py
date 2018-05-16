# Originally written by Kyle Chickering, Taaha Chaudhry, Xin Jin, Alex Mirov and
# Simon Wu for ECS 170 @ UC Davis, Spring 2018.
#

"""
nn.py

Implements a non-recurrent, NEAT compatiable neural network. NEAT was created by
Kenneth Stanley and Risto Miikkulainen @ University of Texas, Austin.
"""

MAX_NETWORK_SIZE = 1000

import math
import copy
import queue
import random

from activation import ActivationFunction

class NeuralNetwork(object):

    def __init__(self, input_size, output_size, input_labels, output_labels,
                 output_function_type='tanh', hidden_function_type='tanh',
                 learning_constant=1, struct_mut_con_rate=0.5,
                 struct_mut_new_rate=0.5, struct_mut_rm_rate=0.5,
                 n_struct_mut_rate=0.001):
        """
        """
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
        """
        Renumber and parse the graph.
        """
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
        if random.uniform(0, 1) <= self.struct_mut_con_rate:
            #self.print_connections()
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
                        # print(depths[node], depths[choice])
                        self.g_dict[node]['connections'][choice] = {
                            'weight': random.gauss(0, 1),
                            'enabled': True
                        }
                        # print("Before")
                        # self.dfs()
                        # print("After")
                        pass
                # else:
                #     self.g_dict[node]['connections'] = {
                #         choice: {
                #             'weight': random.uniform(0, 1),
                #             'enabled': True
                #         }
                #     }
                pass

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
        for n_id in self.g_dict:
            cons = self.g_dict[n_id]['connections']
            if cons:
                for con in cons:
                    if random.uniform(0, 1) <= self.n_struct_mut_rate:
                        cons[con]['weight'] += random.gauss(0, 1
                    )
                        # cons[con]['weight'] *= random.uniform(-1, 1)
                        # cons[con]['weight'] /= 2

        return


    def breed(self, mate, fit_me, fit_mate):
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

        #child.build_network()

        m_fit.bfs_reorder()
        l_fit.bfs_reorder()

        m_fit_depths = m_fit.dfs()
        l_fit_depths = l_fit.dfs()

        ### ________________________________________________________________ ###

        # TODO: DFS when adding nodes to make sure we don't cycle
        # 11 -> 16
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


        ### ________________________________________________________________ ###

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

        # print("Checking")
        # m_fit.print_connections()
        # l_fit.print_connections()
        # child.print_connections()
        # child.dfs()
        # print("PASS")
        # child.bfs_reorder()
        child.node_id = len(child.g_dict.keys())
        
        return child


    def dfs(self):
        depths = {}
        
        for con in self.g_dict.keys():
            if self.g_dict[con]['type'] == 'input':
                depths = self._dfs(con, depths, 0)

        return depths


    def _dfs(self, node, depths, current_depth):
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
        for con in self.g_dict.keys():
            if self.g_dict[con]['type'] == 'input':
                if self._detect_cycle(con, [con]):
                    return True

        return False


    def _detect_cycle(self, node, callers):
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


if __name__ == "__main__":
    import sys
    sys.path.insert(0, './../')

    from ga.individual import Individual
    from ga.evolution import EvolutionaryController

    #np.random.seed(1)
    #random.seed(1)

    nn = NeuralNetwork(2, 2, ['a', 'b'], ['c', 'd'],
                        n_struct_mut_rate=1, struct_mut_new_rate=1)
    
    nn.build_network()
    print(nn.feed_forward({'a': 1, 'b': 2}))
    nn.non_structural_mutation()
    nn.structural_mutation()
    print(nn.feed_forward({'a': 1, 'b': 2}))

    nn2 = NeuralNetwork(2, 2, ['a', 'b'], ['c', 'd'],
                        n_struct_mut_rate=1, struct_mut_new_rate=1,
                        struct_mut_con_rate=1)
    nn2.build_network()
    for i in range(10):
        nn.non_structural_mutation()
        nn2.structural_mutation()
        nn.non_structural_mutation()
        nn2.structural_mutation()
        nn.dfs()
        nn2.dfs()

    child = nn.breed(nn2, 1, 2)
    print(child.feed_forward({'a': 1, 'b': 2}))


    class NEATNN(object):
        def __init__(self):
            self.network = None


        def individual_print(self):
            return ""

        
        def fitness_function(self):
            fitness = 0
            results = [list(self.network.feed_forward(a).values())[0] for a in [{'a': 0,'b': 0}, {'a': 0, 'b':1}, {'a':1, 'b':0}, {'a': 1, 'b':1}]]
            actual = [0, 1, 1, 0]
            for i in range(len(actual)):
                fitness += (abs(actual[i] - results[i]))**2

            #a = (1 - 1/self.network.node_id)
            if self.network.node_id > 5:
                fitness += 30*self.network.node_id

            if self.network.node_id < 5:
                fitness += 10*(5 - self.network.node_id)
            return fitness


        def breed_parents(self, parent_tuple, child, reproduction_constant):
            child.network = parent_tuple[0][1].network.breed(
                parent_tuple[1][1].network,
                parent_tuple[0][0],
                parent_tuple[1][0]
            )

            return 

        
        def mutate(self, mutation_constant):
            self.network.structural_mutation()
            self.network.non_structural_mutation()

            return

        
        def generate_random(self):
            self.network = NeuralNetwork(2, 1, ['a', 'b'], ['o'],
                                          struct_mut_new_rate=0.2,
                                          struct_mut_con_rate=0.2,
                                          n_struct_mut_rate=0.2)

            self.network.build_network()
            return


        def __lt__(self, other):
            return self
    

    EC = EvolutionaryController(NEATNN)
    nn = EC.run_evolution(10, 1, 1, printing='minimal', generation_limit=5000)
    print([list(nn.network.feed_forward(a).values())[0] for a in [{'a': 0,'b': 0}, {'a': 0, 'b':1}, {'a':1, 'b':0}, {'a': 1, 'b':1}]])
    nn.network.print_connections()

    print()
    for node in nn.network.g_dict:
        data = {}
        data[node] = {}

        if nn.network.g_dict[node]['connections']:
            for connection in nn.network.g_dict[node]['connections']:
                data[node][connection] = nn.network.g_dict[node]['connections'][connection]['weight']

        print(data)
#    print(nn.network.g_dict)
