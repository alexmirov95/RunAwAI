#TODO might no longer be working
'''\
    Simple neural network example
'''

import sys
sys.path.append('../')

from ga import EvolutionaryController

if __name__ == "__main__":
    nn_use = NeuralNetwork(0.125)
    o = nn_use.add_output('o')
    h1 = nn_use.add_hidden({o: 0.5}, 'binary')
    h2 = nn_use.add_hidden({o: 0.3}, 'binary')
    h3 = nn_use.add_hidden({o: 0.5}, 'binary')
    a = nn_use.add_input({h1: 0.6, h3: 0.9, h2: 0.3}, 'a')
    b = nn_use.add_input({h1: 0.6, h3: 0.9, h2: 0.3}, 'b')

    print(nn_use.feed_forward({a: 30, b:20}))
