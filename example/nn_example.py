#

import sys
sys.path.insert(0, './../')

from ga.evolution.py import EvolutionaryController

if __name__ == "__main__":
    nn = NeuralNetwork(0.125)
    o = nn.add_output('o')
    h1 = nn.add_hidden({o: 0.5}, 'binary')
    h2 = nn.add_hidden({o: 0.3}, 'binary')
    h3 = nn.add_hidden({o: 0.5}, 'binary')
    a = nn.add_input({h1: 0.6, h3: 0.9, h2: 0.3}, 'a')
    b = nn.add_input({h1: 0.6, h3: 0.9, h2: 0.3}, 'b')

    print(nn.feed_forward({a: 30, b:20}))
