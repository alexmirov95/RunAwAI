#
# string_evolution.py
#

import sys
sys.path.insert(0, './../')

if __name__ == "__main__":
    import math
    import random
    
    from src.ga.individual import Individual
    from src.ga.evolution import EvolutionaryController
    
    # Test with the string example
    string = input("Enter a string: ")
    
    class String(Individual):
        def __init__(self):
            self.length = len(string)
            self.string = ''.join([" " for ii in range(len(string))])
            self.alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*(){}[].:,'"

        def individual_print(self):
            pass

        def fitness_function(self):
            fitness = self.length * 2
            for ii in range(len(self.string)):
                if self.string[ii] == string[ii]:
                    fitness -= 1

            test_dict = {}
            fit_dict = {}
            for ii in range(self.length):
                if self.string[ii] in test_dict:
                    test_dict[self.string[ii]] += 1
                else:
                    test_dict[self.string[ii]] = 1

                if string[ii] in fit_dict:
                    fit_dict[string[ii]] += 1
                else:
                    fit_dict[string[ii]] = 1

            for key in fit_dict:
                if key in test_dict:
                    fitness -= fit_dict[key]
            
            return fitness


        def breed_parents(self, parent_tuple, C, breeding_factor):
            fitA = parent_tuple[0][0]
            fitB = parent_tuple[1][0]
            A = parent_tuple[0][1]
            B = parent_tuple[1][1]
            
            if fitA > fitB:
                child = list(A.string)
            else:
                child = list(B.string)

            for ii in range( int((1 - breeding_factor)*self.length) ):
                rand_ind = random.randint(0, len(A.string) - 1)
                if random.randint(0, 1) == 1:
                    child[rand_ind] = A.string[rand_ind]
                else:
                    child[rand_ind] = B.string[rand_ind]
                                                
            return C.assign_string(''.join(child))


        def mutate(self, mutation_constant):
            string = list(self.string)
            for ii in range( math.ceil((1 - mutation_constant)*self.length) ):
                jj = random.randint(0, len(string) - 1)
                kk = random.randint(0, len(self.alphabet) - 1)
                string[jj] = self.alphabet[kk]

            self.string = ''.join(string)


        def generate_random(self):
            parent = []
            while len(parent) < self.length:
                sample_size = min(self.length - len(parent), len(self.alphabet))
                parent.extend(random.sample(self.alphabet, sample_size))
        
            self.string = ''.join(parent)


        def assign_string(self, string):
            # TODO: This method is useless?
            self.string = string


    trial = EvolutionaryController(String)
    trial.run_evolution(15, 0, 1 - 1/len(string))
