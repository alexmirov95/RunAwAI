#
# individual.py
#

class Individual(object):
    """
    The Individual class must be implemented by the user to run evolutionary
    trials. Must implement all methods to use the EvolutionaryController class.
    """
    def __init__(self):
        """
        Take some standardized input to form a new individual, whatever the
        data type.
        """
        pass


    def __lt__(self, other):
        return False
    

    def individual_print(self):
        """
        Print an individual's data in a working format.

        return: None
        """
        pass

    def fitness_function(self):
        """
        Assigns a "score" to the individual. This can be implemented with either
        a score that increases for better individuals, or decreases for better
        individuals. If you use a function that increases, make sure to include
        maximize=True in the EvolutionaryController class.

        return: A floating point value reflecting the individual's fitness score
        """
        pass

    def breed_parents(self, parent_tuple, child, reproduction_constant):
        """
        This method does not act on the individual itself, instead, it takes a
        tuple of two individuals and their fitness scores, and creates a new 
        individual with the desired combined traits from the two "parents".

        This function can be tuned with a reproduction_constant, which will
        be between 0 and 1. Standard implementations will have 1 clone the more
        fit parent, and 0 do a 50-50 mix of the the two parents.

        parent_tuple: A tuple of the form: 
          ( (fitnessA, parentA), (fitnessB, parentB) ). parentA and parentB will
          be instances of the same class as the implementation of Individual.
        reproduction_constant: A floating point value between 1 and 0 inclusive.
        return: a new Individual of the same implementaion.
        """
        pass

    def mutate(self, mutation_constant):
        """
        Mutates the Individual object
        return: None
        """
        pass

    def generate_random(self):
        """
        Changes the instance to a random permutation of the Individual class.
        This can be tweaked to give a starting point for the evolution. This
        method is called when we generate generation 0 for the evolution.

        return: None
        """
        pass
