#
# evolution.py
#
# Implementation of an evolutionary algorithm framework that runs on instances
# of classes implementing the individual abstract class.

import time
import heapq
import datetime
import itertools


class EvolutionaryController():
    """
    The EvolutionaryController class is an abstract controller for evolving
    any "learning" structure. Instances of EvolutionaryController take 
    implementations of the Individual class and run evolution simulations on
    generations of individuals.
    """
    def __init__(self, Individual, halt_on_errors=True):
        """
        Takes as input a class implementing the Individual abstract class.
        
        Individual: Class implementing the Individual abstract class.
        halt_on_errors: [ True | False ] Specify wheter or not the  evolutionary
          controller will halt on non-fatal errors.
        return: None
        """
        self.Individual = Individual

    
    def error(self, error_string, fatal="False", exit_code=1):
        """
        Handle errors that occur in the EvolutionaryController.

        fatal: [ False | True ] If error is fatal, exit the program.
        return: None
        """
        # TODO: Print to err and throw exception instead of exiting
        if fatal:
            print("Encountered fatal error: \"%s\"" % error_string)
        else:
            print("Encountered error: \"%s\"" % error_string)
            
        if halt_on_errors:
            print("Exiting due to error")
            exit(exit_code)
            

    def generate_seed_parents(self, breeding_constant):
        """
        Generate a list of random parents for generation 0.
        
        return: A list of parents (Individual() instances). Returns nC2 random
          individuals.
        """

        # check that the breeding constant is an integer
        if not isinstance(breeding_constant, int):
            self.error("Breeding constant must be integer", fatal=True)
        
        parents = []
        
        for ii in range( breeding_constant * (breeding_constant - 1)// 2 ):
            parent = self.Individual()
            parent.generate_random()
            parents.append(parent)
        
        return parents

    
    def run_evolution(self, breeding_constant, reproduction_constant,
                      mutation_constant, optimal_fitness=0.0, maximize=False,
                      generation_limit=1000, collect_data=False,
                      printing="minimal"):
        """
        breeding_const: How many individuals to be chosen from each generation 
          to be bred together.
        reproduction_const: Float between 1 and 0. 1 means more fit parent is 
          cloned, 0 means parent's "genes" are combined 50-50
        mutation_const: A number between 1 and 0. 1 means total mutation (random
          individual created), 0 is no mutation at all.
        optimal_fitness: This is a floating point value that is a cutoff for the
          fitness function, when we reach this value we stop the evolution.
        maximize: [ False | True ] This tells the evolution whether or not to 
          treat fitness as a value to be maximized or minimized. By default we
          try to minimize fitness.
        generation_limit: [ float | "inf" ] When we hit the generation limit we
          stop iterating. If generation_limit is set to "inf", we do not stop
          unless we find an optimal individual.
        collect_data: [ False | True ] Toggle data collection
        printing: [ off | minimal | full ] How much printing should we do.
        """
        if maximize:
            optimal_fitness *= -1
        
        if printing is not "off":
            start_time = datetime.datetime.now()

        if collect_data:
            # TODO: initialize data collection
            data_map = {}

        generation_id = 0
        generation = self.generate_seed_parents( breeding_constant )

        heap = []

        best_fitness = float("inf")
        best_indv = generation[0]
        
        for individual in generation:
            fitness = individual.fitness_function()
            
            if maximize:
                fitness *= -1
            if fitness < best_fitness:
                best_fitness = fitness
                best_indv = individual
                
            heapq.heappush(heap, (fitness, individual))

        while True:
            most_fit = []
            avg_fit = 0
            top_fit = heap[0][0]
            
            for _ in range( breeding_constant ):
                pop = heapq.heappop(heap)
                most_fit.append(pop)
                avg_fit += pop[0]
                if pop[0] < top_fit:
                    top_fit = pop[0]

            avg_fit /= breeding_constant

            #[print(fit[0], " ", fit[1].string) for fit in most_fit]
            #print()

            if printing == 'full':
                print("%d : %f : %f" % (generation_id, avg_fit, top_fit))

            heap = []
            
            for pair in itertools.combinations(most_fit, 2):
                new_individual = self.Individual()
                (pair[0])[1].breed_parents(pair, new_individual,
                                           reproduction_constant)
                new_individual.mutate(mutation_constant)
            
                fitness = new_individual.fitness_function()
                
                if maximize:
                    fitness *= -1
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_indv = new_individual

                #print(fitness)
                heapq.heappush(heap, (fitness, new_individual))
            
            generation_id += 1
            
            if generation_limit is not "inf":
                if generation_id >= generation_limit:
                    stop_time = datetime.datetime.now() - start_time
                    if printing is not "off":
                        print("Stopped in ", stop_time, " seconds on generation %d" % generation_id)
                        print("Average fitness %f" % avg_fit)
                    return best_indv
                
            if best_fitness <= optimal_fitness:
                stop_time = datetime.datetime.now() - start_time
                
                if printing is not "off":
                    print("Found optimal solution in %d generations in " %
                          generation_id, stop_time, " seconds")
                return best_indv


