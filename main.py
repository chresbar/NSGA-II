import random as rn
import numpy as np
import matplotlib.pyplot as plt
import math


# _____________________________________________________________________________
def random_population(nv, n, lb, ub):
    # nv = number of variables
    # n = number of random solutions
    # lb = lower bound
    # ub = upper bound
    pop = np.zeros((n, nv))
    for i in range(n):
        pop[i, :] = np.random.uniform(lb, ub)

    return pop


# _____________________________________________________________________________
def crossover(pop, crossover_rate):
    offspring = np.zeros((crossover_rate, pop.shape[1]))
    for i in range(int(crossover_rate / 2)):
        r1 = np.random.randint(0, pop.shape[0])
        r2 = np.random.randint(0, pop.shape[0])
        while r1 == r2:
            r1 = np.random.randint(0, pop.shape[0])
            r2 = np.random.randint(0, pop.shape[0])
        cutting_point = np.random.randint(1, pop.shape[1])
        offspring[2 * i, 0:cutting_point] = pop[r1, 0:cutting_point]
        offspring[2 * i, cutting_point:] = pop[r2, cutting_point:]
        offspring[2 * i + 1, 0:cutting_point] = pop[r2, 0:cutting_point]
        offspring[2 * i + 1, cutting_point:] = pop[r1, cutting_point:]

    return offspring


# _____________________________________________________________________________
def mutation(pop, mutation_rate):
    offspring = np.zeros((mutation_rate, pop.shape[1]))
    for i in range(int(mutation_rate / 2)):
        r1 = np.random.randint(0, pop.shape[0])
        r2 = np.random.randint(0, pop.shape[0])
        while r1 == r2:
            r1 = np.random.randint(0, pop.shape[0])
            r2 = np.random.randint(0, pop.shape[0])
        cutting_point = np.random.randint(0, pop.shape[1])
        offspring[2 * i] = pop[r1]
        offspring[2 * i, cutting_point] = pop[r2, cutting_point]
        offspring[2 * i + 1] = pop[r2]
        offspring[2 * i + 1, cutting_point] = pop[r1, cutting_point]

    return offspring


# _____________________________________________________________________________
def local_search(pop, n, step_size):
    # number of offspring chromosomes generated from the local search
    offspring = np.zeros((n, pop.shape[1]))
    for i in range(n):
        r1 = np.random.randint(0, pop.shape[0])
        chromosome = pop[r1, :]
        r2 = np.random.randint(0, pop.shape[1])
        chromosome[r2] += np.random.uniform(-step_size, step_size)
        if chromosome[r2] < lb[r2]:
            chromosome[r2] = lb[r2]
        if chromosome[r2] > ub[r2]:
            chromosome[r2] = ub[r2]

        offspring[i, :] = chromosome
    return offspring


# _____________________________________________________________________________
def evaluation(pop):
    fitness_values = np.zeros((pop.shape[0], 2))  # Assuming ZDT1 has two objectives
    for i, chromosome in enumerate(pop):
        fitness_values[i, 0] = chromosome[0]  # Objective 1 is the first variable in ZDT1
        g = 1 + 9 * sum(chromosome[1:]) / (pop.shape[1] - 1)
        fitness_values[i, 1] = g * (1 - math.sqrt(chromosome[0] / g))

    return fitness_values


# _____________________________________________________________________________
def crowding_calculation(fitness_values):
    pop_size = len(fitness_values[:, 0])
    fitness_value_number = len(fitness_values[0, :])
    matrix_for_crowding = np.zeros((pop_size, fitness_value_number))
    normalize_fitness_values = (fitness_values - fitness_values.min(0)) / fitness_values.ptp(0)
    for i in range(fitness_value_number):
        crowding_results = np.zeros(pop_size)
        crowding_results[0] = 1
        crowding_results[pop_size - 1] = 1
        sorting_normalize_fitness_values = np.sort(normalize_fitness_values[:, i])
        sorting_normalized_values_index = np.argsort(normalize_fitness_values[:, i])

        ###UNSURE

        crowding_results[1:pop_size - 1] = sorting_normalize_fitness_values[
                                           2:pop_size] - sorting_normalize_fitness_values[0:pop_size - 2]
        re_sorting = np.argsort(sorting_normalized_values_index)
        matrix_for_crowding[:, i] = crowding_results[re_sorting]
    crowding_distance = np.sum(matrix_for_crowding, axis=1)  # crowding distance of each solution

    return crowding_distance


# _____________________________________________________________________________
def remove_using_crowding(fitness_values, number_solutions_needed):
    pop_index = np.arange(fitness_values.shape[0])
    crowding_distance = crowding_calculation(fitness_values)
    selected_pop_index = np.zeros(number_solutions_needed)
    selected_fitness_values = np.zeros((number_solutions_needed, len(fitness_values[0, :])))

    for i in range(number_solutions_needed):
        pop_size = pop_index.shape[0]
        solution_1 = rn.randint(0, pop_size - 1)
        solution_2 = rn.randint(0, pop_size - 1)
        if crowding_distance[solution_1] >= crowding_distance[solution_2]:
            #
            selected_pop_index[i] = pop_index[solution_1]
            selected_fitness_values[i, :] = fitness_values[solution_1, :]
            pop_index = np.delete(pop_index, solution_1, axis=0)
            fitness_values = np.delete(fitness_values, solution_1, axis=0)
            crowding_distance = np.delete(crowding_distance, solution_1, axis=0)
        else:
            #
            selected_pop_index[i] = pop_index[solution_2]
            selected_fitness_values[i, :] = fitness_values[solution_2, :]
            pop_index = np.delete(pop_index, (solution_2), axis=0)
            fitness_values = np.delete(fitness_values, (solution_2), axis=0)
            crowding_distance = np.delete(crowding_distance, (solution_2), axis=0)
    selected_pop_index = np.asarray(selected_pop_index, dtype=int)  # Convert the data to integer

    return (selected_pop_index)


# _____________________________________________________________________________
def pareto_front_finding(fitness_values, pop_index):
    pop_size = fitness_values.shape[0]
    pareto_front = np.ones(pop_size, dtype=bool)  # initially assume all solutions are in pareto front by using "1"
    for i in range(pop_size):
        for j in range(pop_size):
            if all(fitness_values[j] <= fitness_values[i]) and any(fitness_values[j] < fitness_values[i]):
                pareto_front[i] = 0
                break
    return pop_index[pareto_front]


# _____________________________________________________________________________
def selection(pop, fitness_values, pop_size):
    pop_index_0 = np.arange(pop.shape[0])
    pop_index = np.arange(pop.shape[0])
    pareto_front_index = []

    while len(pareto_front_index) < pop_size:
        new_pareto_front = pareto_front_finding(fitness_values[pop_index_0, :], pop_index_0)
        total_pareto_size = len(pareto_front_index) + len(new_pareto_front)

        if total_pareto_size > pop_size:
            number_solutions_needed = pop_size - len(pareto_front_index)
            selected_solutions = (remove_using_crowding(fitness_values[new_pareto_front], number_solutions_needed))
            new_pareto_front = new_pareto_front[selected_solutions]

        pareto_front_index = np.hstack((pareto_front_index, new_pareto_front))
        remaining_index = set(pop_index) - set(pareto_front_index)
        pop_index_0 = np.array(list(remaining_index))

    selected_pop = pop[pareto_front_index.astype(int)]

    return selected_pop


# _____________________________________________________________________________
# Parameters
nv = 2
lb = [0, 0]
ub = [1, 1]
pop_size = 100
rate_crossover = 30
rate_mutation = 20
rate_local_search = 30
step_size = 0.1
pop = random_population(nv, pop_size, lb, ub)
# _____________________________________________________________________________
# Main loop of NSGA II

for i in range(150):
    offspring_from_crossover = crossover(pop, rate_crossover)
    offspring_from_mutation = mutation(pop, rate_mutation)
    offspring_from_local_search = local_search(pop, rate_local_search, step_size)
    pop = np.append(pop, offspring_from_crossover, axis=0)
    pop = np.append(pop, offspring_from_mutation, axis=0)
    pop = np.append(pop, offspring_from_local_search, axis=0)
    fitness_values = evaluation(pop)
    pop = selection(pop, fitness_values, pop_size)
    print('iteration', i)
# _____________________________________________________________________________
# Pareto front visualization
fitness_values = evaluation(pop)
index = np.arange(pop.shape[0]).astype(int)
pareto_front_index = pareto_front_finding(fitness_values, index)
pop = pop[pareto_front_index, :]
print("_________________")
print("Optimal solutions:")
print("     x1           x2      ")
print(pop)  # show optimal solutions
fitness_values = fitness_values[pareto_front_index]
print("______________")
print("Fitness values:")
print("objective 1  objective 2")
print("      |          |")
print(fitness_values)
plt.scatter(fitness_values[:, 0], fitness_values[:, 1])
plt.xlabel('Objective function 1')
plt.ylabel('Objective function 2')
plt.show()
