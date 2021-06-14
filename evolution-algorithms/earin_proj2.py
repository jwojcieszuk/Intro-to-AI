# EARIN Project #2: Wojcieszuk Jakub, Gałecki Szymon
# Program implementing evolution algorithm

import numpy as np
import argparse
import functools
import random


# Function for obtaining and validating input parameters for the evolution algorithm
def parse_arguments(parser):

    parser.add_argument("-dim", type=int, required=True,
                        help="Problem dimensionality")

    parser.add_argument("-d", type=int, required=True,
                        help="Range of integers")

    parser.add_argument("-A", type=process_input,
                        required=True, help="NxN matrix")

    parser.add_argument("-b", type=process_input,
                        required=True, help="Vector of n numbers.")

    parser.add_argument("-c", type=float, required=True, help="A scalar.")

    parser.add_argument("-p", type=int, required=True, help="Population size")

    parser.add_argument("-cr", type=float, required=True,
                        help="Crossover probability")

    parser.add_argument("-m", type=float, required=True,
                        help="Mutation probability")

    parser.add_argument("-i", type=int, required=True,
                        help="Number of algorithm iterations")

    args = parser.parse_args()

    if args.dim < 1:
        parser.error("Dimension cannot be lower than 1.")

    if args.d < 1:
        parser.error("D parameter must be greater than 0.")

    if args.A.shape[0] != args.d or args.A.shape[1] != args.d:
        parser.error("Matrix dimensions does not match value of parameter d.")

    if not args.A.shape[0] == args.A.shape[1]:
        parser.error("Parameter A has to be n x n matrix.")

    if args.b.ndim != 1 or args.b.shape[0] != args.d:
        parser.error("Parameter b has to be vector of n numbers.")

    if args.p < 1:
        parser.error("Population size cannot be lower than 1.")

    if args.cr > 1:
        parser.error("Crossover probability cannot be bigger than 1.")

    if args.m > 1:
        parser.error("Mutation probability cannot be bigger than 1.")

    if args.i < 1:
        parser.error("Number of iterations cannot be lower than 1.")

    if args.p % 2 != 0:
        parser.error("Population size must be an even number")

    return args


# Function for producing the matrix A
def process_input(input):
    # matrix should be provided with columns separated by commas and rows separated by semoicolons
    # e.g '1, 2, 3; 4, 5, 6; 7, 8, 9'
    arr = np.array([[float(x) for x in y.split(",")]
                    for y in input.split(";")])

    # flatten to a vector if it is a column matrix
    if arr.shape[0] == 1:
        return arr.flatten()

    return arr


# Function to generate initial population
def generate_population(d: int, dimension: int, population_size: int):

    # border values
    low = -2**d
    high = 2**d

    # 'dimension' numbers in a single row
    # 'population_size' number of rows
    # uniformly distributed integers from range <low, high>
    population = np.random.randint(
        low, high+1, size=(population_size, dimension))

    return population


# Function that converts decimal representation of the population to binary
def convert_to_binary(population):

    # append strings with binary representation to an empty list
    binary_population = []
    for i in population:
        for j in i:
            element = f'{j:08b}'
            binary_population.append(element)

    # convert to numpy array, reshape it to 'population' array dimensions
    binary_population = np.array(binary_population).reshape(population.shape)

    # return binary values of a population
    return binary_population


# Function that converts binary representation of the population to decimal
def convert_to_integer(population):

    # append strings with decimal representation to an empty list
    decimal_population = []
    for i in population:
        for j in i:
            element = int(j, 2)
            decimal_population.append(element)

    # convert to numpy array, reshape it to 'population' array dimensions
    decimal_population = np.array(decimal_population).reshape(population.shape)

    # return decimal values of a population
    return decimal_population


# Function that performs single-point crossover on the consecutive pairs,
# selection and crossover function implement FIFO replacement strategy
def single_point_crossover(crossover_probability, population):
    offspring = population
    # two parents generates two children
    # if crossover probability doesn't occur, pass parents to offspring
    for i in range(0, offspring.shape[0], 2):

        x1 = population[i]
        x2 = population[i+1]

        for j in range(offspring.shape[1]):
            if random.random() < crossover_probability:
                # pick random index to perform crossover
                k = random.randint(0, 7)
                # bit switch
                offspring[i][j] = x1[j][:k] + x2[j][k:]
                offspring[i+1][j] = x2[j][:k] + x1[j][k:]

    return offspring


# Function to be used inside the Fitness function, since we need column vectors of the solutions
def reshape_3D(population):

    # what this produces: array of 2D column arrays
    reshape = population.reshape(population.shape[0], population.shape[1], 1)
    return reshape


# Function that outputs list of function values for the input population, in corresponding order
def fitness_function(population, A, b, c):

    # to evaluate mathematical expressions we need array of column vectors to iterate through
    reshaped = reshape_3D(population)

    # list to store function values for every solution x
    values = []

    # vector b is input as already transposed
    # go through the whole array, obtain function values
    for i in range(reshaped.shape[0]):
        x = reshaped[i]
        f = np.dot(np.dot(x.T, A), x) + np.dot(b, x) + c
        f = f[0][0]
        values.append(f)

    # return list o function values
    return values


# Function for selecting solutions that will be used for generating new population
def roulette_wheel_selection(population, values):

    if max(values) - min(values) == 0:
        return population
    # normalisation of values
    values = (values - min(values)) / (max(values) - min(values))

    # probabilites corresponding to respective values
    probabilities = values / sum(values)

    # indices of selected solutions
    indices = np.random.choice(
        population.shape[0], size=population.shape[0], p=probabilities)

    # empty list for storing all selected solutions
    selected = []

    for i in indices:
        selected.append(population[i])

    # convert to numpy.array for compatibility, return selected solutions
    selected = np.array(selected)
    return selected


# Function for performing mutation on the binary representation of population
def mutation(mut_prob, offspring, d):
    mutated = offspring

    for i in range(offspring.shape[0]):
        x = offspring[i]

        for j in range(offspring.shape[1]):

            l = list(range(d+1, 8))
            l.append(0)
            temp = list(x[j])
            
            for k in l:
                if random.random() < mut_prob:
                 # index for bit switch is selected from list [0, d+1 up to 8]
                    
                    # if index 0 was picked and it is '0', switch to a negative sign
                    if k == 0:
                        if temp[k] == '0':
                            temp[k] = '-'
                        if temp[k] == '-':
                            temp[k] = '0'

                    # switch 0 to 1
                    if temp[k] == '0':
                        temp[k] = '1'

                    # switch 1 to 0
                    elif temp[k] == '1':
                        temp[k] = '0'

            temp = ''.join(temp)
            mutated[i][j] = temp

    return mutated


# Main program loop
def evolution(args):

    # create initial population
    population = generate_population(args.d, args.dim, args.p)
    print("Initial population:\n", population)

    for i in range(args.i):
        # evaluating values of population
        values = fitness_function(population, args.A, args.b, args.c)

        # selecting solutions from population
        selection = roulette_wheel_selection(population, values)

        # convert selection to binary to enable single-point crossover and mutation
        bin_selection = convert_to_binary(selection)

        # last iteration population and their values
        if i == args.i - 1:
            print("Final population:\n", selection)
            print("Final population values:\n", fitness_function(
                selection, args.A, args.b, args.c))
            return

        # single-point crossover
        crossover = single_point_crossover(args.cr, bin_selection)

        # mutation
        mutated = mutation(args.m, crossover, args.d)

        # conversion of the new population to integer
        population = convert_to_integer(mutated)


if __name__ == '__main__':

    args = parse_arguments(argparse.ArgumentParser())

    print("dimension =", args.dim, "\n")
    print("d =", args.d, "\n")
    print("A =\n", args.A, "\n")
    print("b =", args.b, "\n")
    print("c =", args.c, "\n")
    print("Population size =", args.p, "\n")
    print("Crossover probability =", args.cr, "\n")
    print("Mutation probability =", args.m, "\n")
    print("Number of iterations =", args.i, "\n")

    evolution(args)
