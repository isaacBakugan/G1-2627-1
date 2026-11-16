# Nombre del integrante: Daniela Zambrano
# Cédula del integrante: 30956881


import random
import json
import statistics
import importlib.util
import os
import copy
from dataclasses import dataclass



@dataclass
class Individual:
    genes: list
    fitness: float = 0.0

    def evaluate(self, fitness_function):
        self.fitness = fitness_function(*self.genes)
        return self.fitness


class Population:

    def __init__(self, size, bounds, variable_count, fitness_function):
        self.size = size
        self.bounds = bounds
        self.variable_count = variable_count
        self.fitness_function = fitness_function
        self.members = []

    def create_random(self):
        self.members = []

        for _ in range(self.size):
            genes = [
                random.uniform(self.bounds[0], self.bounds[1])
                for _ in range(self.variable_count)
            ]

            individual = Individual(genes)
            individual.evaluate(self.fitness_function)

            self.members.append(individual)

        self.sort()

    def evaluate_all(self):
        for individual in self.members:
            individual.evaluate(self.fitness_function)

        self.sort()

    def sort(self):
        self.members.sort(
            key=lambda individual: individual.fitness,
            reverse=True
        )

    def get_statistics(self):
        fitness_values = [
            individual.fitness for individual in self.members
        ]

        return {
            "maximum": max(fitness_values),
            "median": statistics.median(fitness_values),
            "minimum": min(fitness_values)
        }

    def tournament_selection(self, tournament_size=2):
        contestants = random.sample(
            self.members,
            min(tournament_size, len(self.members))
        )

        winner = max(
            contestants,
            key=lambda individual: individual.fitness
        )

        return copy.deepcopy(winner)

    def crossover(self, parent1, parent2):
        child_genes = []

        for gene1, gene2 in zip(parent1.genes, parent2.genes):
            alpha = random.random()

            new_gene = (
                alpha * gene1
                + (1 - alpha) * gene2
            )

            new_gene = max(
                self.bounds[0],
                min(self.bounds[1], new_gene)
            )

            child_genes.append(new_gene)

        child = Individual(child_genes)
        child.evaluate(self.fitness_function)

        return child

    def mutate(self, individual, mutation_variability):
        gene_index = random.randrange(self.variable_count)

        mutation = random.gauss(0, mutation_variability)

        individual.genes[gene_index] += mutation

        individual.genes[gene_index] = max(
            self.bounds[0],
            min(self.bounds[1], individual.genes[gene_index])
        )

        individual.evaluate(self.fitness_function)

        return individual

    def evolve_generation(self, mutation_variability):
        self.sort()

        new_members = []

        # Preserve the best individual (elitism)
        elite = copy.deepcopy(self.members[0])
        new_members.append(elite)

        # Generate the remaining individuals
        while len(new_members) < self.size:

            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            child = self.crossover(parent1, parent2)

            self.mutate(child, mutation_variability)

            new_members.append(child)

        self.members = new_members
        self.sort()





class GeneticAlgorithm:

    def __init__(
        self,
        population,
        difference_threshold=0.05,
        max_generations=500,
        mutation_variability=1
    ):
        self.population = population
        self.difference_threshold = difference_threshold
        self.max_generations = max_generations
        self.mutation_variability = mutation_variability
        self.generations_elapsed = 0
    
    def train(self):
        stable_generations = 0

        for generation in range(
            self.max_generations - self.generations_elapsed
        ):
            previous_best = self.population.members[0].fitness

            self.population.evolve_generation(
                self.mutation_variability
            )

            self.generations_elapsed += 1

            stats = self.population.get_statistics()

            print(
                f"Generation {self.generations_elapsed} | "
                f"Maximum: {stats['maximum']:.6f} | "
                f"Median: {stats['median']:.6f} | "
                f"Minimum: {stats['minimum']:.6f}"
            )

            improvement = stats["maximum"] - previous_best

            if improvement <= self.difference_threshold:
                stable_generations += 1
            else:
                stable_generations = 0

            if stable_generations >= 10:
                print("Convergence reached.")
                break

        return self.population



def load_function(filename):
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        filename
    )

    spec = importlib.util.spec_from_file_location(
        "user_function",
        file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load function file: {filename}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required_attributes = [
        "BOUNDS",
        "VARIABLE_COUNT",
        "fitness"
    ]

    for attribute in required_attributes:
        if not hasattr(module, attribute):
            raise AttributeError(
                f"Function file must define {attribute}"
            )

    return module

def ask_integer(message, default, minimum=1):
    while True:
        value = input(f"{message} [{default}]: ").strip()

        if value == "":
            return default

        try:
            number = int(value)

            if number >= minimum:
                return number

            print(f"Enter a number greater than or equal to {minimum}.")

        except ValueError:
            print("Invalid input. Enter an integer.")

def ask_float(message, default, minimum=0.0, exclusive=False):
    while True:
        value = input(f"{message} [{default}]: ").strip()

        if value == "":
            return default

        try:
            number = float(value)

            valid = number > minimum if exclusive else number >= minimum

            if valid:
                return number

            print("Invalid value. Please try again.")

        except ValueError:
            print("Invalid input. Enter a decimal number.")

def create_algorithm(function_filename):

    function_module = load_function(function_filename)

    print("\n--- Population Configuration ---")

    population_size = ask_integer(
        "Population size",
        10,
        2
    )

    difference_threshold = ask_float(
        "Difference threshold",
        0.05
    )

    max_generations = ask_integer(
        "Maximum generations",
        500
    )

    mutation_variability = ask_float(
        "Mutation variability",
        1.0,
        0.0,
        exclusive=True
    )

    population = Population(
        population_size,
        function_module.BOUNDS,
        function_module.VARIABLE_COUNT,
        function_module.fitness
    )

    population.create_random()

    algorithm = GeneticAlgorithm(
        population,
        difference_threshold,
        max_generations,
        mutation_variability
    )

    return algorithm

def display_results(algorithm, function_filename):

    population = algorithm.population
    population.sort()

    print("\n" + "=" * 65)
    print("                 GENETIC ALGORITHM RESULTS")
    print("=" * 65)

    print(f"Function: {function_filename}")
    print(f"Population size: {population.size}")
    print(f"Difference threshold: {algorithm.difference_threshold}")
    print(f"Maximum generations: {algorithm.max_generations}")
    print(f"Generations elapsed: {algorithm.generations_elapsed}")
    print(f"Mutation variability: {algorithm.mutation_variability}")

    print("\nMembers (best to worst):")
    print("-" * 65)

    for index, individual in enumerate(population.members, start=1):
        variables = ", ".join(
            f"{value:.6f}" for value in individual.genes
        )

        print(
            f"{index:2}. Variables: [{variables}] | "
            f"Fitness: {individual.fitness:.8f}"
        )

    print("=" * 65)

def save_population(algorithm, function_filename, output_filename):

    population = algorithm.population
    population.sort()

    data = {
        "function": function_filename,

        "parameters": {
            "population_size": population.size,
            "difference_threshold": algorithm.difference_threshold,
            "max_generations": algorithm.max_generations,
            "mutation_variability": algorithm.mutation_variability,
            "generations_elapsed": algorithm.generations_elapsed
        },

        "members": [
            {
                "variables": individual.genes,
                "fitness": individual.fitness,
                "genes": individual.genes
            }
            for individual in population.members
        ]
    }

    with open(output_filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"\nPopulation successfully saved to: {output_filename}")

def load_population(input_filename):

    with open(input_filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    function_filename = data["function"]
    function_module = load_function(function_filename)

    parameters = data["parameters"]

    population_size = parameters["population_size"]

    population = Population(
        population_size,
        function_module.BOUNDS,
        function_module.VARIABLE_COUNT,
        function_module.fitness
    )

    population.members = []

    for member_data in data["members"]:
        genes = member_data["variables"]

        if len(genes) != population.variable_count:
            raise ValueError("Invalid number of variables in population.")

        individual = Individual(list(genes))
        individual.evaluate(population.fitness_function)

        population.members.append(individual)

    if len(population.members) != population_size:
        raise ValueError("Population size does not match the saved parameters.")

    population.sort()

    algorithm = GeneticAlgorithm(
        population,
        parameters["difference_threshold"],
        parameters["max_generations"],
        parameters["mutation_variability"]
    )

    algorithm.generations_elapsed = parameters["generations_elapsed"]

    return algorithm, function_filename


def main():

    algorithm = None
    function_filename = None

    while True:

        print("\n" + "=" * 45)
        print("          GENETIC ALGORITHM")
        print("=" * 45)
        print("1. Create a new population")
        print("2. Load a saved population")
        print("3. Train current population")
        print("4. Display population results")
        print("5. Save current population")
        print("0. Exit")
        print("=" * 45)

        option = input("Select an option: ").strip()

        if option == "1":

            print("\nAvailable functions:")
            print("1. Function A")
            print("2. Function B")

            function_option = input(
                "Select a function: "
            ).strip()

            if function_option == "1":
                function_filename = "funcion_a_2.py"

            elif function_option == "2":
                function_filename = "funcion_b_2.py"

            else:
                print("Invalid function selection.")
                continue

            algorithm = create_algorithm(function_filename)

            print("\nPopulation created successfully.")

        elif option == "2":

            filename = input(
                "Enter the JSON filename to load: "
            ).strip()

            try:
                algorithm, function_filename = load_population(filename)

                print("\nPopulation loaded successfully.")

                display_results(algorithm, function_filename)

            except (OSError, ValueError, KeyError, ImportError, AttributeError) as error:
                print(f"Error loading population: {error}")

        elif option == "3":

            if algorithm is None:
                print("Create or load a population first.")
                continue

            if algorithm.generations_elapsed >= algorithm.max_generations:
                print("Maximum number of generations already reached.")
                continue

            print("\nStarting genetic algorithm training...")

            algorithm.train()

            print("\nTraining completed.")

            display_results(algorithm, function_filename)

        elif option == "4":

            if algorithm is None:
                print("Create or load a population first.")
                continue

            display_results(algorithm, function_filename)

        elif option == "5":

            if algorithm is None:
                print("Create or load a population first.")
                continue

            filename = input(
                "Enter the JSON filename to save: "
            ).strip()

            if not filename:
                print("Filename cannot be empty.")
                continue

            if not filename.lower().endswith(".json"):
                filename += ".json"

            save_population(
                algorithm,
                function_filename,
                filename
            )

        elif option == "0":

            print("\nExiting the program...")
            break

        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()




