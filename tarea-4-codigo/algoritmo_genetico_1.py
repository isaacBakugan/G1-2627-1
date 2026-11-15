# Nombre del integrante: Ares Ramírez
# Cédula del integrante: 30.382.924

# Archivo 1: algoritmo_genetico_1.py

# Genetic algorithm engine. It finds the maximum of a function that lives in a
# separate Python file (contract: BOUNDS, VARIABLE_COUNT and fitness(*variables)),
# so the algorithm never has to be touched to change the function.
#
# The difference threshold is a CONVERGENCE criterion. After every generation the
# program computes the spread of the population:

#     spread = fitness(best member) - fitness(worst member)
#
# When spread < difference_threshold, all the members are practically equally
# good (the population has converged), so training stops even if the maximum
# number of generations has not been reached. A threshold of 0 disables the
# early stop. The same rule is used when a population is loaded from a file:
# if it already converged, the program reports it instead of training it again.
#
# Two interchangeable representations of a member are supported:
#     - "real":   the genes are the decoded values themselves (list of floats).
#     - "binary": every variable is a fixed-length bit string (BITS_PER_VARIABLE
#                 bits) mapped linearly onto BOUNDS; the genes are the
#                 concatenation of all the bit strings.
# Mutation variability scales the mutation:
#     - binary: every bit flips with probability variability / total_bits, so
#               on average `variability` bits flip per child.
#     - real:   one variable (on average) receives Gaussian noise whose standard
#               deviation is variability * 5% of the width of BOUNDS.
#   A variability below 1 explores less (fine tuning); above 1 explores more.
# Selection is a tournament of 3, the best member always survives (elitism)
#   and the rest of the next generation is made of mutated children.
# The population file always stores the decoded values (`variables`) and
#   their `fitness`; `genes` is only used to continue training.

import dataclasses
import importlib.util
import json
import logging
import math
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_POPULATION_SIZE = 10
DEFAULT_DIFFERENCE_THRESHOLD = 0.05
DEFAULT_MAX_GENERATIONS = 500
DEFAULT_MUTATION_VARIABILITY = 1.0

ELITE_COUNT = 1
TOURNAMENT_SIZE = 3
BITS_PER_VARIABLE = 20
REAL_STEP_FRACTION = 0.05
BLEND_ALPHA = 0.25
WORST_FITNESS = -1e300  # finite value used when a function cannot be evaluated

STOP_CONVERGED = "converged"
STOP_MAX_GENERATIONS = "max_generations"


class PopulationError(Exception):
    """Base class of the errors raised by this module."""


class FunctionLoadError(PopulationError):
    """The function file is missing or does not follow the contract."""


class PopulationFileError(PopulationError):
    """The population file cannot be read or has an invalid content."""


# --------------------------------------------------------------------------- #
# Function files
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Problem:
    """A function to maximize, loaded from its own Python file."""

    file_name: str
    bounds: tuple
    variable_count: int
    fitness_function: object

    def evaluate(self, variables):
        """Evaluate the function; undefined points get the worst fitness."""
        try:
            value = float(self.fitness_function(*variables))
        except (ArithmeticError, ValueError):
            return WORST_FITNESS
        return value if math.isfinite(value) else WORST_FITNESS


def resolve_function_path(function_file):
    """Find the function file in the current folder or next to this program."""
    candidate = Path(str(function_file).strip()).expanduser()
    options = [candidate]
    if not candidate.is_absolute():
        options = [Path.cwd() / candidate, BASE_DIR / candidate]
    options.append(BASE_DIR / candidate.name)
    for option in options:
        if option.is_file():
            return option
    return None


def load_problem(function_file):
    """Import a function file and check that it follows the contract."""
    path = resolve_function_path(function_file)
    if path is None:
        raise FunctionLoadError(f"Function file not found: {function_file}")

    spec = importlib.util.spec_from_file_location(f"fitness_module_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise FunctionLoadError(f"Not a Python file: {path.name}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # the user's file can fail in any way
        raise FunctionLoadError(f"Cannot import {path.name}: {error!r}") from error

    for attribute in ("BOUNDS", "VARIABLE_COUNT", "fitness"):
        if not hasattr(module, attribute):
            raise FunctionLoadError(f"{path.name} does not define {attribute}")
    try:
        low, high = (float(value) for value in module.BOUNDS)
        variable_count = int(module.VARIABLE_COUNT)
    except (TypeError, ValueError) as error:
        raise FunctionLoadError(f"Invalid BOUNDS or VARIABLE_COUNT in {path.name}") from error
    if not low < high or variable_count < 1 or not callable(module.fitness):
        raise FunctionLoadError(f"{path.name} does not follow the function contract")

    LOGGER.info("Loaded function %s (%d variable(s))", path.name, variable_count)
    return Problem(path.name, (low, high), variable_count, module.fitness)


# --------------------------------------------------------------------------- #
# Representations of a member
# --------------------------------------------------------------------------- #
class Representation:
    """Strategy that knows how to build, decode, cross and mutate genes."""

    name = "base"

    def __init__(self, bounds, variable_count):
        self.low, self.high = bounds
        self.variable_count = variable_count

    def clamp(self, value):
        return min(max(value, self.low), self.high)

    def random_genes(self, rng):
        raise NotImplementedError

    def decode(self, genes):
        raise NotImplementedError

    def encode(self, variables):
        raise NotImplementedError

    def crossover(self, first, second, rng):
        raise NotImplementedError

    def mutate(self, genes, variability, rng):
        raise NotImplementedError

    def genes_from_entry(self, genes, variables):
        """Genes of a member read from a file; rebuilt from `variables` if invalid."""
        raise NotImplementedError


class BinaryRepresentation(Representation):
    """Every variable is a bit string mapped linearly onto the bounds."""

    name = "binary"

    def __init__(self, bounds, variable_count, bits_per_variable=BITS_PER_VARIABLE):
        super().__init__(bounds, variable_count)
        self.bits = bits_per_variable
        self.total_bits = bits_per_variable * variable_count
        self.max_integer = 2**bits_per_variable - 1

    def random_genes(self, rng):
        return format(rng.getrandbits(self.total_bits), f"0{self.total_bits}b")

    def decode(self, genes):
        variables = []
        for index in range(self.variable_count):
            chunk = genes[index * self.bits : (index + 1) * self.bits]
            fraction = int(chunk, 2) / self.max_integer
            variables.append(self.clamp(self.low + fraction * (self.high - self.low)))
        return variables

    def encode(self, variables):
        chunks = []
        for value in variables:
            fraction = (self.clamp(value) - self.low) / (self.high - self.low)
            integer = round(fraction * self.max_integer)
            chunks.append(format(integer, f"0{self.bits}b"))
        return "".join(chunks)

    def crossover(self, first, second, rng):
        point = rng.randint(1, self.total_bits - 1)  # single-point crossover
        return first[:point] + second[point:]

    def mutate(self, genes, variability, rng):
        flip_probability = min(0.5, variability / self.total_bits)
        return "".join(
            ("1" if bit == "0" else "0") if rng.random() < flip_probability else bit
            for bit in genes
        )

    def genes_from_entry(self, genes, variables):
        valid = (
            isinstance(genes, str)
            and len(genes) == self.total_bits
            and set(genes) <= {"0", "1"}
        )
        return genes if valid else self.encode(variables)


class RealRepresentation(Representation):
    """The genes are the decoded values themselves."""

    name = "real"

    def random_genes(self, rng):
        return [rng.uniform(self.low, self.high) for _ in range(self.variable_count)]

    def decode(self, genes):
        return [self.clamp(float(value)) for value in genes]

    def encode(self, variables):
        return [self.clamp(float(value)) for value in variables]

    def crossover(self, first, second, rng):
        """Blend crossover (BLX-alpha): a random point around the two parents."""
        child = []
        for a, b in zip(first, second):
            smallest, largest = min(a, b), max(a, b)
            margin = BLEND_ALPHA * (largest - smallest)
            child.append(self.clamp(rng.uniform(smallest - margin, largest + margin)))
        return child

    def mutate(self, genes, variability, rng):
        sigma = variability * REAL_STEP_FRACTION * (self.high - self.low)
        probability = 1 / self.variable_count
        return [
            self.clamp(gene + rng.gauss(0, sigma)) if rng.random() < probability else gene
            for gene in genes
        ]

    def genes_from_entry(self, genes, variables):
        valid = (
            isinstance(genes, list)
            and len(genes) == self.variable_count
            and all(isinstance(value, (int, float)) for value in genes)
        )
        return self.encode(genes if valid else variables)


def build_representation(name, problem):
    """Create the representation called `name` ("real" or "binary")."""
    if name == BinaryRepresentation.name:
        return BinaryRepresentation(problem.bounds, problem.variable_count)
    if name == RealRepresentation.name:
        return RealRepresentation(problem.bounds, problem.variable_count)
    raise ValueError(f"Unknown representation: {name}")


def detect_representation(problem, raw_members):
    """Guess the representation of a file from the `genes` of its members."""
    real = RealRepresentation(problem.bounds, problem.variable_count)
    lengths = set()
    for entry in raw_members:
        genes = entry.get("genes")
        if not (isinstance(genes, str) and genes and set(genes) <= {"0", "1"}):
            return real
        lengths.add(len(genes))
    if len(lengths) == 1:
        total_bits = lengths.pop()
        if total_bits >= 2 and total_bits % problem.variable_count == 0:
            return BinaryRepresentation(
                problem.bounds, problem.variable_count, total_bits // problem.variable_count
            )
    return real


# --------------------------------------------------------------------------- #
# Population
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Member:
    genes: object
    variables: list
    fitness: float


@dataclass
class Parameters:
    population_size: int = DEFAULT_POPULATION_SIZE
    difference_threshold: float = DEFAULT_DIFFERENCE_THRESHOLD
    max_generations: int = DEFAULT_MAX_GENERATIONS
    mutation_variability: float = DEFAULT_MUTATION_VARIABILITY
    generations_elapsed: int = 0

    def validate(self):
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not (math.isfinite(self.difference_threshold) and self.difference_threshold >= 0):
            raise ValueError("difference_threshold must be a number >= 0")
        if self.max_generations < 1:
            raise ValueError("max_generations must be at least 1")
        if not (math.isfinite(self.mutation_variability) and self.mutation_variability > 0):
            raise ValueError("mutation_variability must be a number > 0")
        if self.generations_elapsed < 0:
            raise ValueError("generations_elapsed cannot be negative")


class Population:
    """A group of members that evolves to maximize a function."""

    def __init__(self, problem, representation, parameters, members, rng):
        self.problem = problem
        self.representation = representation
        self.parameters = parameters
        self.rng = rng
        self.members = self._sorted(members)

    # ----- construction ---------------------------------------------------- #
    @classmethod
    def create(cls, problem, parameters, representation_name="real", seed=None):
        """Create a random population (generation 0)."""
        parameters.validate()
        parameters.generations_elapsed = 0
        representation = build_representation(representation_name, problem)
        rng = random.Random(seed)
        members = [
            cls._build_member(problem, representation, representation.random_genes(rng))
            for _ in range(parameters.population_size)
        ]
        LOGGER.info(
            "Created a %s population of %d members for %s",
            representation.name,
            parameters.population_size,
            problem.file_name,
        )
        return cls(problem, representation, parameters, members, rng)

    @staticmethod
    def _build_member(problem, representation, genes):
        variables = representation.decode(genes)
        return Member(genes, variables, problem.evaluate(variables))

    @staticmethod
    def _sorted(members):
        return sorted(members, key=lambda member: member.fitness, reverse=True)

    # ----- evolution ------------------------------------------------------- #
    def _tournament(self):
        contenders = self.rng.sample(self.members, min(TOURNAMENT_SIZE, len(self.members)))
        return max(contenders, key=lambda member: member.fitness)

    def evolve_one_generation(self):
        """Build the next generation: elite members plus mutated children."""
        size = self.parameters.population_size
        next_generation = list(self.members[: min(ELITE_COUNT, size - 1)])
        while len(next_generation) < size:
            first, second = self._tournament(), self._tournament()
            genes = self.representation.crossover(first.genes, second.genes, self.rng)
            genes = self.representation.mutate(
                genes, self.parameters.mutation_variability, self.rng
            )
            next_generation.append(self._build_member(self.problem, self.representation, genes))
        self.members = self._sorted(next_generation)
        self.parameters.generations_elapsed += 1

    def fitness_summary(self):
        """Return (maximum, median, minimum) of the fitness of the population."""
        values = [member.fitness for member in self.members]
        return max(values), statistics.median(values), min(values)

    def spread(self):
        """Difference between the best and the worst fitness."""
        best, _, worst = self.fitness_summary()
        return best - worst

    def is_converged(self):
        return self.spread() < self.parameters.difference_threshold

    def train(self, on_generation=None):
        """Evolve until the maximum number of generations or until convergence.

        `on_generation(population)` is called after every generation. Returns
        STOP_CONVERGED or STOP_MAX_GENERATIONS.
        """
        if self.is_converged():
            return STOP_CONVERGED
        while self.parameters.generations_elapsed < self.parameters.max_generations:
            self.evolve_one_generation()
            if on_generation is not None:
                on_generation(self)
            if self.is_converged():
                LOGGER.info(
                    "Converged at generation %d", self.parameters.generations_elapsed
                )
                return STOP_CONVERGED
        return STOP_MAX_GENERATIONS

    def update_parameters(
        self, difference_threshold=None, max_generations=None, mutation_variability=None
    ):
        """Change the parameters that can be modified after the creation."""
        candidate = dataclasses.replace(
            self.parameters,
            difference_threshold=(
                self.parameters.difference_threshold
                if difference_threshold is None
                else difference_threshold
            ),
            max_generations=(
                self.parameters.max_generations if max_generations is None else max_generations
            ),
            mutation_variability=(
                self.parameters.mutation_variability
                if mutation_variability is None
                else mutation_variability
            ),
        )
        candidate.validate()
        self.parameters = candidate

    # ----- persistence ----------------------------------------------------- #
    def to_dict(self):
        """Dictionary that follows the population file contract of the README."""
        parameters = self.parameters
        return {
            "function": self.problem.file_name,
            "parameters": {
                "population_size": parameters.population_size,
                "difference_threshold": parameters.difference_threshold,
                "max_generations": parameters.max_generations,
                "mutation_variability": parameters.mutation_variability,
                "generations_elapsed": parameters.generations_elapsed,
            },
            "members": [
                {
                    "variables": list(member.variables),
                    "fitness": member.fitness,
                    "genes": member.genes,
                }
                for member in self.members  # already sorted from best to worst
            ],
        }

    def save(self, file_path):
        """Write the population to a JSON file (atomically)."""
        path = Path(file_path)
        if path.parent != Path():
            path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
        LOGGER.info("Saved population to %s", path)

    @classmethod
    def load(cls, file_path, function_file=None, seed=None):
        """Read a population file; `function_file` overrides the one in the file."""
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise PopulationFileError(f"Cannot read '{file_path}': {error}") from error
        try:
            population = cls._from_dict(data, function_file, seed)
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise PopulationFileError(f"Invalid population file '{file_path}': {error!r}") from error
        LOGGER.info("Loaded population from %s", file_path)
        return population

    @classmethod
    def _from_dict(cls, data, function_file, seed):
        problem = load_problem(function_file or data["function"])
        raw = data["parameters"]
        parameters = Parameters(
            population_size=int(raw["population_size"]),
            difference_threshold=float(raw["difference_threshold"]),
            max_generations=int(raw["max_generations"]),
            mutation_variability=float(raw["mutation_variability"]),
            generations_elapsed=int(raw["generations_elapsed"]),
        )
        raw_members = data["members"]
        if not raw_members:
            raise ValueError("the file has no members")

        representation = detect_representation(problem, raw_members)
        members = []
        for entry in raw_members:
            variables = [float(value) for value in entry["variables"]]
            if len(variables) != problem.variable_count:
                raise ValueError("a member has the wrong number of variables")
            genes = representation.genes_from_entry(entry.get("genes"), variables)
            member = cls._build_member(problem, representation, genes)
            stored = float(entry["fitness"])
            if not math.isclose(stored, member.fitness, rel_tol=1e-6, abs_tol=1e-9):
                LOGGER.warning(
                    "Stored fitness %s differs from the recomputed %s", stored, member.fitness
                )
            members.append(member)

        if parameters.population_size != len(members):
            LOGGER.warning(
                "population_size is %d but the file has %d members; using %d",
                parameters.population_size,
                len(members),
                len(members),
            )
            parameters.population_size = len(members)
        parameters.validate()
        return cls(problem, representation, parameters, members, random.Random(seed))


# Archivo 2: App.py

# import math
import re
# from pathlib import Path

# from algoritmo_genetico_1 import *

CANCEL_KEY = "x"
DEFAULT_FUNCTION_FILE = "funcion_a_1.py"
DEFAULT_EXTRA_GENERATIONS = 100
VARIABLE_NAMES = ("x", "y", "z")


class Cancelled(Exception):
    """The user typed the cancel key while answering a question."""


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def read_text(prompt, default=None):
    """Ask for a text; empty answer = default (or ask again if there is none)."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix} ({CANCEL_KEY} para cancelar): ").strip()
        if raw.lower() == CANCEL_KEY:
            raise Cancelled
        if raw:
            return raw
        if default is not None:
            return default
        print("  Debes escribir un valor.")


def read_int(prompt, default, minimum):
    """Ask for an integer >= minimum (decimals such as '3.5' are rejected)."""
    while True:
        raw = read_text(prompt, default)
        try:
            value = int(str(raw).strip())
        except ValueError:
            print("  Escribe un número entero (sin decimales).")
            continue
        if value < minimum:
            print(f"  El valor debe ser como mínimo {minimum}.")
            continue
        return value


def read_float(prompt, default, minimum, strictly_greater=False):
    """Ask for a finite number; a comma is accepted as decimal separator."""
    while True:
        raw = read_text(prompt, default)
        try:
            value = float(str(raw).strip().replace(",", "."))
        except ValueError:
            print("  Escribe un número válido (por ejemplo 0.5).")
            continue
        if not math.isfinite(value):
            print("  Escribe un número finito.")
            continue
        if value < minimum or (strictly_greater and value == minimum):
            comparison = "mayor que" if strictly_greater else "como mínimo"
            print(f"  El valor debe ser {comparison} {minimum}.")
            continue
        return value


def confirm(question):
    """Ask a yes/no question; anything that is not 's' or 'si' means no."""
    return input(f"{question} (s/n): ").strip().lower() in ("s", "si", "sí")


def format_number(value):
    return f"{value:.6f}"


def variable_name(index):
    return VARIABLE_NAMES[index] if index < len(VARIABLE_NAMES) else f"x{index + 1}"


def suggest_file_name(population):
    """Suggest population_N_<function>_<low|high>.json from the function file name."""
    match = re.fullmatch(r"funcion_(\w+?)_(\d+)\.py", population.problem.file_name)
    variability = population.parameters.mutation_variability
    if match is None or variability == 1:
        return "population.json"
    level = "low" if variability < 1 else "high"
    return f"population_{match.group(2)}_{match.group(1)}_{level}.json"


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class App:
    MENU = (
        "\n=== Algoritmo genético ===\n"
        "1. Crear una población\n"
        "2. Cargar una población desde un archivo\n"
        "3. Entrenar la población\n"
        "4. Mostrar la población\n"
        "5. Guardar la población en un archivo\n"
        "6. Cambiar umbral, generaciones máximas o mutación\n"
        "7. Salir"
    )

    def __init__(self):
        self.population = None
        self.unsaved_changes = False

    # ----- main loop ------------------------------------------------------- #
    def run(self):
        actions = {
            "1": self.create_population,
            "2": self.load_population,
            "3": self.train_population,
            "4": self.show_population,
            "5": self.save_population,
            "6": self.change_parameters,
        }
        while True:
            print(self.MENU)
            option = input("Elige una opción: ").strip()
            if option == "7":
                if self.confirm_discard("Hay cambios sin guardar. ¿Salir de todas formas?"):
                    print("Fin.")
                    return
                continue
            action = actions.get(option)
            if action is None:
                print("Opción inválida. Escribe un número del 1 al 7.")
                continue
            try:
                action()
            except Cancelled:
                print("Operación cancelada.")

    def confirm_discard(self, question):
        """True if there is nothing unsaved or the user accepts to lose it."""
        return not self.unsaved_changes or confirm(question)

    def require_population(self):
        if self.population is None:
            print("Primero crea o carga una población (opciones 1 o 2).")
        return self.population

    # ----- menu actions ---------------------------------------------------- #
    def create_population(self):
        if not self.confirm_discard("Hay una población sin guardar. ¿Reemplazarla?"):
            return
        function_file = read_text("Archivo de la función", DEFAULT_FUNCTION_FILE)
        try:
            problem = load_problem(function_file)
        except FunctionLoadError as error:
            print(f"No se pudo cargar la función: {error}")
            return

        parameters = Parameters(
            population_size=read_int("Tamaño de la población", DEFAULT_POPULATION_SIZE, 2),
            difference_threshold=read_float(
                "Umbral de diferencia", DEFAULT_DIFFERENCE_THRESHOLD, 0.0
            ),
            max_generations=read_int(
                "Número máximo de generaciones", DEFAULT_MAX_GENERATIONS, 1
            ),
            mutation_variability=read_float(
                "Variabilidad de la mutación",
                DEFAULT_MUTATION_VARIABILITY,
                0.0,
                strictly_greater=True,
            ),
        )
        representation = (
            "binary"
            if read_text("Representación: 1 = real, 2 = binaria", "1") == "2"
            else "real"
        )
        self.population = Population.create(problem, parameters, representation)
        self.unsaved_changes = True
        print("Población creada en la generación 0. Usa la opción 3 para entrenarla.")

    def load_population(self):
        if not self.confirm_discard("Hay una población sin guardar. ¿Reemplazarla?"):
            return
        file_path = read_text("Archivo de la población")
        function_file = None
        while True:
            try:
                self.population = Population.load(file_path, function_file)
                break
            except FunctionLoadError as error:
                print(f"No se pudo cargar la función de la población: {error}")
                function_file = read_text("Archivo de la función a usar")
            except PopulationFileError as error:
                print(f"No se pudo cargar la población: {error}")
                return
        self.unsaved_changes = False
        print("Población cargada.")
        self.show_population()

    def train_population(self):
        population = self.require_population()
        if population is None:
            return
        parameters = population.parameters

        if parameters.generations_elapsed >= parameters.max_generations:
            print(f"Ya se completaron las {parameters.max_generations} generaciones.")
            extra = read_int(
                "¿Cuántas generaciones adicionales quieres entrenar?",
                DEFAULT_EXTRA_GENERATIONS,
                1,
            )
            population.update_parameters(
                max_generations=parameters.generations_elapsed + extra
            )
        if population.is_converged():
            print(
                f"La población ya convergió: la diferencia entre el mejor y el peor "
                f"({format_number(population.spread())}) es menor que el umbral "
                f"({parameters.difference_threshold}). Puedes cambiar el umbral con la opción 6."
            )
            return

        print("Entrenando (Ctrl+C para interrumpir)...")
        try:
            reason = population.train(on_generation=self.print_generation)
        except KeyboardInterrupt:
            reason = None
            print(
                f"\nEntrenamiento interrumpido en la generación "
                f"{parameters.generations_elapsed}."
            )
        self.unsaved_changes = True

        if reason == STOP_CONVERGED:
            print(
                f"La población convergió en la generación {parameters.generations_elapsed} "
                f"(diferencia menor que el umbral {parameters.difference_threshold})."
            )
        elif reason is not None:
            print(f"Se alcanzó el máximo de {parameters.max_generations} generaciones.")
        self.show_population()

    def show_population(self):
        population = self.require_population()
        if population is None:
            return
        parameters = population.parameters
        print(f"\nFunción: {population.problem.file_name}")
        print(f"Representación: {'binaria' if population.representation.name == 'binary' else 'real'}")
        print("Parámetros:")
        print(f"  Tamaño de la población: {parameters.population_size}")
        print(f"  Umbral de diferencia: {parameters.difference_threshold}")
        print(f"  Número máximo de generaciones: {parameters.max_generations}")
        print(f"  Generaciones transcurridas: {parameters.generations_elapsed}")
        print(f"  Variabilidad de la mutación: {parameters.mutation_variability}")
        print(
            f"  Diferencia entre el mejor y el peor: {format_number(population.spread())} "
            f"({'convergida' if population.is_converged() else 'no convergida'})"
        )
        print("Miembros (de mejor a peor):")
        for position, member in enumerate(population.members, start=1):
            values = ", ".join(
                f"{variable_name(index)} = {format_number(value)}"
                for index, value in enumerate(member.variables)
            )
            print(f"  {position:>3}. {values} | función = {format_number(member.fitness)}")

    def save_population(self):
        population = self.require_population()
        if population is None:
            return
        file_path = Path(read_text("Archivo donde guardar", suggest_file_name(population)))
        if file_path.suffix.lower() != ".json":
            file_path = file_path.with_name(file_path.name + ".json")
        if file_path.exists() and not confirm(f"{file_path} ya existe. ¿Sobrescribirlo?"):
            print("No se guardó la población.")
            return
        try:
            population.save(file_path)
        except OSError as error:
            print(f"No se pudo guardar la población: {error}")
            return
        self.unsaved_changes = False
        print(f"Población guardada en {file_path}.")

    def change_parameters(self):
        population = self.require_population()
        if population is None:
            return
        parameters = population.parameters
        print("Deja una respuesta vacía para conservar el valor actual.")
        threshold = read_float(
            "Umbral de diferencia", parameters.difference_threshold, 0.0
        )
        max_generations = read_int(
            "Número máximo de generaciones", parameters.max_generations, 1
        )
        variability = read_float(
            "Variabilidad de la mutación",
            parameters.mutation_variability,
            0.0,
            strictly_greater=True,
        )
        population.update_parameters(threshold, max_generations, variability)
        self.unsaved_changes = True
        print("Parámetros actualizados.")

    # ----- progress -------------------------------------------------------- #
    @staticmethod
    def print_generation(population):
        """Show the maximum, median and minimum at the end of every generation."""
        best, median, worst = population.fitness_summary()
        print(
            f"Generación {population.parameters.generations_elapsed:>4} | "
            f"máximo: {format_number(best)} | "
            f"mediana: {format_number(median)} | "
            f"mínimo: {format_number(worst)}"
        )

# main.py

# import logging
#from app_1 import App

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        App().run()
    except (EOFError, KeyboardInterrupt):
        print("\nFin.")

main()