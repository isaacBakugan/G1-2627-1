# Nombre del integrante: Giselle Esclasans
# Cédula del integrante: 30715145
"""
Tarea 4 - Genetic algorithm

Finds the max of a function that is in another .py file (funcion_a_3.py,
funcion_b_3.py or any file that has BOUNDS, VARIABLE_COUNT and fitness()).

Each member is just a list of real numbers (one per variable), so the genes
are already the x (or x, y) values. Selection by tournament, blend crossover
and gaussian mutation where the std deviation is the mutation variability.
The best one always passes to the next generation (elitism).

About the difference threshold: I use it to know when to stop. If the
difference between the best and the worst of the population is less than the
threshold, they are all almost the same so it converged. I also check that the
best didn't improve more than the threshold in the last 30 generations,
because with only the first check sometimes it stopped too early in a point
that wasn't the max (all members became copies of the same one by luck).
"""

import importlib.util
import json
import os
import random
import statistics

FOLDER = os.path.dirname(os.path.abspath(__file__))

MUTATION_PROB = 0.3   # chance of mutating each gene
PATIENCE = 30


def find_file(path):
    # try the path as it is, if not look in the same folder as this script
    if os.path.isfile(path):
        return path
    if os.path.isfile(os.path.join(FOLDER, path)):
        return os.path.join(FOLDER, path)
    return None


def load_function(path):
    real = find_file(path)
    if real == None:
        raise FileNotFoundError("file not found: " + path)
    spec = importlib.util.spec_from_file_location("function_module", real)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "fitness") or not hasattr(module, "BOUNDS") or not hasattr(module, "VARIABLE_COUNT"):
        raise ValueError("the file needs BOUNDS, VARIABLE_COUNT and fitness")
    return module


def evaluate(pop, genes):
    try:
        value = float(pop["function"].fitness(*genes))
    except (ZeroDivisionError, ValueError, OverflowError):
        return float("-inf")
    if value != value:  # nan
        return float("-inf")
    return value


def clip(pop, value):
    low, high = pop["function"].BOUNDS
    if value < low:
        return low
    if value > high:
        return high
    return value


def set_members(pop, all_genes):
    pop["members"] = []
    for genes in all_genes:
        pop["members"].append({"genes": genes, "fitness": evaluate(pop, genes)})
    # best first
    pop["members"].sort(key=lambda m: m["fitness"], reverse=True)


def new_population(function_file, function, size, threshold, max_gen, variability):
    pop = {
        "function_file": function_file,
        "function": function,
        "size": size,
        "threshold": threshold,
        "max_generations": max_gen,
        "variability": variability,
        "generations_elapsed": 0,
        "members": [],
    }
    low, high = function.BOUNDS
    all_genes = []
    for i in range(size):
        all_genes.append([random.uniform(low, high) for j in range(function.VARIABLE_COUNT)])
    set_members(pop, all_genes)
    return pop


def tournament(pop):
    # pick 3 random and keep the best one
    group = random.sample(pop["members"], min(3, len(pop["members"])))
    best = group[0]
    for m in group:
        if m["fitness"] > best["fitness"]:
            best = m
    return best["genes"]


def crossover(pop, a, b):
    if random.random() > 0.9:
        return a[:], b[:]
    child1 = []
    child2 = []
    for i in range(len(a)):
        lo = min(a[i], b[i])
        hi = max(a[i], b[i])
        d = (hi - lo) * 0.5   # BLX-0.5, lets the child go a bit outside the parents
        child1.append(clip(pop, random.uniform(lo - d, hi + d)))
        child2.append(clip(pop, random.uniform(lo - d, hi + d)))
    return child1, child2


def mutate(pop, genes):
    for i in range(len(genes)):
        if random.random() < MUTATION_PROB:
            genes[i] = clip(pop, genes[i] + random.gauss(0, pop["variability"]))
    return genes


def next_generation(pop):
    new = [pop["members"][0]["genes"][:]]  # elitism
    while len(new) < pop["size"]:
        c1, c2 = crossover(pop, tournament(pop), tournament(pop))
        new.append(mutate(pop, c1))
        if len(new) < pop["size"]:
            new.append(mutate(pop, c2))
    set_members(pop, new)
    pop["generations_elapsed"] += 1


def train(pop, verbose=True):
    best_history = [pop["members"][0]["fitness"]]
    for g in range(pop["max_generations"]):
        next_generation(pop)
        values = [m["fitness"] for m in pop["members"]]
        best_history.append(values[0])
        if verbose:
            print("Generacion %d | max: %.6f | mediana: %.6f | min: %.6f"
                  % (pop["generations_elapsed"], max(values), statistics.median(values), min(values)))

        # stop criteria (see the comment at the top)
        if max(values) - min(values) < pop["threshold"] and len(best_history) > PATIENCE:
            if best_history[-1] - best_history[-1 - PATIENCE] < pop["threshold"]:
                if verbose:
                    print("La poblacion convergio despues de", g + 1, "generaciones.")
                break


def show(pop):
    print()
    print("Funcion:", pop["function_file"])
    print("Tamano de la poblacion:", pop["size"])
    print("Umbral de diferencia:", pop["threshold"])
    print("Generaciones maximas:", pop["max_generations"])
    print("Generaciones transcurridas:", pop["generations_elapsed"])
    print("Variabilidad de mutacion:", pop["variability"])
    print("Miembros de mejor a peor:")
    names = ["x", "y", "z"]
    for i, m in enumerate(pop["members"]):
        text = ""
        for j in range(len(m["genes"])):
            name = names[j] if j < len(names) else "v" + str(j + 1)
            text += "%s = %.6f  " % (name, m["genes"][j])
        print("%2d) %s-> f = %.6f" % (i + 1, text, m["fitness"]))


def save(pop, path):
    data = {
        "function": pop["function_file"],
        "parameters": {
            "population_size": pop["size"],
            "difference_threshold": pop["threshold"],
            "max_generations": pop["max_generations"],
            "mutation_variability": pop["variability"],
            "generations_elapsed": pop["generations_elapsed"],
        },
        "members": [],
    }
    for m in pop["members"]:
        data["members"].append({
            "variables": [round(v, 10) for v in m["genes"]],
            "fitness": round(m["fitness"], 10),
            "genes": [round(v, 10) for v in m["genes"]],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load(path):
    real = find_file(path)
    if real == None:
        raise FileNotFoundError("file not found: " + path)
    with open(real, encoding="utf-8") as f:
        data = json.load(f)

    # the function file should be next to the json
    function_file = data["function"]
    next_to_json = os.path.join(os.path.dirname(os.path.abspath(real)), function_file)
    if os.path.isfile(next_to_json):
        function = load_function(next_to_json)
    else:
        function = load_function(function_file)

    p = data["parameters"]
    pop = {
        "function_file": function_file,
        "function": function,
        "size": int(p["population_size"]),
        "threshold": float(p["difference_threshold"]),
        "max_generations": int(p["max_generations"]),
        "variability": float(p["mutation_variability"]),
        "generations_elapsed": int(p.get("generations_elapsed", 0)),
        "members": [],
    }
    all_genes = []
    for m in data["members"]:
        genes = m.get("genes")
        if type(genes) != list or len(genes) != function.VARIABLE_COUNT:
            genes = m["variables"]
        all_genes.append([float(v) for v in genes])
    set_members(pop, all_genes)
    return pop


def read_number(text, default, kind):
    while True:
        answer = input(text + " [" + str(default) + "]: ").strip()
        if answer == "":
            return default
        try:
            n = kind(answer)
            if n > 0:
                return n
            print("Tiene que ser mayor que 0")
        except ValueError:
            print("Eso no es un numero valido")


def read_path(text):
    return input(text).strip().strip('"').strip("'")


def main():
    pop = None
    while True:
        print()
        print("=== Algoritmo genetico ===")
        print("1. Crear poblacion")
        print("2. Cargar poblacion")
        print("3. Entrenar")
        print("4. Ver poblacion")
        print("5. Guardar poblacion")
        print("6. Salir")
        op = input("Opcion: ").strip()

        if op == "1":
            file = read_path("Archivo de la funcion (ej: funcion_a_3.py): ")
            try:
                function = load_function(file)
            except Exception as e:
                print("No se pudo cargar la funcion:", e)
                continue
            size = read_number("Tamano de la poblacion", 10, int)
            if size < 2:
                print("Minimo 2 miembros, se usa 2")
                size = 2
            threshold = read_number("Umbral de diferencia", 0.05, float)
            max_gen = read_number("Numero maximo de generaciones", 500, int)
            variability = read_number("Variabilidad de la mutacion", 1.0, float)
            pop = new_population(os.path.basename(file), function, size, threshold, max_gen, variability)
            print("Poblacion creada")
        elif op == "2":
            path = read_path("Archivo .json: ")
            try:
                pop = load(path)
                show(pop)
            except Exception as e:
                print("No se pudo cargar:", e)
        elif op in ["3", "4", "5"] and pop == None:
            print("Primero tienes que crear o cargar una poblacion")
        elif op == "3":
            train(pop)
            show(pop)
        elif op == "4":
            show(pop)
        elif op == "5":
            path = read_path("Nombre del archivo (ej: population_3_a_low.json): ")
            if not path.endswith(".json"):
                path = path + ".json"
            try:
                save(pop, path)
                print("Guardada en", path)
            except OSError as e:
                print("No se pudo guardar:", e)
        elif op == "6":
            break
        else:
            print("Opcion invalida")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nchao")
