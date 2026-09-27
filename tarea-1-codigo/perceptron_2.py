# Nombre del integrante: Ares Ramírez
# Cédula del integrante: 30.382.924

# Comentarios de la actividad: 
#   -   A fin de implementar el preceptron se investigó la funcion sigmoide y ReLU.
#   -   Funciones implementadas: Sigmoide y ReLU

# Comentarios de la entrega: Inicialmente realicé la actividad en 3 archivos y luego los unifiqué 
# para subirla acá siguiendo las instrucciones del repositorio, por eso a continuación
# se muestran el codigo dividido por 3 comentarios de "Archivo 1"

# Archivo 1. Main
import matplotlib.pyplot as plt
# from Perceptron import *
# from functions import *

def main():
    entries, outputs = request_file()

    while True:
        print("\n----- Ejecución de un perceptrón -----")
        weights, slant, option = request_parameters(len(entries[0]))
        name = ACTIVATION_FUNCTIONS[option][0]
        predicted = execute_perceptron(entries, weights, slant, option)
        correct = 0
        for expected, result in zip(outputs, predicted):
            if expected == result:
                correct += 1
        print("Aciertos: %d de %d (%.1f%%)"% (correct, len(outputs), 100.0 * correct / len(outputs)))

        title = "%s | b=%s, w=%s | aciertos %d/%d" % (name, slant, weights, correct, len(outputs))
        graph(entries, outputs, predicted, title)
        plt.show()

        response = input("¿Probar otros pesos? (s = sí, a = otro archivo, n = salir): ").strip().lower()
        if response == "a":
            entries, outputs = request_file()
        elif response == "n":
            print("Saliendo del sistema ...")
            break

# Archivo 2. Perceptron (funciones principales)

#import matplotlib.pyplot as plt
#from functions import execute_vector

def execute_perceptron(entries, weights, slant, activation_function):
    """Ejecuta el perceptrón para cada muestra y devuelve las clases predichas."""
    predicted = []
    for row in entries:
        result = execute_vector(row, weights, slant, activation_function)
        predicted.append(result)
    return predicted


def show_subplt(ax, xs, ys, etiquetas_grupos, titulo):
    """Dibuja grupos de puntos definidos por índices, color y leyenda."""
    for indices, color, leyenda in etiquetas_grupos:
        if indices:
            ax.scatter(
                [xs[i] for i in indices],
                [ys[i] for i in indices],
                c=color,
                label=leyenda,
                edgecolors="black",
                s=70,
            )
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def graph(entries, outputs, predicted, title="Resultados del perceptrón"):
    """Grafica esperado, predicho y coincidencia."""
    xs = [row[0] for row in entries]
    ys = [row[1] if len(row) > 1 else 0 for row in entries]
    indices = range(len(entries))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    show_subplt(
        axes[0], xs, ys,
        [([i for i in indices if outputs[i] == 0], "tab:blue", "esperado = 0"),
         ([i for i in indices if outputs[i] == 1], "tab:orange", "esperado = 1")],
        "Valor esperado",
    )
    show_subplt(
        axes[1], xs, ys,
        [([i for i in indices if predicted[i] == 0], "tab:blue", "predicho = 0"),
         ([i for i in indices if predicted[i] == 1], "tab:orange", "predicho = 1")],
        "Valor predicho por el perceptrón",
    )
    show_subplt(
        axes[2], xs, ys,
        [([i for i in indices if outputs[i] == predicted[i]], "tab:green", "coincide"),
         ([i for i in indices if outputs[i] != predicted[i]], "tab:red", "no coincide")],
        "¿Coinciden?",
    )
    for axis in axes:
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# Archivo 3. Funciones (para modularizar y aislar cierta logica)
def validate_number(value):
    try:
        return float(value.strip().replace(",", "."))
    except (ValueError, AttributeError):
        return False

def ask_number(msg="    Ingrese un número: "):
    while True:
        number = validate_number(input(msg))
        if number is not False:
            return number
        print("     Entrada inválida. Por favor, ingrese un número válido.")

# Funcion de carga del CSV
# n colunmas de entrada
# n-1 columnas de valores de entrada
# n-esima columna es la salida esperada    
def upload_csv(file_name):
    entries = []
    outputs = []
    column_count = None
    with open(file_name, "r") as file:
        for line_number, raw_line in enumerate(file, start=1):
            
            if line_number == 1:
                continue
            
            line = raw_line.strip()
            if not line:
                continue

            separator = ";" if ";" in line else ","
            values = []
            for part in line.split(separator):
                values.append(validate_number(part))
            
            for value in values:
                if value is False:
                    raise ValueError("La línea %d contiene valores no numéricos." % line_number)

            if column_count is None:
                column_count = len(values)
                if column_count < 2:
                    raise ValueError("El CSV necesita al menos dos columnas.")
            elif len(values) != column_count:
                raise ValueError(
                    "La línea %d tiene %d columnas; se esperaban %d."
                    % (line_number, len(values), column_count)
                )

            entries.append(values[:-1])
            outputs.append(values[-1])

    if not entries:
        raise ValueError("El archivo no contiene datos.")
    
    outputs = [int(output) for output in outputs]

    return entries, outputs

def request_file():
    
    while True:
        file_name = input("Ruta del archivo CSV: ").strip()
        try:
            entries, outputs = upload_csv(file_name)
            print("Archivo cargado: %d muestras, %d entradas por muestra." % (len(entries), len(entries[0])))
            
            return entries, outputs
        
        except (OSError, ValueError) as error:
            print("  No se pudo cargar el archivo:", error)


def request_parameters(entry_count):
    
    slant = ask_number("Sesgo: ")
    
    weights = []
    for index in range(entry_count):
        weight = ask_number("Peso w%d: " % (index + 1))
        weights.append(weight)

    for key, (name, _, _) in ACTIVATION_FUNCTIONS.items():
        print("  %s) %s" % (key, name))

    option = input("Elige (1/2): ").strip()
    while option not in ACTIVATION_FUNCTIONS:
        option = input("  Opción inválida, elige 1 o 2: ").strip()
    return weights, slant, option


def z_function(entries, weights, slant):
    if len(entries) != len(weights):
        raise ValueError("Debe haber un peso por cada entrada.")
    z_value = slant
    for entry, weight in zip(entries, weights):
        z_value += weight * entry
    return z_value

def sigmoid(z):
    return 1 / (1 + 2.718281 ** (-z))

def relu(z):
    return max(0.0, z)


ACTIVATION_FUNCTIONS = {
    "1": ("Sigmoide", sigmoid, 0.5),
    "2": ("ReLU", relu, 0.0),
}

def execute_vector(entries, weights, slant, activation_function):
    if activation_function not in ACTIVATION_FUNCTIONS:
        raise ValueError("Función de activación no reconocida.")
    _, function, limit = ACTIVATION_FUNCTIONS[activation_function]
    activation = function(z_function(entries, weights, slant))
    return 1 if activation > limit else 0


# Ejecución

main()
