# Nombre del integrante: Daniela Zambrano
# Cédula del integrante: 30956881

import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# CLASE PRINCIPAL: PERCEPTRÓN MULTICAPA
# ==========================================
class PerceptronMulticapa:
    def __init__(self, u: int, v: int, L: int, b: int, e: int = 0):
        """
        u: Neuronas de entrada
        v: Neuronas de salida
        L: Número de capas ocultas
        b: Neuronas por capa oculta
        e: Número de épocas acumuladas
        """
        self.u = int(u)
        self.v = int(v)
        self.L = int(L)
        self.b = int(b)
        self.e = int(e)
        self.c = max(self.u, self.b)

        # Inicialización de matrices de pesos
        # Capa 1: b neuronas, cada una recibe u entradas
        # Capas 2..L: b neuronas, cada una recibe b entradas
        # Capa salida (L+1): v neuronas, recibe b entradas (o u si L == 0)
        self.weights = []
        if self.L > 0:
            # Capa oculta 1
            limit_1 = np.sqrt(6.0 / (self.u + self.b))
            self.weights.append(np.random.uniform(-limit_1, limit_1, (self.b, self.u)))
            # Capas ocultas intermedias 2..L
            for _ in range(1, self.L):
                limit_mid = np.sqrt(6.0 / (self.b + self.b))
                self.weights.append(np.random.uniform(-limit_mid, limit_mid, (self.b, self.b)))
            # Capa de salida
            limit_out = np.sqrt(6.0 / (self.b + self.v))
            self.weights.append(np.random.uniform(-limit_out, limit_out, (self.v, self.b)))
        else:
            # Caso directo sin capas ocultas
            limit_direct = np.sqrt(6.0 / (self.u + self.v))
            self.weights.append(np.random.uniform(-limit_direct, limit_direct, (self.v, self.u)))

    @staticmethod
    def _act(z):
        # Clip evita desbordamientos (overflow) con np.exp en valores muy grandes
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _act_deriv(a):
        return a * (1.0 - a)

    def forward(self, x: np.ndarray):
        """
        Calcula las activaciones hacia adelante.
        x: vector columna de dimensión (u, 1)
        Retorna: lista de activaciones [a0, a1, ..., a_out]
        """
        activations = [x]
        curr = x
        for w in self.weights:
            z = np.dot(w, curr)
            curr = self._act(z)
            activations.append(curr)
        return activations

    def predict(self, X: np.ndarray, classes_list: list):
        """
        Genera predicciones de clase para una matriz de datos X de tamaño (N, u).
        """
        preds = []
        for i in range(len(X)):
            x_vec = X[i].reshape(-1, 1)
            acts = self.forward(x_vec)
            out = acts[-1].flatten()

            if self.v == 1:
                # El umbral para sigmoide es 0.5
                pred_val = classes_list[1] if out[0] >= 0.5 else classes_list[0]
            else:
                best_idx = int(np.argmax(out))
                pred_val = classes_list[best_idx]
            preds.append(pred_val)
        return np.array(preds)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, lr: float = 0.03):
        """
        Entrenamiento por retropropagación con gradiente estocástico/mini-lotes.
        """
        classes_list = sorted(list(np.unique(y)))
        n_samples = len(X)
        history_errors = []

        # Convertir etiquetas y a vectores target adaptados a sigmoide [0, 1]
        targets = []
        for label in y:
            if self.v == 1:
                t = np.array([[1.0]] if label == classes_list[1] else [[0.0]])
            else:
                t = np.zeros((self.v, 1)) # Modificado de np.full a np.zeros
                idx = classes_list.index(label)
                t[idx, 0] = 1.0
            targets.append(t)
        targets = np.array(targets)

        for ep in range(epochs):
            indices = np.random.permutation(n_samples)
            total_epoch_err = 0.0

            for idx in indices:
                x_i = X[idx].reshape(-1, 1)
                t_i = targets[idx]

                # 1. Forward
                acts = self.forward(x_i)
                out = acts[-1]

                # Error cuadrático
                err = 0.5 * np.sum((t_i - out) ** 2)
                total_epoch_err += err

                # 2. Backward
                deltas = [None] * len(self.weights)
                # Gradiente en la salida
                deltas[-1] = (out - t_i) * self._act_deriv(out)

                # Propagación hacia capas ocultas
                for layer_idx in range(len(self.weights) - 2, -1, -1):
                    deltas[layer_idx] = np.dot(self.weights[layer_idx + 1].T, deltas[layer_idx + 1]) * self._act_deriv(acts[layer_idx + 1])

                # 3. Actualización de pesos
                for layer_idx in range(len(self.weights)):
                    grad = np.dot(deltas[layer_idx], acts[layer_idx].T)
                    self.weights[layer_idx] -= lr * grad

            history_errors.append(total_epoch_err / n_samples)

        self.e += epochs
        return history_errors

    def save_to_csv(self, filename: str):
        """
        Guarda la red según el formato CSV exacto de la especificación.
        """
        c = self.c
        header = ["tipo", "capa", "posicion"] + [f"w{i}" for i in range(c)]

        rows = [header]
        # Filas 1 a 5: Hiperparámetros
        params = [
            ("u", self.u),
            ("v", self.v),
            ("L", self.L),
            ("b", self.b),
            ("e", self.e)
        ]
        for letter, val in params:
            row = [letter, str(val), ""] + [""] * c
            rows.append(row)

        # Filas 6 en adelante: Neuronas
        total_layers = len(self.weights)
        for layer_zero_idx, w_mat in enumerate(self.weights):
            layer_num = layer_zero_idx + 1
            is_output = (layer_num == total_layers)
            neuron_type = "o" if is_output else "h"

            for neuron_idx in range(w_mat.shape[0]):
                neuron_weights = list(w_mat[neuron_idx])
                # Rellenar con cadenas vacías hasta completar c columnas de peso
                pad_count = c - len(neuron_weights)
                padded_weights = [f"{w:.8f}" for w in neuron_weights] + [""] * pad_count
                row = [neuron_type, str(layer_num), str(neuron_idx)] + padded_weights
                rows.append(row)

        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"Red guardada exitosamente en: {filename}")

    @classmethod
    def load_from_csv(cls, filename: str):
        """
        Carga la red desde un archivo CSV con el formato especificado.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"El archivo {filename} no existe.")

        with open(filename, mode="r", newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        if len(reader) < 6:
            raise ValueError("El archivo no tiene el formato válido de perceptrón multicapa.")

        # Lectura de hiperparámetros en filas 1 a 5
        param_dict = {}
        for r_idx in range(1, 6):
            row = reader[r_idx]
            param_dict[row[0].strip()] = int(row[1].strip())

        u = param_dict["u"]
        v = param_dict["v"]
        L = param_dict["L"]
        b = param_dict["b"]
        e = param_dict["e"]

        net = cls(u=u, v=v, L=L, b=b, e=e)

        # Carga de los pesos
        weight_rows = reader[6:]
        layers_data = {}
        for row in weight_rows:
            if not row or not row[0].strip():
                continue
            layer_num = int(row[1].strip())
            neuron_idx = int(row[2].strip())

            # Cantidad de entradas esperadas para esta capa
            expected_inputs = u if layer_num == 1 else b
            weights = [float(row[3 + i]) for i in range(expected_inputs)]

            if layer_num not in layers_data:
                layers_data[layer_num] = {}
            layers_data[layer_num][neuron_idx] = weights

        for layer_zero_idx in range(len(net.weights)):
            layer_num = layer_zero_idx + 1
            neurons_in_layer = sorted(layers_data[layer_num].keys())
            w_list = [layers_data[layer_num][idx] for idx in neurons_in_layer]
            net.weights[layer_zero_idx] = np.array(w_list, dtype=float)

        print(f"Red cargada exitosamente desde: {filename}")
        return net


# ==========================================
# UTILIDADES DE DATOS Y VISUALIZACIÓN
# ==========================================
def load_dataset(filename: str):
    """
    Carga un archivo CSV con columnas x1, x2, ..., y.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Archivo no encontrado: {filename}")

    data = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                data.append([float(val) for val in row])
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def validate_network_against_data(net: PerceptronMulticapa, X: np.ndarray, y: np.ndarray):
    """
    Verifica que la red coincida con las dimensiones del dataset.
    Si no coincide, arroja una excepción detallada.
    """
    n_inputs = X.shape[1]
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # Coincidencia de entrada
    if net.u != n_inputs:
        raise ValueError(
            f"\n[ERROR DE CONFIGURACIÓN]\n"
            f"El archivo requiere {n_inputs} neuronas de entrada, pero la red tiene {net.u}.\n"
            f"Por favor configure la red con u={n_inputs}."
        )

    # Coincidencia de salida
    valid_outputs = [n_classes] if n_classes > 2 else [1, 2]
    if net.v not in valid_outputs:
        raise ValueError(
            f"\n[ERROR DE CONFIGURACIÓN]\n"
            f"El archivo posee {n_classes} clases ({unique_classes.tolist()}).\n"
            f"La red tiene v={net.v} neuronas de salida. Debe configurarse con v={n_classes} (o v=1 para clasificación binaria)."
        )


def plot_results(X: np.ndarray, y: np.ndarray, preds: np.ndarray, error_data, is_training: bool, classes_list: list):
    """
    Genera los tres gráficos obligatorios:
    1. Datos reales con scatter (2D o 3D).
    2. Predicción correcta vs incorrecta.
    3. Curva de error de épocas (entrenamiento) o Matriz de confusión (prueba).
    """
    u_dims = X.shape[1]
    is_3d = u_dims >= 3

    fig = plt.figure(figsize=(17, 5))

    # Gráfico 1: Datos reales
    if is_3d:
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="tab10", edgecolors="k", s=35)
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_zlabel("x3")
    else:
        ax1 = fig.add_subplot(1, 3, 1)
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", edgecolors="k", s=40)
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
    ax1.set_title("1. Datos originales por clase")
    plt.colorbar(scatter1, ax=ax1, label="Clase real")

    # Gráfico 2: Aciertos vs Errores
    correct = (y == preds)
    colors = np.where(correct, "#2ca02c", "#d62728")  # Verde: correcto, Rojo: incorrecto

    if is_3d:
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax2.scatter(X[correct, 0], X[correct, 1], X[correct, 2], c="#2ca02c", label="Correcto", s=35, edgecolors="k")
        ax2.scatter(X[~correct, 0], X[~correct, 1], X[~correct, 2], c="#d62728", label="Incorrecto", s=50, marker="x")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        ax2.set_zlabel("x3")
    else:
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.scatter(X[correct, 0], X[correct, 1], c="#2ca02c", label="Correcto", s=40, edgecolors="k")
        ax2.scatter(X[~correct, 0], X[~correct, 1], c="#d62728", label="Incorrecto", s=60, marker="x")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
    acc = np.mean(correct) * 100
    ax2.set_title(f"2. Predicciones (Precisión: {acc:.1f}%)")
    ax2.legend()

    # Gráfico 3: Error de la red
    if is_training:
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(range(1, len(error_data) + 1), error_data, color="#1f77b4", linewidth=2)
        ax3.set_title("3. Error de la red por época")
        ax3.set_xlabel("Época")
        ax3.set_ylabel("MSE (Costo)")
        ax3.grid(True, linestyle="--", alpha=0.6)
    else:
        # Matriz de confusión
        n_c = len(classes_list)
        conf_mat = np.zeros((n_c, n_c), dtype=int)
        for true_val, pred_val in zip(y, preds):
            i = classes_list.index(true_val)
            j = classes_list.index(pred_val)
            conf_mat[i, j] += 1

        ax3 = fig.add_subplot(1, 3, 3)
        cax = ax3.matshow(conf_mat, cmap="Blues")
        fig.colorbar(cax, ax=ax3)
        ax3.set_xticks(range(n_c))
        ax3.set_yticks(range(n_c))
        ax3.set_xticklabels(classes_list)
        ax3.set_yticklabels(classes_list)
        ax3.set_title("3. Matriz de confusión", pad=15)
        ax3.set_xlabel("Predicción")
        ax3.set_ylabel("Clase Real")
        for i in range(n_c):
            for j in range(n_c):
                ax3.text(j, i, str(conf_mat[i, j]), ha="center", va="center", color="red" if conf_mat[i, j] > 0 else "gray")

    plt.tight_layout()
    plt.show()


def print_hyperparameters(net: PerceptronMulticapa):
    print("\n" + "=" * 45)
    print("       HIPERPARÁMETROS DE LA RED")
    print("=" * 45)
    print(f"  • Entradas (u):                  {net.u}")
    print(f"  • Salidas (v):                   {net.v}")
    print(f"  • Capas ocultas (L):             {net.L}")
    print(f"  • Neuronas por capa oculta (b):  {net.b}")
    print(f"  • Épocas acumuladas (e):         {net.e}")
    print("=" * 45 + "\n")


# ==========================================
# MENÚ INTERACTIVO Y FLUJO DE EJECUCIÓN
# ==========================================
def main():
    print("==============================================")
    print("       PERCEPTRÓN MULTICAPA — TAREA 2")
    print("==============================================")

    net = None
    while net is None:
        print("\nSeleccione una opción de inicio:")
        print("  1. Crear un nuevo perceptrón multicapa")
        print("  2. Cargar red desde un archivo CSV")
        print("  3. Salir")
        opc = input("Opción (1-3): ").strip()

        if opc == "1":
            try:
                u = int(input("Número de neuronas de entrada: "))
                v = int(input("Número de neuronas de salida: "))
                L = int(input("Número de capas ocultas: "))
                b = int(input("Número de neuronas por capa oculta: "))
                net = PerceptronMulticapa(u=u, v=v, L=L, b=b, e=0)
                print("Red creada satisfactoriamente.")
            except ValueError:
                print("Error: Todos los parámetros deben ser valores enteros.")
        elif opc == "2":
            filepath = input("Ruta del archivo CSV a cargar: ").strip()
            try:
                net = PerceptronMulticapa.load_from_csv(filepath)
            except Exception as ex:
                print(f"Error al cargar el archivo: {ex}")
        elif opc == "3":
            print("Saliendo del programa.")
            sys.exit(0)
        else:
            print("Opción inválida. Intente de nuevo.")

    # Menú de operaciones
    while True:
        print_hyperparameters(net)
        print("Opciones disponibles:")
        print("  1. Entrenar la red con retropropagación")
        print("  2. Probar la red con alimentación hacia delante")
        print("  3. Guardar la red")
        print("  4. Salir")
        opc = input("Seleccione una opción (1-4): ").strip()

        if opc == "1":
            train_file = input("Archivo de datos de entrenamiento (ej: assets/2-d_2-class_train.csv): ").strip()
            try:
                epochs = int(input("Número de épocas a entrenar: "))
                if epochs <= 0:
                    print("El número de épocas debe ser mayor a 0.")
                    continue
                X_train, y_train = load_dataset(train_file)
                validate_network_against_data(net, X_train, y_train)

                print(f"\nEntrenando durante {epochs} épocas...")
                loss_history = net.train(X_train, y_train, epochs=epochs, lr=0.03)
                classes_list = sorted(list(np.unique(y_train)))
                preds = net.predict(X_train, classes_list)

                plot_results(X_train, y_train, preds, loss_history, is_training=True, classes_list=classes_list)

            except Exception as ex:
                print(f"\nError durante el entrenamiento: {ex}")

        elif opc == "2":
            test_file = input("Archivo de datos de prueba (ej: assets/2-d_2-class_test.csv): ").strip()
            try:
                X_test, y_test = load_dataset(test_file)
                validate_network_against_data(net, X_test, y_test)

                classes_list = sorted(list(np.unique(y_test)))
                preds = net.predict(X_test, classes_list)

                plot_results(X_test, y_test, preds, None, is_training=False, classes_list=classes_list)

            except Exception as ex:
                print(f"\nError durante la prueba: {ex}")

        elif opc == "3":
            save_file = input("Nombre o ruta del archivo para guardar (ej: red_2d.csv): ").strip()
            try:
                net.save_to_csv(save_file)
            except Exception as ex:
                print(f"Error al guardar la red: {ex}")

        elif opc == "4":
            print("Cerrando el programa. ¡Hasta luego!")
            break
        else:
            print("Opción no válida. Ingrese un número entre 1 y 4.")


if __name__ == "__main__":
    main()