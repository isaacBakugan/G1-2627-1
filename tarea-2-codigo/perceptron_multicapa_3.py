# Nombre del integrante: Giselle Esclasans
# Cédula del integrante: 30715145
"""
Tarea 2 - Perceptron multicapa
Computacion Emergente (FPTSP25)

Crear o cargar una red, entrenarla con retropropagacion, probarla y guardarla.

- Todas las neuronas usan sigmoide y no tienen sesgo (el formato no lo guarda).
- Una neurona de salida por clase; gana la de mayor activacion.
  Si v = 1, la salida se compara con 0.5.
- Clases -1/1 -> neuronas 0/1. Clases 1..v -> neuronas 0..v-1.
"""

import numpy as np
import matplotlib.pyplot as plt

TASA = 0.2
MOMENTO = 0.5


#UTILIDADES:

def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def es_numero(texto):
    try:
        float(texto)
        return True
    except ValueError:
        return False


# Lee un csv de datos: retorna (entradas, clases) o (None, None) si falla:
def leer_csv(ruta):
    try:
        with open(ruta, encoding="utf-8-sig") as f:
            lineas = [l.strip() for l in f if l.strip()]
    except OSError:
        print("  -> Error: no se pudo abrir el archivo.")
        return None, None

    # Salta el encabezado
    if lineas and not es_numero(lineas[0].split(",")[0]):
        lineas = lineas[1:]

    try:
        datos = np.array([[float(x) for x in l.split(",")] for l in lineas])
    except ValueError:
        print("  -> Error: el archivo tiene valores no numericos o filas incompletas.")
        return None, None

    if datos.ndim != 2 or datos.shape[0] == 0 or datos.shape[1] < 2:
        print("  -> Error: el archivo no tiene datos validos.")
        return None, None

    return datos[:, :-1], datos[:, -1]


def pedir_entero(mensaje, minimo):
    while True:
        texto = input(mensaje).strip()
        if texto.lstrip("-").isdigit() and int(texto) >= minimo:
            return int(texto)
        print("  -> Escriba un entero mayor o igual a %d." % minimo)


def pedir_ruta(mensaje):
    return input(mensaje).strip().strip('"').strip("'")


#RED

class PerceptronMulticapa:

    # u: entradas - v: salidas - L: capas ocultas - b: neuronas por capa oculta
    def __init__(self, u, v, L, b, epocas=0):
        self.u, self.v, self.L, self.b = u, v, L, b
        self.epocas = epocas

        t = self.tamanos()
        self.pesos = []
        for k in range(len(t) - 1):
            limite = np.sqrt(6.0 / (t[k] + t[k + 1]))
            self.pesos.append(np.random.uniform(-limite, limite, (t[k + 1], t[k])))

    def tamanos(self):
        return [self.u] + [self.b] * self.L + [self.v]

    def imprimir_hiperparametros(self):
        print("\nHiperparametros:")
        print("  u (entradas)          = %d" % self.u)
        print("  v (salidas)           = %d" % self.v)
        print("  L (capas ocultas)     = %d" % self.L)
        print("  b (neuronas por capa) = %d" % self.b)
        print("  e (epocas)            = %d" % self.epocas)

    
    def es_binaria(self, clases):
        return self.v <= 2 and set(clases.tolist()) <= {-1.0, 1.0}

    def a_indices(self, clases):
        if self.es_binaria(clases):
            return (clases > 0).astype(int)
        if np.any(clases != np.round(clases)):
            return None
        indices = clases.astype(int)
        if indices.min() >= 1:
            indices = indices - 1
        if indices.min() < 0 or indices.max() >= max(self.v, 2):
            return None
        return indices


    def a_clases(self, indices, clases):
        if self.es_binaria(clases):
            return np.where(indices == 1, 1.0, -1.0)
        if clases.min() >= 1:
            return indices + 1.0
        return indices.astype(float)

    # Revisa que el archivo sirva para la red:
    def validar(self, entradas, clases):
        indices = self.a_indices(clases)
        if entradas.shape[1] == self.u and indices is not None:
            return indices

        distintas = np.unique(clases)
        print("\n  -> Error: el archivo no corresponde con la red.")
        print("     Red: %d entradas, %d salidas." % (self.u, self.v))
        print("     Archivo: %d entradas, %d clases." % (entradas.shape[1], len(distintas)))
        print("     Configure la red con %d entradas y %d salidas."
              % (entradas.shape[1], max(len(distintas), int(distintas.max()))))
        return None


    def objetivos(self, indices):
        if self.v == 1:
            return indices.reshape(-1, 1).astype(float)
        return np.eye(self.v)[indices]


    def alimentar(self, x):
        capas = [x]
        for W in self.pesos:
            capas.append(sigmoide(capas[-1] @ W.T))
        return capas

    def predecir(self, entradas):
        salida = self.alimentar(entradas)[-1]
        if self.v == 1:
            return (salida[:, 0] >= 0.5).astype(int)
        return np.argmax(salida, axis=1)

    # RETROPROPAGACIÓN:

    # Entrena vector por vector:
    def entrenar(self, entradas, indices, epocas):
        y = self.objetivos(indices)
        cambios = [np.zeros_like(W) for W in self.pesos]
        errores = []

        for epoca in range(epocas):
            for i in np.random.permutation(len(entradas)):
                capas = self.alimentar(entradas[i:i + 1])
                salida = capas[-1]
                delta = (salida - y[i:i + 1]) * salida * (1 - salida)

                for k in reversed(range(len(self.pesos))):
                    gradiente = delta.T @ capas[k]
                    delta = (delta @ self.pesos[k]) * capas[k] * (1 - capas[k])
                    cambios[k] = MOMENTO * cambios[k] - TASA * gradiente
                    self.pesos[k] += cambios[k]

            errores.append(float(np.mean((y - self.alimentar(entradas)[-1]) ** 2)))
            self.epocas += 1

            if (epoca + 1) % max(1, epocas // 10) == 0:
                aciertos = np.mean(self.predecir(entradas) == indices) * 100
                print("  Epoca %d/%d  error = %.5f  aciertos = %.1f%%"
                      % (epoca + 1, epocas, errores[-1], aciertos))

        return errores

    #ARCHIVO:

    def guardar(self, ruta):
        c = max(self.u, self.b)
        lineas = ["tipo,capa,posicion," + ",".join("w%d" % i for i in range(c))]
        for letra, valor in [("u", self.u), ("v", self.v), ("L", self.L),
                             ("b", self.b), ("e", self.epocas)]:
            lineas.append("%s,%d," % (letra, valor) + "," * c)

        for k, W in enumerate(self.pesos):
            tipo = "o" if k == len(self.pesos) - 1 else "h"
            for pos, fila in enumerate(W):
                pesos = ["%.10f" % w for w in fila] + [""] * (c - len(fila))
                lineas.append("%s,%d,%d,%s" % (tipo, k + 1, pos, ",".join(pesos)))

        with open(ruta, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas) + "\n")

    # Crea la red desde un archivo:
    @staticmethod
    def cargar(ruta):
        try:
            with open(ruta, encoding="utf-8-sig") as f:
                filas = [l.strip().split(",") for l in f if l.strip()]
        except OSError:
            print("  -> Error: no se pudo abrir el archivo.")
            return None

        hiper, neuronas = {}, {}
        try:
            for fila in filas:
                if fila[0] in ("u", "v", "L", "b", "e"):
                    hiper[fila[0]] = int(float(fila[1]))
                elif fila[0] in ("h", "o"):
                    pesos = [float(x) for x in fila[3:] if x.strip() != ""]
                    neuronas[(int(fila[1]), int(fila[2]))] = pesos
        except (ValueError, IndexError):
            print("  -> Error: el archivo de la red tiene valores invalidos.")
            return None

        faltan = [x for x in "uvLbe" if x not in hiper]
        if faltan:
            print("  -> Error: faltan hiperparametros: %s" % ", ".join(faltan))
            return None

        red = PerceptronMulticapa(hiper["u"], hiper["v"], hiper["L"], hiper["b"], hiper["e"])
        t = red.tamanos()
        for k in range(len(t) - 1):
            for pos in range(t[k + 1]):
                pesos = neuronas.get((k + 1, pos))
                if pesos is None or len(pesos) != t[k]:
                    print("  -> Error: la neurona %d de la capa %d falta o no tiene %d pesos."
                          % (pos, k + 1, t[k]))
                    return None
                red.pesos[k][pos] = pesos
        return red


#GRÁFICOS:

def crear_eje(figura, n, entradas):
    if entradas.shape[1] >= 3:
        eje = figura.add_subplot(1, 3, n, projection="3d")
        eje.set_zlabel("x3")
    else:
        eje = figura.add_subplot(1, 3, n)
    eje.set_xlabel("x1")
    eje.set_ylabel("x2")
    return eje

def puntos(eje, entradas, color, etiqueta):
    columnas = [entradas[:, j] for j in range(min(3, entradas.shape[1]))]
    if len(columnas) == 1:
        columnas.append(np.zeros(len(entradas)))
    eje.scatter(*columnas, c=color, edgecolors="black", s=40, label=etiqueta)

def graficar(red, ruta, entradas, clases, predichas, modo, errores=None):
    distintas = sorted(set(clases.tolist()) | set(predichas.tolist()))
    bien = clases == predichas

    figura = plt.figure(modo, figsize=(17, 5.5))
    figura.suptitle("%s con %s  |  u=%d v=%d L=%d b=%d e=%d  |  aciertos: %d/%d"
                    % (modo, ruta, red.u, red.v, red.L, red.b, red.epocas,
                       bien.sum(), len(bien)))

    eje = crear_eje(figura, 1, entradas)
    for i, c in enumerate(distintas):
        if np.any(clases == c):
            puntos(eje, entradas[clases == c], "C%d" % i, "clase %g" % c)
    eje.set_title("Datos del archivo")
    eje.legend()

    eje = crear_eje(figura, 2, entradas)
    if bien.any():
        puntos(eje, entradas[bien], "green", "correcta")
    if (~bien).any():
        puntos(eje, entradas[~bien], "red", "incorrecta")
    eje.set_title("Prediccion de la red")
    eje.legend()

    eje = figura.add_subplot(1, 3, 3)
    if errores is not None:
        eje.plot(range(red.epocas - len(errores) + 1, red.epocas + 1), errores)
        eje.set_title("Error por epoca")
        eje.set_xlabel("Epoca")
        eje.set_ylabel("Error cuadratico medio")
    else:
        n = len(distintas)
        matriz = np.zeros((n, n), dtype=int)
        for real, pred in zip(clases, predichas):
            matriz[distintas.index(real), distintas.index(pred)] += 1
        eje.matshow(matriz, cmap="Blues")
        for i in range(n):
            for j in range(n):
                eje.text(j, i, matriz[i, j], ha="center", va="center")
        etiquetas = ["%g" % c for c in distintas]
        eje.set_xticks(range(n))
        eje.set_xticklabels(etiquetas)
        eje.set_yticks(range(n))
        eje.set_yticklabels(etiquetas)
        eje.set_xlabel("Predicha")
        eje.set_ylabel("Real")
        eje.set_title("Matriz de confusion")

    figura.tight_layout()
    print("\nCierre la ventana del grafico para continuar...")
    plt.show()


#MENÚ:

def crear_red():
    print("\nNueva red:")
    u = pedir_entero("  Neuronas de entrada: ", 1)
    v = pedir_entero("  Neuronas de salida: ", 1)
    L = pedir_entero("  Capas ocultas: ", 0)
    b = pedir_entero("  Neuronas por capa oculta: ", 1) if L > 0 else 0
    return PerceptronMulticapa(u, v, L, b)


def cargar_red():
    while True:
        red = PerceptronMulticapa.cargar(pedir_ruta("\nArchivo de la red: "))
        if red is not None:
            print("Red cargada.")
            return red


# Pide un archivo de datos compatible:
def pedir_datos(red, mensaje):
    ruta = pedir_ruta(mensaje)
    entradas, clases = leer_csv(ruta)
    if entradas is None:
        return None
    indices = red.validar(entradas, clases)
    if indices is None:
        return None
    return ruta, entradas, clases, indices


def opcion_entrenar(red):
    datos = pedir_datos(red, "\nArchivo de entrenamiento: ")
    if datos is None:
        return
    ruta, entradas, clases, indices = datos
    epocas = pedir_entero("Epocas: ", 1)

    print("\nEntrenando...")
    errores = red.entrenar(entradas, indices, epocas)
    predichas = red.a_clases(red.predecir(entradas), clases)

    red.imprimir_hiperparametros()
    graficar(red, ruta, entradas, clases, predichas, "Entrenamiento", errores)


def opcion_probar(red):
    datos = pedir_datos(red, "\nArchivo de prueba: ")
    if datos is None:
        return
    ruta, entradas, clases, _ = datos

    predichas = red.a_clases(red.predecir(entradas), clases)
    print("\nAciertos: %d/%d" % (np.sum(predichas == clases), len(clases)))

    red.imprimir_hiperparametros()
    graficar(red, ruta, entradas, clases, predichas, "Prueba")


def opcion_guardar(red):
    ruta = pedir_ruta("\nArchivo donde guardar (ej. red.csv): ")
    if ruta == "":
        return
    if not ruta.lower().endswith(".csv"):
        ruta += ".csv"
    try:
        red.guardar(ruta)
        print("Red guardada en %s" % ruta)
    except OSError:
        print("  -> Error: no se pudo escribir el archivo.")


def main():
    print("TAREA 2 - PERCEPTRON MULTICAPA\n")
    print("  1) Crear red nueva")
    print("  2) Cargar red de un archivo")
    opcion = ""
    while opcion not in ("1", "2"):
        opcion = input("Opcion: ").strip()
    red = crear_red() if opcion == "1" else cargar_red()
    red.imprimir_hiperparametros()

    acciones = {"1": opcion_entrenar, "2": opcion_probar, "3": opcion_guardar}
    while True:
        print("\n  1) Entrenar\n  2) Probar\n  3) Guardar\n  4) Salir")
        opcion = input("Opcion: ").strip()
        if opcion == "4":
            break
        if opcion in acciones:
            acciones[opcion](red)
        else:
            print("  -> Opcion invalida.")

    print("\nFin.")


if __name__ == "__main__":
    main()
