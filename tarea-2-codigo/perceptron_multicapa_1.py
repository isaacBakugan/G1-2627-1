# Nombre del integrante: Ares Ramírez
# Cédula del integrante: 30.382.924

# Comentarios de la actividad: 


# Comentarios de la entrega: Al igual que con la entrega 1, realicé la actividad en 4 archivos
# y luego los unifiqué para subirla acá siguiendo las instrucciones del repositorio, por eso 
# a continuación se muestran el codigo dividido por 4 comentarios de "Archivo n"

# ---------- Archivo 1. Main ----------
#from App import App
 
def main():
    App().menu_principal()
    
# ---------- Archivo 2. App ----------
import numpy as np
import matplotlib.pyplot as plt
# from Perceptron import Perceptron
# from funciones import *

class App:
    def __init__(self):
        self.red = None # el Perceptron con el que se está trabajando

    # -------------------- Entrada del usuario --------------------
    def leer_entero(self, mensaje, minimo):
        """Repite la pregunta hasta que el usuario escribe un entero >= minimo."""
        while True:
            try:
                valor = int(a_numero(input(mensaje)))
                if valor >= minimo:
                    return valor
                print("  Debe ser un entero mayor o igual a %d." % minimo)
            except ValueError:
                print("  Valor inválido, escribe un número entero.")

    
    # -------------------- Datos --------------------
    def cargar_csv(self,ruta):
        """
        Devuelve (X, y).
        X: matriz (N, n-1) con las entradas. y: vector (N,) de enteros con la clase
        (puede ser -1/1, 0..k o 1..k: la columna se toma tal cual viene en el archivo).
        Tolera: encabezado, líneas vacías, separador ',' o ';', decimales con coma.
        """
        filas = []
        n_columnas = None
        with open(ruta, encoding="utf-8-sig") as archivo:
            for numero_linea, linea in enumerate(archivo, start=1):
                linea = linea.strip()
                if linea == "":
                    continue
                separador = ";" if ";" in linea else ","
                try:
                    valores = [a_numero(p) for p in linea.split(separador)]
                except ValueError:
                    if n_columnas is None:          # primera línea de texto = encabezado
                        continue
                    raise ValueError("Línea %d con valores no numéricos." % numero_linea)
                if n_columnas is None:
                    n_columnas = len(valores)
                    if n_columnas < 2:
                        raise ValueError("El CSV necesita al menos 2 columnas.")
                if len(valores) != n_columnas:
                    raise ValueError("La línea %d tiene %d columnas y se esperaban %d."
                                     % (numero_linea, len(valores), n_columnas))
                filas.append(valores)
        if not filas:
            raise ValueError("El archivo no tiene datos.")
        datos = np.array(filas, dtype=float)
        X = datos[:, :-1]
        clases = datos[:, -1]
        if np.any(clases != np.round(clases)):
            raise ValueError("La última columna debe tener la clase como número entero.")
        return X, clases.astype(int)

    def pedir_datos(self):
        """Pide un CSV hasta que cargue y corresponda a la red. Devuelve (X, y) o (None, None)."""
        while True:
            ruta = input("Ruta del archivo CSV de datos: ").strip().strip('"').strip("'")
            try:
                X, y = self.cargar_csv(ruta)
            except (OSError, ValueError) as error:
                print("  No se pudo leer el archivo:", error)
                continue                       # volver a pedir la ruta
            try:
                self.red.validar_datos(X, y)
            except ValueError as error:
                print("  El archivo no corresponde a la red:", error)
                return None, None              # volver al menú para reconfigurar
            print("Archivo cargado: %d muestras, %d entradas." % (X.shape[0], X.shape[1]))
            return X, y

    # -------------------- Gráficos --------------------
    def matriz_confusion(self, reales, predichas, n):
        """reales y predichas son índices de clase (0..n-1)."""
        cm = np.zeros((n, n), dtype=int)
        for real, pred in zip(reales, predichas):
            cm[real, pred] += 1
        return cm

    def ejes(self, fig, posicion, tres_d):
        if tres_d:
            return fig.add_subplot(1, 3, posicion, projection="3d")
        return fig.add_subplot(1, 3, posicion)

    def scatterplot(self, ax, X, grupos, titulo, tres_d):
        """grupos = (mascaras, colores, nombres): un scatter por grupo de puntos."""
        xs = X[:, 0]
        ys = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
        zs = X[:, 2] if X.shape[1] > 2 else None
        for mascara, color, nombre in zip(*grupos):
            if not np.any(mascara):
                continue
            if tres_d:
                ax.scatter(xs[mascara], ys[mascara], zs[mascara], c=[color], label=nombre,
                           edgecolors="black", s=40)
            else:
                ax.scatter(xs[mascara], ys[mascara], c=[color], label=nombre,
                           edgecolors="black", s=40)
        ax.set_title(titulo)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        if tres_d:
            ax.set_zlabel("x3")
        ax.legend(loc="best", fontsize=7)

    def graficar(self, X, reales, predichas, modo, historial=None):
        """
        Una figura con tres gráficos (reales y predichas son índices de clase 0..v-1):
          1) los datos (primeras 2 entradas, o 3 en proyección 3D si hay 3 o más)
          2) predicción correcta (verde) o incorrecta (roja)
          3) modo 'entrenar': error por época (línea); modo 'Evaluar': matriz de confusión
        """
        red = self.red
        tres_d = X.shape[1] >= 3
        n_clases = max(red.v, 2)
        nombres = [str(et) for et in (red.etiquetas or range(n_clases))]
        paleta = plt.get_cmap("tab10")
        fig = plt.figure(figsize=(16, 5.2))

        ax1 = self.ejes(fig, 1, tres_d)
        self.scatterplot(ax1, X, ([reales == k for k in range(n_clases)],
                               [paleta(k % 10) for k in range(n_clases)],
                               ["clase %s" % nombre for nombre in nombres]),
                      "Datos del archivo", tres_d)

        ax2 = self.ejes(fig, 2, tres_d)
        acierta = predichas == reales
        self.scatterplot(ax2, X, ([acierta, ~acierta], ["green", "red"],
                               ["correcta (%d)" % acierta.sum(), "incorrecta (%d)" % (~acierta).sum()]),
                      "Predicción de la red", tres_d)

        ax3 = fig.add_subplot(1, 3, 3)
        if modo == "entrenar":
            primera = red.e - len(historial) + 1
            ax3.plot(range(primera, red.e + 1), historial, color="tab:blue")
            ax3.set_title("Error por época (entrenamiento)")
            ax3.set_xlabel("época")
            ax3.set_ylabel("error cuadrático medio")
            ax3.grid(True, alpha=0.3)
        else:
            cm = self.matriz_confusion(reales, predichas, n_clases)
            plt.sca(ax3)
            plt.matshow(cm, fignum=0, cmap="Blues")
            ax3.xaxis.tick_bottom()
            for i in range(n_clases):
                for j in range(n_clases):
                    ax3.text(j, i, str(cm[i, j]), ha="center", va="center",
                             color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax3.set_xticks(range(n_clases))
            ax3.set_yticks(range(n_clases))
            ax3.set_xticklabels(nombres)
            ax3.set_yticklabels(nombres)
            ax3.set_xlabel("clase predicha")
            ax3.set_ylabel("clase real")
            ax3.set_title("Matriz de confusión (prueba)")
        fig.suptitle("%s | aciertos %d/%d (%.1f%%)"
                     % ("Entrenamiento" if modo == "entrenar" else "Prueba",
                        acierta.sum(), len(reales), 100.0 * acierta.mean()))
        fig.tight_layout()
        return fig

    # -------------------- Crear o cargar la red --------------------
    def crear(self):
        print(" ----- Creación de un perceptrón multicapa -----")
        u = self.leer_entero("Número de neuronas de entrada: ", 1)
        v = self.leer_entero("Número de neuronas de salida: ", 1)
        L = self.leer_entero("Número de capas ocultas: ", 0)
        b = self.leer_entero("Número de neuronas por capa oculta: ", 1) if L > 0 else 0
        self.red = Perceptron(u, v, L, b)

    def _cargar(self):
        while True:
            ruta = input("Ruta del archivo de la red: ").strip().strip('"').strip("'")
            try:
                self.red = Perceptron.cargar(ruta)
                print("Red cargada.")
                return
            except (OSError, ValueError) as error:
                print("  No se pudo cargar la red:", error)

    # -------------------- Menu principal --------------------
    def entrenar(self):
        X, y = self.pedir_datos()
        if X is None:
            return
        epocas = self.leer_entero("Número de épocas: ", 1)
        historial = self.red.entrenar(X, y, epocas)
        reales = self.red.indices_de_clases(y)
        self.graficar(X, reales, self.red.predecir_indices(X), "entrenar", historial)
        self.imprimir_hiperparametros()
        plt.show()

    def evaluar(self):
        X, y = self.pedir_datos()
        if X is None:
            return
        reales = self.red.indices_de_clases(y)
        predichas = self.red.predecir_indices(X)
        self.graficar(X, reales, predichas, "Evaluar")
        self.imprimir_hiperparametros()
        plt.show()

    def guardar_red(self):
        ruta = input("Nombre del archivo donde guardar la red (ej. red.csv): ").strip()
        if not ruta:
            print("  No se guardó la red. Ingrese un nombre de archivo.")
            return
        if not ruta.lower().endswith(".csv"):
            ruta += ".csv"
        try:
            self.red.guardar(ruta)
            print("Red guardada en", ruta)
        except OSError as error:
            print("  No se pudo guardar:", error)
        self.imprimir_hiperparametros()

    def imprimir_hiperparametros(self):
        r = self.red
        print("---- Hiperparámetros ----")
        print("  u (neuronas de entrada)     :", r.u)
        print("  v (neuronas de salida)      :", r.v)
        print("  L (capas ocultas)           :", r.L)
        print("  b (neuronas por capa oculta):", r.b)
        print("  e (épocas entrenadas)       :", r.e)
        print("  tasa de aprendizaje %.3g, momentum %.3g, lote %d" % (r.ALFA, r.BETA, r.LOTE))

    def menu_principal(self):
        while True:
            print("\n----- Perceptrón multicapa -----")
            print("""1) Crear un nuevo perceptrón multicapa
2) Cargar uno desde un archivo
3) Salir
                  """)
            opcion = input("Elige (1/2/3): ").strip()
            while opcion not in ("1", "2", "3"):
                opcion = input("  Opción inválida, elige 1, 2 o 3: ").strip()
            if opcion == "1":
                self.crear()
            elif opcion == "2":
                self._cargar()
            elif opcion == "3":
                print("Fin.")
                break
            self.imprimir_hiperparametros()

            while True:
                print("\n1) Entrenar con retropropagación\n2) Evaluar (hacia adelante)\n3) Guardar la red\n4) Salir")
                opcion = input("Elige (1-4): ").strip()
                if opcion == "1":
                    self.entrenar()
                elif opcion == "2":
                    self.evaluar()
                elif opcion == "3":
                    self.guardar_red()
                if opcion == "4":
                    print("Fin.")
                    break
                else:
                    print("  Opción inválida.")
                
# Archivo 3. Perceptron 
# import numpy as np
# from funciones import *

class Perceptron:
    # Constantes de entrenamiento. 
    # El CSV de la red solo guarda u, v, L, b, e así que estas las definí acá
    ALFA = 0.5          # tasa de aprendizaje
    BETA = 0.9          # momentum
    LOTE = 16           # tamaño del minilote (1 = aprendizaje en línea)
    NORMALIZAR = True   # entrena con entradas estandarizadas y luego "pliega"
                        # la normalización dentro de la primera capa
                        
    def __init__(self, u, v, L, b, rng=None, pesos=None, epocas=0):
        self.u = u
        self.v = v
        self.L = L
        self.b = b if L > 0 else 0 # Capas ocultas
        self.e = epocas # épocas entrenadas (acumuladas para la carga)
        self.etiquetas = None           # etiqueta de clase de cada neurona de salida
        if pesos is not None:
            self.w = pesos
            return
        if rng is None:
            rng = np.random.default_rng()
        tamaños = self.tamaños()
        self.w = []
        for l in range(len(tamaños) - 1):
            n_in, n_out = tamaños[l], tamaños[l + 1]
            limite = np.sqrt(6.0 / (n_in + n_out))
            capa = rng.uniform(-limite, limite, size=(n_out, n_in + 1))
            capa[:, 0] = 0.0            # los sesgos empiezan en 0
            self.w.append(capa)

    
    def tamaños(self):
        """Lista con el número de neuronas de cada capa: [u, b, ..., b, v]."""
        return [self.u] + [self.b] * self.L + [self.v]

    def hiperparametros(self):
        return {"u": self.u, "v": self.v, "L": self.L, "b": self.b, "e": self.e}

    def _inferir_etiquetas(self, unicas):
        """Decide qué etiqueta le toca a cada neurona de salida a partir de un archivo."""
        n_clases = max(self.v, 2)
        if len(unicas) > n_clases:
            raise ValueError(
                "El archivo tiene %d clases (%s) pero la red tiene %d neuronas de salida. "
                "Cree la red con %d neuronas de salida (una por clase)."
                % (len(unicas), ", ".join(str(int(c)) for c in unicas), self.v, len(unicas)))
        if len(unicas) == n_clases:
            return [int(c) for c in unicas]          # se usan tal cual, de menor a mayor
        # el archivo no trae todas las clases: se adivina el convenio
        if n_clases == 2 and set(unicas) <= {-1, 1}:
            return [-1, 1]
        if 0 in unicas and unicas.max() <= n_clases - 1:
            return list(range(n_clases))
        if unicas.min() >= 1 and unicas.max() <= n_clases:
            return list(range(1, n_clases + 1))
        raise ValueError("No se pudo asociar las clases %s con las %d neuronas de salida."
                         % (list(unicas), self.v))

    def indices_de_clases(self, y):
        """Convierte las etiquetas del archivo (p. ej. -1/1 o 1..4) en índices 0..v-1."""
        unicas = np.unique(y)
        if self.etiquetas is None:
            self.etiquetas = self._inferir_etiquetas(unicas)
        desconocidas = [int(c) for c in unicas if int(c) not in self.etiquetas]
        if desconocidas:
            raise ValueError("El archivo tiene clases %s que esta red no conoce (clases de la red: %s)."
                             % (desconocidas, self.etiquetas))
        mapa = {et: i for i, et in enumerate(self.etiquetas)}
        return np.array([mapa[int(c)] for c in y])

    def validar_datos(self, X, y):
        """Si el archivo no corresponde a la red, explica cómo configurarla."""
        n_entradas = X.shape[1]
        n_clases = len(np.unique(y))
        if n_entradas != self.u:
            raise ValueError(
                "El archivo tiene %d valores de entrada y la red tiene %d neuronas de entrada. "
                "Para estos datos cree la red con %d neuronas de entrada y %d de salida."
                % (n_entradas, self.u, n_entradas, max(n_clases, 2)))
        # Las clases del archivo deben caber en las neuronas de salida (no cambia la red).
        etiquetas = self.etiquetas
        if etiquetas is None:
            etiquetas = self._inferir_etiquetas(np.unique(y))
        desconocidas = [int(c) for c in np.unique(y) if int(c) not in etiquetas]
        if desconocidas:
            raise ValueError("El archivo tiene clases %s que esta red no conoce (clases de la red: %s)."
                             % (desconocidas, etiquetas))

    def codificar_objetivos(self, indices):
        """Índice de clase -> vector objetivo. v == 1: el propio índice. v > 1: one-hot."""
        if self.v == 1:
            return indices.reshape(-1, 1).astype(float)
        T = np.zeros((len(indices), self.v))
        T[np.arange(len(indices)), indices] = 1.0
        return T

    
    # -------------------- Propagación hacia adelante y predicción --------------------
    def propagar_adelante(self, X):
        """Devuelve la lista de activaciones [A0 = X, A1, ..., A_salida]. X es (N, u)."""
        activaciones = [X]
        for capa in self.w:
            Z = con_unos(activaciones[-1]) @ capa.T       # Z = b + suma(w * a)
            activaciones.append(sigmoide(Z))              # A = f(Z)
        return activaciones

    def predecir_indices(self, X):
        """Índice de clase predicho: umbral 0.5 si v == 1, o la neurona de mayor salida."""
        salida = self.propagar_adelante(X)[-1]
        if self.v == 1:
            return (salida[:, 0] > 0.5).astype(int)
        return np.argmax(salida, axis=1)

    def predecir(self, X):
        """Etiqueta de clase predicha (la misma que usa el archivo de datos)."""
        etiquetas = self.etiquetas if self.etiquetas is not None else list(range(max(self.v, 2)))
        return np.array(etiquetas)[self.predecir_indices(X)]

    def error_mse(self, X, T):
        """Error cuadrático medio por muestra: 1/2 * suma((t - o)^2), promediado."""
        salida = self.propagar_adelante(X)[-1]
        return float(np.mean(0.5 * np.sum((T - salida) ** 2, axis=1)))

    
    # -------------------- Entrenamiento --------------------
    def retropropagar(self, activaciones, T):
        """
        Gradiente de E respecto a cada matriz W (promedio del minilote).
        delta de salida:  (o - t) * o * (1 - o)
        delta oculta:     (W_sig^T delta_sig) * a * (1 - a)
        gradiente de W:   delta^T * [1, a_anterior] / N
        """
        N = T.shape[0]
        salida = activaciones[-1]
        delta = (salida - T) * salida * (1.0 - salida)
        gradientes = [None] * len(self.w)
        for l in range(len(self.w) - 1, -1, -1):
            gradientes[l] = delta.T @ con_unos(activaciones[l]) / N
            if l > 0:
                a = activaciones[l]
                delta = (delta @ self.w[l][:, 1:]) * a * (1.0 - a)   # [:, 1:] deja fuera el sesgo
        return gradientes

    def _normalizar_primera_capa(self, mu, sigma, hacia_normalizado):
        """
        Cambia la primera capa para trabajar con entradas estandarizadas z = (x - mu) / sigma
        (hacia_normalizado=True) o para volver a las entradas originales (False).
        Es un cambio exacto: la red calcula lo mismo, solo en otra escala.
        """
        W1 = self.w[0]
        if hacia_normalizado:
            W1[:, 0] = W1[:, 0] + W1[:, 1:] @ mu
            W1[:, 1:] = W1[:, 1:] * sigma
        else:
            W1[:, 1:] = W1[:, 1:] / sigma
            W1[:, 0] = W1[:, 0] - W1[:, 1:] @ mu

    def entrenar(self, X, y, epocas, rng=None, mostrar=True):
        """
        Retropropagación: descenso de gradiente con momentum y minilotes.
        y trae las etiquetas del archivo. Devuelve el error de cada época y suma las
        épocas a self.e.
        """
        if rng is None:
            rng = np.random.default_rng()
        T = self.codificar_objetivos(self.indices_de_clases(y))
        N = X.shape[0]
        if self.NORMALIZAR:
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            sigma[sigma == 0] = 1.0
            if self.e > 0:    # red ya entrenada: pasar sus pesos a la escala normalizada
                self._normalizar_primera_capa(mu, sigma, True)
            # (red sin entrenar: sus pesos aleatorios ya se interpretan en la escala normalizada)
            Xe = (X - mu) / sigma
        else:
            Xe = X
        velocidad = [np.zeros_like(capa) for capa in self.w]
        historial = []
        paso = max(1, epocas // 10)
        for epoca in range(1, epocas + 1):
            orden = rng.permutation(N)                     # se baraja en cada época
            for inicio in range(0, N, self.LOTE):
                idx = orden[inicio:inicio + self.LOTE]
                activaciones = self.propagar_adelante(Xe[idx])
                gradientes = self.retropropagar(activaciones, T[idx])
                for l in range(len(self.w)):
                    velocidad[l] = self.BETA * velocidad[l] - self.ALFA * gradientes[l]
                    self.w[l] += velocidad[l]
            historial.append(self.error_mse(Xe, T))
            if mostrar and (epoca == 1 or epoca % paso == 0 or epoca == epocas):
                print("  Época %d/%d  error = %.5f" % (epoca, epocas, historial[-1]))
        if self.NORMALIZAR:
            self._normalizar_primera_capa(mu, sigma, False)
        self.e += epocas
        return historial

    
    # -------------------- Guardado y cargado --------------------
    def guardar(self, ruta):
        """
        Encabezado + 5 filas de hiperparámetros (u, v, L, b, e) + una fila por neurona:
        tipo (h/o), capa (1 = primera oculta), posición (desde 0), pesos (el primero es el sesgo).
        """
        c = max(self.u, self.b)
        ancho = 3 + (c + 1)                      # columnas 0..c+3
        filas = [["tipo", "capa", "posicion"] + ["peso_%d" % i for i in range(c + 1)]]
        for letra, valor in self.hiperparametros().items():
            fila = [letra, str(valor)]
            filas.append(fila + [""] * (ancho - len(fila)))
        n_capas = len(self.w)
        for l, capa in enumerate(self.w):
            tipo = "o" if l == n_capas - 1 else "h"
            for pos in range(capa.shape[0]):
                fila = [tipo, str(l + 1), str(pos)] + [repr(float(w)) for w in capa[pos]]
                filas.append(fila + [""] * (ancho - len(fila)))
        with open(ruta, "w", encoding="utf-8") as archivo:
            for fila in filas:
                archivo.write(",".join(fila) + "\n")

    def cargar(cls, ruta):
        """Lee un archivo de red y devuelve un Perceptron. Reconoce cada fila por su primera columna."""
        hiper = {}
        neuronas = {}                            # (capa, posicion) -> pesos
        with open(ruta, encoding="utf-8-sig") as archivo:
            for linea in archivo:
                linea = linea.strip()
                if linea == "":
                    continue
                separador = ";" if (";" in linea and "," not in linea) else ","
                partes = [p.strip() for p in linea.split(separador)]
                marca = partes[0]
                if marca in ("u", "v", "L", "b", "e"):
                    hiper[marca] = int(a_numero(partes[1]))
                elif marca in ("h", "o"):
                    capa, pos = int(partes[1]), int(partes[2])
                    neuronas[(capa, pos)] = [a_numero(p) for p in partes[3:] if p != ""]
                # cualquier otra cosa (el encabezado) se ignora
        for letra in ("u", "v", "e"):
            if letra not in hiper:
                raise ValueError("Al archivo le falta el hiperparámetro '%s'." % letra)
        if not neuronas:
            raise ValueError("El archivo no tiene neuronas.")
        n_capas = max(capa for capa, _ in neuronas)
        tamaños = [hiper["u"]]
        for capa in range(1, n_capas + 1):
            tamaños.append(1 + max(pos for c_, pos in neuronas if c_ == capa))
        W = []
        for capa in range(1, n_capas + 1):
            n_in = tamaños[capa - 1]
            matriz = np.zeros((tamaños[capa], n_in + 1))
            for pos in range(tamaños[capa]):
                if (capa, pos) not in neuronas:
                    raise ValueError("Falta la neurona %d de la capa %d." % (pos, capa))
                pesos = neuronas[(capa, pos)]
                if len(pesos) < n_in + 1:
                    raise ValueError("La neurona %d de la capa %d tiene %d pesos y necesita %d."
                                     % (pos, capa, len(pesos), n_in + 1))
                matriz[pos] = pesos[:n_in + 1]
            W.append(matriz)
        if tamaños[-1] != hiper["v"]:
            raise ValueError("El archivo dice v=%d pero la última capa tiene %d neuronas."
                             % (hiper["v"], tamaños[-1]))
        L = n_capas - 1
        b = tamaños[1] if L > 0 else 0
        return cls(hiper["u"], hiper["v"], L, b, pesos=W, epocas=hiper["e"]) 
    
# Archivo 4. funciones
import numpy as np

def a_numero(texto):
        return float(texto.strip().replace(",", "."))
    
def sigmoide(z):
    return 1 / (1 + 2.718281 ** (-z))

def con_unos(A):
    """Agrega la columna de unos (entrada 0) que multiplica al sesgo."""
    return np.hstack([np.ones((A.shape[0], 1)), A])


#Ejecución
main() 