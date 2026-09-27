# Nombre del integrante: Daniela Zambrano
# Cédula del integrante: 30956881

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def es_fila_numerica(campos):
    """
    Determina si todos los campos de una fila se pueden interpretar como
    numeros. Para evitar encabezados.
    """
    for campo in campos:
        try:
            float(campo)
        except ValueError:
            return False
    return True


def cargar_csv(ruta):
    """
    Lee un archivo CSV donde cada fila tiene n valores 
    numericos: las primeras n-1 columnas son la
    entrada y la ultima es la salida esperada.

    Si la primera fila es un título o texto, se ignora automaticamente.

    Retorna dos listas:
        entradas  -> lista de listas (cada una con n-1 valores float)
        esperados -> lista de floats (la salida esperada de cada fila)
    """
    entradas = []
    esperados = []

    with open(ruta) as archivo:
        lineas = archivo.readlines()

    for i, linea in enumerate(lineas):
        linea = linea.strip()
        if linea == "":
            continue  # ignorar lineas vacias

        campos = [campo.strip() for campo in linea.split(",")]

        if i == 0 and not es_fila_numerica(campos):
            continue

        valores = [float(campo) for campo in campos]
        entradas.append(valores[:-1])
        esperados.append(valores[-1])

    return entradas, esperados


# ---------------------------------------------------------------------------
# Funcion suma (combinacion lineal: sesgo + pesos * entradas)
# ---------------------------------------------------------------------------

def funcion_suma(entrada, pesos, sesgo):
    """
    Calcula la suma ponderada: sesgo + w1*x1 + w2*x2 + ... + wn*xn
    """
    total = sesgo
    for xi, wi in zip(entrada, pesos):
        total += xi * wi
    return total


# ---------------------------------------------------------------------------
# Funciones de activacion
# ---------------------------------------------------------------------------

def activacion_escalon(z):
    """
    Funcion escalon (step function).
    Retorna 1 si z >= 0, 0 en caso contrario.
    """
    if z >= 0:
        return 1
    else:
        return 0


def activacion_signo(z):
    """
    Funcion signo (sign function).
    Retorna 1 si z >= 0, -1 en caso contrario.
    """
    if z >= 0:
        return 1
    else:
        return -1


ACTIVACIONES = {
    "1": ("Escalon (0/1)", activacion_escalon),
    "2": ("Signo (-1/1)", activacion_signo),
}


# ---------------------------------------------------------------------------
# Entrada de usuario
# ---------------------------------------------------------------------------

def pedir_float(mensaje):
    while True:
        texto = input(mensaje)
        try:
            return float(texto)
        except ValueError:
            print("  -> Valor invalido, intente de nuevo (ej: 0.5 o -1.2)")


def pedir_pesos(n_columnas):
    """
    Pide al usuario el peso del sesgo y el peso de cada una de las
    n_columnas de entrada. Retorna (sesgo, [pesos...]).
    """
    print("\n--- Ingreso de pesos ---")
    sesgo = pedir_float("Peso del sesgo: ")
    pesos = []
    for i in range(n_columnas):
        w = pedir_float(f"Peso para la columna {i + 1}: ")
        pesos.append(w)
    return sesgo, pesos


def pedir_activacion():
    """Pide al usuario que escoja una funcion de activacion."""
    print("\n--- Funcion de activacion ---")
    print("IMPORTANTE: La función debe coincidir con el formato de salida de sus datos.")
    print(" - Use 'Escalon (0/1)' si los valores esperados en su dataset son 0 y 1.")
    print(" - Use 'Signo (-1/1)' si los valores esperados en su dataset son -1 y 1.")
    print("Si usa la función incorrecta (ej. datos de 0/1 con función de Signo), el perceptrón no acertará las predicciones.\n")
    
    for clave, (nombre, _) in ACTIVACIONES.items():
        print(f"  {clave}) {nombre}")
    while True:
        opcion = input("Escoja una opcion: ").strip()
        if opcion in ACTIVACIONES:
            nombre, funcion = ACTIVACIONES[opcion]
            return nombre, funcion
        print("  -> Opcion invalida, intente de nuevo.")


# ---------------------------------------------------------------------------
# Prediccion
# ---------------------------------------------------------------------------

def predecir_todo(entradas, pesos, sesgo, funcion_activacion):
    """
    Aplica el perceptron a cada vector de entrada.
    Retorna una lista con la salida predicha para cada vector.
    """
    predicciones = []
    for entrada in entradas:
        z = funcion_suma(entrada, pesos, sesgo)
        y_pred = funcion_activacion(z)
        predicciones.append(y_pred)
    return predicciones


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------

def coordenadas_para_graficar(entradas):
    """
    Retorna las coordenadas (x, y) a usar en los graficos.
    Si hay mas de 2 dimensiones de entrada, solo se usan las primeras 2.
    Si solo hay 1 dimension, se usa esa como x y 0 como y.
    """
    xs = []
    ys = []
    for entrada in entradas:
        xs.append(entrada[0])
        if len(entrada) >= 2:
            ys.append(entrada[1])
        else:
            ys.append(0)
    return xs, ys

def graficar_resultados(entradas, esperados, predicciones, nombre_activacion):
    """
    Crea 3 graficos de dispersion (scatter) con leyendas descriptivas
    que se adaptan a la función de activación elegida.
    """
    xs, ys = coordenadas_para_graficar(entradas)

    fig, ejes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Elementos visuales para las leyendas dinámicas ---
    if "Signo" in nombre_activacion:
        label_azul = 'Clase -1 (Azul)'
        label_rojo = 'Clase 1 (Rojo)'
    else:
        label_azul = 'Clase 0 (Azul)'
        label_rojo = 'Clase 1 (Rojo)'

    leyenda_clases = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4575b4', markersize=10, label=label_azul),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d73027', markersize=10, label=label_rojo)
    ]
    leyenda_coincidencias = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Acierto (Verde)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Fallo (Rojo)')
    ]

    # Grafico 1: valor esperado
    ejes[0].scatter(xs, ys, c=esperados, cmap="coolwarm", s=50)
    ejes[0].set_title("Valor esperado")
    ejes[0].set_xlabel("x1")
    ejes[0].set_ylabel("x2")
    ejes[0].legend(handles=leyenda_clases, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Grafico 2: valor predicho
    ejes[1].scatter(xs, ys, c=predicciones, cmap="coolwarm", s=50)
    ejes[1].set_title("Valor predicho")
    ejes[1].set_xlabel("x1")
    ejes[1].set_ylabel("x2")
    ejes[1].legend(handles=leyenda_clases, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Grafico 3: coincidencia (verde) / no coincidencia (rojo)
    colores = []
    for esperado, predicho in zip(esperados, predicciones):
        if esperado == predicho:
            colores.append("green")
        else:
            colores.append("red")
            
    ejes[2].scatter(xs, ys, c=colores, s=50)
    ejes[2].set_title("Coincidencia (verde) / No coincidencia (rojo)")
    ejes[2].set_xlabel("x1")
    ejes[2].set_ylabel("x2")
    ejes[2].legend(handles=leyenda_coincidencias, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plt.show()    
# ---------------------------------------------------------------------------
# Procedimiento principal
# ---------------------------------------------------------------------------
def imprimir_tabla(entradas, esperados, predicciones):
    """
    Muestra en la terminal una tabla comparativa con los valores de entrada, 
    el resultado esperado, el resultado predicho y si hubo acierto.
    """
    print("\n--- Tabla de Resultados Detallada ---")
    # Usamos formato de cadenas para alinear las columnas como una tabla
    print(f"{'Entradas (x)':<30} | {'Esperado (y)':<12} | {'Predicción':<12} | {'¿Acierto?'}")
    print("-" * 75)
    
    for x, y, pred in zip(entradas, esperados, predicciones):
        acierto = "Sí" if y == pred else "No"
        # Convertimos la lista de entradas a texto (redondeando para que se vea limpio)
        x_str = str([round(val, 4) for val in x]) 
        
        print(f"{x_str:<30} | {y:<12} | {pred:<12} | {acierto}")

def main():
    print("=== Perceptrón - Tarea 1 ===\n")
    print("=" * 65)
    print("           SIMULADOR BÁSICO DE PERCEPTRÓN - TAREA 1")
    print("=" * 65)
    print("Daniela Zambrano C.I: 30956881\n")
    print("Este programa implementa un perceptrón de una sola capa.")
    print("Permite evaluar un conjunto de datos (cargado desde un archivo CSV)")
    print("utilizando diferentes pesos, un sesgo (bias) y la función de")
    print("activación que mejor se adapte a su tipo de datos (Escalón o Signo).")
    print("Al finalizar, se mostrará una tabla de aciertos y una representación")
    print("gráfica de los resultados esperados frente a las predicciones.")
    print("-" * 65 + "\n")

    ruta = input("Ingrese la ruta del archivo CSV: ").strip()
    entradas, esperados = cargar_csv(ruta)

    n_columnas = len(entradas[0])  # n-1 columnas de entrada
    print(f"\nSe cargaron {len(entradas)} vectores con {n_columnas} "
        f"columna(s) de entrada cada uno.")

    seguir = True
    while seguir:
        sesgo, pesos = pedir_pesos(n_columnas)
        nombre_activacion, funcion_activacion = pedir_activacion()

        predicciones = predecir_todo(entradas, pesos, sesgo, funcion_activacion)

        aciertos = sum(
            1 for e, p in zip(esperados, predicciones) if e == p
        )
        print(f"\nActivacion usada: {nombre_activacion}")
        print(f"Aciertos: {aciertos} / {len(esperados)}")

        imprimir_tabla(entradas, esperados, predicciones)
        
        graficar_resultados(entradas, esperados, predicciones, nombre_activacion)

        respuesta = input("\nDesea probar con otros pesos? (s/n): ").strip().lower()
        seguir = respuesta == "s"

    print("\nPrograma finalizado.")


if __name__ == "__main__":
    main()

