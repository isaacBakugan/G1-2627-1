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


def activacion_sigmoide(z):
    """
    Funcion sigmoide: 1 / (1 + e^(-z)).
    Su salida esta entre 0 y 1; se convierte a clase con umbral 0.5
    (retorna 1 si sigmoide(z) >= 0.5, 0 en caso contrario).
    """
    if z < -700:  # evita desbordamiento al calcular e^(-z)
        return 0.0
    return 1 / (1 + 2.718281828459045 ** (-z))


ACTIVACIONES = {
    "1": ("Escalon (0/1)", activacion_escalon),
    "2": ("Sigmoide (0/1)", activacion_sigmoide),
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

    for clave, (nombre, _) in ACTIVACIONES.items():
        print(f"  {clave}) {nombre}")
    while True:
        opcion = input("Escoja una opcion: ").strip()
        if opcion in ACTIVACIONES:
            nombre, funcion = ACTIVACIONES[opcion]
            umbral = 0.5 # valor por defecto
            if nombre == "Sigmoide":
                umbral = pedir_float("Ingrese el nivel de umbral (ej: 0.5 para clasificar): ")
            return nombre, funcion, umbral
        print("  -> Opcion invalida, intente de nuevo.")


# ---------------------------------------------------------------------------
# Prediccion
# ---------------------------------------------------------------------------

def predecir_todo(entradas, pesos, sesgo, funcion_activacion, nombre_activacion, umbral):
    """
    Retorna dos listas: 
    - valores_crudos: las probabilidades (para Sigmoide) o 0/1 (para escalón).
    - clases_predichas: las clasificaciones finales (0 o 1) aplicando el umbral.
    """
    valores_crudos = []
    clases_predichas = []
    
    for entrada in entradas:
        z = funcion_suma(entrada, pesos, sesgo)
        val = funcion_activacion(z)
        valores_crudos.append(val)
        
        if nombre_activacion == "Sigmoide":
            clases_predichas.append(1 if val >= umbral else 0)
        else:
            clases_predichas.append(val)
            
    return valores_crudos, clases_predichas


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------

def coordenadas_para_graficar(entradas):
    """
    Retorna las coordenadas (x, y) a usar en los graficos.
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

def graficar_resultados(entradas, esperados, valores_crudos, clases_predichas, nombre_activacion, umbral):
    """
    Crea 3 graficos de dispersion con leyendas descriptivas adaptadas.
    """
    xs, ys = coordenadas_para_graficar(entradas)

    fig, ejes = plt.subplots(1, 3, figsize=(15, 5))

    leyenda_clases_esperadas = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4575b4', markersize=10, label='Clase 0 (Azul)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d73027', markersize=10, label='Clase 1 (Rojo)')
    ]
    leyenda_coincidencias = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Acierto (Verde)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Fallo (Rojo)')
    ]

    # Grafico 1: valor esperado
    ejes[0].scatter(xs, ys, c=esperados, cmap="coolwarm", s=50, vmin=0, vmax=1)
    ejes[0].set_title("Valor esperado (Real)")
    ejes[0].set_xlabel("x1")
    ejes[0].set_ylabel("x2")
    ejes[0].legend(handles=leyenda_clases_esperadas, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Grafico 2: valor predicho (Probabilidades o Clases)
    ejes[1].scatter(xs, ys, c=valores_crudos, cmap="coolwarm", s=50, vmin=0, vmax=1)
    
    if nombre_activacion == "Sigmoide":
        ejes[1].set_title(f"Probabilidades (Umbral: {umbral})")
        leyenda_prediccion = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#4575b4', markersize=10, label='Tendencia a Clase 0'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#d73027', markersize=10, label='Tendencia a Clase 1')
        ]
    else:
        ejes[1].set_title("Clasificación del Perceptrón")
        leyenda_prediccion = leyenda_clases_esperadas

    ejes[1].set_xlabel("x1")
    ejes[1].set_ylabel("x2")
    ejes[1].legend(handles=leyenda_prediccion, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Grafico 3: coincidencia (verde) / no coincidencia (rojo)
    colores = ["green" if e == p else "red" for e, p in zip(esperados, clases_predichas)]
            
    ejes[2].scatter(xs, ys, c=colores, s=50)
    ejes[2].set_title("Coincidencias (Aciertos/Fallos)")
    ejes[2].set_xlabel("x1")
    ejes[2].set_ylabel("x2")
    ejes[2].legend(handles=leyenda_coincidencias, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plt.show()
    
# ---------------------------------------------------------------------------
# Procedimiento principal
# ---------------------------------------------------------------------------
def imprimir_tabla(entradas, esperados, valores_crudos, clases_predichas, nombre_activacion):
    """
    Muestra en la terminal una tabla comparativa adaptada a la función utilizada.
    """
    print("\n--- Tabla de Resultados Detallada ---")
    
    if nombre_activacion == "Sigmoide":
        print(f"{'Entradas (x)':<30} | {'Esperado':<8} | {'Probabilidad':<12} | {'Clase Pred.':<11} | {'¿Acierto?'}")
        print("-" * 80)
        for x, y, prob, clase_p in zip(entradas, esperados, valores_crudos, clases_predichas):
            acierto = "Sí" if y == clase_p else "No"
            x_str = str([round(val, 4) for val in x]) 
            print(f"{x_str:<30} | {y:<8} | {prob:<12.4f} | {clase_p:<11} | {acierto}")
    else:
        print(f"{'Entradas (x)':<30} | {'Esperado':<8} | {'Predicción':<12} | {'¿Acierto?'}")
        print("-" * 75)
        for x, y, pred, clase_p in zip(entradas, esperados, valores_crudos, clases_predichas):
            acierto = "Sí" if y == clase_p else "No"
            x_str = str([round(val, 4) for val in x]) 
            print(f"{x_str:<30} | {y:<8} | {clase_p:<12} | {acierto}")

def main():
    print("=== Perceptrón - Tarea 1 ===\n")
    print("=" * 65)
    print("           SIMULADOR BÁSICO DE PERCEPTRÓN - TAREA 1")
    print("=" * 65)
    print("Daniela Zambrano C.I: 30956881\n")
    print("Este programa implementa un perceptrón de una sola capa.")
    print("Permite evaluar un conjunto de datos (cargado desde un archivo CSV)")
    print("utilizando diferentes pesos, un sesgo (bias) y la función de")
    print("activación que se prefiera, Escalón o Sigmoide).")
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
        nombre_activacion, funcion_activacion, umbral = pedir_activacion()

        valores_crudos, clases_predichas = predecir_todo(entradas, pesos, sesgo, funcion_activacion, nombre_activacion, umbral)

        aciertos = sum(
            1 for e, p in zip(esperados, clases_predichas) if e == p
        )
        print(f"\nActivacion usada: {nombre_activacion}")
        if nombre_activacion == "Sigmoide":
            print(f"Umbral de clasificación: {umbral}")
        print(f"Aciertos: {aciertos} / {len(esperados)}")

        imprimir_tabla(entradas, esperados, valores_crudos, clases_predichas, nombre_activacion)
        
        graficar_resultados(entradas, esperados, valores_crudos, clases_predichas, nombre_activacion, umbral)

        respuesta = input("\nDesea probar con otros pesos? (s/n): ").strip().lower()
        seguir = respuesta == "s"

    print("\nPrograma finalizado.")


if __name__ == "__main__":
    main()

