# Nombre del integrante: Giselle Esclasans 
# Cédula del integrante: 30715145
# haga su tarea aqui
"""
Tarea 1 - Perceptron
Computacion Emergente (FPTSP25)

Programa de consola que permite a un usuario interactuar con un perceptron:
carga un archivo CSV, pide al usuario el sesgo, los pesos de entrada y la
funcion de activacion, y grafica los resultados.

Unica libreria externa utilizada: matplotlib.
"""

import matplotlib.pyplot as plt


E = 2.718281828459045


## calcula e**x con serie de taylor, sin usar math.exp
def exponencial(x):
    if x < 0:
        return 1.0 / exponencial(-x)

    parte_entera = int(x)
    resto = x - parte_entera

    termino = 1.0
    acumulado = 1.0
    i = 1
    while i <= 25:
        termino = termino * resto / i
        acumulado = acumulado + termino
        i = i + 1

    resultado = acumulado
    i = 0
    while i < parte_entera:
        resultado = resultado * E
        if resultado > 1e300:
            return 1e300
        i = i + 1

    return resultado


## retorna sesgo + suma de entradas[i] * pesos[i]
def suma(entradas, pesos, sesgo):
    total = sesgo
    i = 0
    while i < len(entradas):
        total = total + entradas[i] * pesos[i]
        i = i + 1
    return total


## funcion escalon: salida 0 / 1
def escalon(x):
    if x >= 0:
        return 1.0
    return 0.0


## funcion sigmoide: salida continua en (0, 1)
def sigmoide(x):
    if x >= 0:
        return 1.0 / (1.0 + exponencial(-x))
    ex = exponencial(x)
    return ex / (1.0 + ex)


ACTIVACIONES = {
    "1": {
        "nombre": "Escalon (salida 0 / 1)",
        "funcion": escalon,
        "umbral": 0.5,
    },
    "2": {
        "nombre": "Sigmoide (salida continua en 0..1)",
        "funcion": sigmoide,
        "umbral": 0.5,
    },
}


## indca si el texto puede convertirse a float
def es_numero(texto):
    try:
        float(texto)
        return True
    except ValueError:
        return False


## leel archivo csv y retorna (entradas, esperados)
def leer_csv(ruta):
    try:
        archivo = open(ruta, "r", encoding="utf-8-sig")
        lineas = archivo.readlines()
        archivo.close()
    except OSError:
        print("  -> Error: no se pudo abrir el archivo. Verifique la ruta.")
        return None, None

    entradas = []
    esperados = []
    numero_de_linea = 0

    for linea in lineas:
        numero_de_linea = numero_de_linea + 1
        linea = linea.strip()

        if linea == "":
            continue

        if linea.count(";") > linea.count(","):
            campos = linea.split(";")
        else:
            campos = linea.split(",")

        if len(campos) < 2:
            print("  -> Error: la linea %d tiene menos de 2 columnas." % numero_de_linea)
            return None, None

        if numero_de_linea == 1 and not es_numero(campos[0]):
            continue

        valores = []
        for campo in campos:
            if not es_numero(campo):
                print("  -> Error: valor no numerico en la linea %d." % numero_de_linea)
                return None, None
            valores.append(float(campo))

        entradas.append(valores[:-1])
        esperados.append(valores[-1])

    if len(entradas) == 0:
        print("  -> Error: el archivo no contiene datos.")
        return None, None

    for vector in entradas:
        if len(vector) != len(entradas[0]):
            print("  -> Error: todas las lineas deben tener el mismo numero de columnas.")
            return None, None

    return entradas, esperados


## obtiene las dos etiquetas de salida que usa el archivo (clase baja y alta)
def etiquetas_del_conjunto(esperados):
    distintos = []
    for v in esperados:
        if v not in distintos:
            distintos.append(v)
    distintos.sort()

    if len(distintos) < 2:
        return distintos[0], distintos[0]

    if len(distintos) > 2:
        print("\nAdvertencia: la columna de salida tiene %d valores distintos."
              % len(distintos))
        print("Un perceptron es un clasificador binario; se usaran %g y %g "
              "como las dos clases." % (distintos[0], distintos[-1]))

    return distintos[0], distintos[-1]


## aplica el perceptron a cada vector y retorna la lista de valores predichos
def ejecutar_perceptron(entradas, sesgo, pesos, activacion, etiquetas):
    funcion = activacion["funcion"]
    umbral = activacion["umbral"]
    etiqueta_baja, etiqueta_alta = etiquetas

    predichos = []

    for vector in entradas:
        neta = suma(vector, pesos, sesgo)
        y = funcion(neta)
        if y >= umbral:
            predichos.append(etiqueta_alta)
        else:
            predichos.append(etiqueta_baja)

    return predichos


## retorna una lista de booleanos que indica si cada prediccion coincide
def comparar(esperados, predichos):
    coincidencias = []

    i = 0
    while i < len(esperados):
        coincidencias.append(abs(esperados[i] - predichos[i]) < 1e-9)
        i = i + 1

    return coincidencias


## pide un numero real al usuario hasta que escriba uno valido
def pedir_numero(mensaje):
    while True:
        texto = input(mensaje).strip().replace(",", ".")
        if es_numero(texto):
            return float(texto)
        print("  -> Eso no es un numero valido. Intente de nuevo.")


## pide el sescgo y un peso por cada columna de entradaa
def pedir_pesos(cantidad_entradas):
    print("\nIntroduzca los pesos del perceptron:")
    sesgo = pedir_numero("  Peso del sesgo (w0): ")

    pesos = []
    i = 0
    while i < cantidad_entradas:
        pesos.append(pedir_numero("  Peso de la columna %d (w%d): " % (i + 1, i + 1)))
        i = i + 1

    return sesgo, pesos


## muesta el menu de funciones de activacion y retorna la escogi
def pedir_activacion():
    print("\nFunciones de activacion disponibles:")
    for clave in sorted(ACTIVACIONES.keys()):
        print("  %s) %s" % (clave, ACTIVACIONES[clave]["nombre"]))

    while True:
        opcion = input("Escoja la funcion de activacion: ").strip()
        if opcion in ACTIVACIONES:
            return ACTIVACIONES[opcion]
        print("  -> Opcion invalida. Escriba uno de los numeros del menu.")


## obtiene las coordenadas (x, y) de las dos primeras columnas para graficar
def coordenadas(entradas):
    xs = []
    ys = []

    for vector in entradas:
        xs.append(vector[0])
        if len(vector) >= 2:
            ys.append(vector[1])
        else:
            ys.append(0.0)

    return xs, ys


## calcula los limites de los ejes con un margen
def limites(entradas):
    xs, ys = coordenadas(entradas)

    margen_x = (max(xs) - min(xs)) * 0.12 + 0.4
    margen_y = (max(ys) - min(ys)) * 0.12 + 0.4

    return (min(xs) - margen_x, max(xs) + margen_x,
            min(ys) - margen_y, max(ys) + margen_y)


## asigna un color a cada valor distinto de la lista
def colores_por_clase(valores):
    paleta = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    distintos = []
    for v in valores:
        if v not in distintos:
            distintos.append(v)
    distintos.sort()

    colores = []
    for v in valores:
        colores.append(paleta[distintos.index(v) % len(paleta)])

    return colores, distintos, paleta


## grafico de dispersion inicial con los datos del csv
def graficar_datos(entradas, esperados, ruta):
    xs, ys = coordenadas(entradas)
    colores, distintos, paleta = colores_por_clase(esperados)

    plt.figure("Datos cargados")
    plt.scatter(xs, ys, c=colores, edgecolors="black", s=90)
    plt.title("Datos de %s\n(color = salida esperada)" % ruta)
    plt.xlabel("Columna 1")
    if len(entradas[0]) >= 2:
        plt.ylabel("Columna 2")
    else:
        plt.ylabel("(dimension unica)")
    plt.grid(True, linestyle=":", alpha=0.5)

    for i in range(len(distintos)):
        plt.scatter([], [], c=paleta[i % len(paleta)], edgecolors="black",
                    s=90, label="esperado = %g" % distintos[i])

    caja = limites(entradas)
    plt.xlim(caja[0], caja[1])
    plt.ylim(caja[2], caja[3])
    plt.legend(loc="upper left")

    print("\nCierre la ventana del grafico para continuar...")
    plt.show()


## crea los tres graficos: esperado, predicho y coincidencias
def graficar_resultados(entradas, esperados, predichos, coincidencias,
                        sesgo, pesos, activacion):
    xs, ys = coordenadas(entradas)

    todas = esperados + predichos
    _, distintos, paleta = colores_por_clase(todas)

    ## retorna el color asignado a un valor
    def color_de(valor):
        return paleta[distintos.index(valor) % len(paleta)]

    etiqueta_y = "Columna 2" if len(entradas[0]) >= 2 else "(dimension unica)"
    aciertos = coincidencias.count(True)
    caja = limites(entradas)

    ## aplica el mismo formato y limites a cada subgrafico
    def ajustar(titulo):
        plt.title(titulo)
        plt.xlabel("Columna 1")
        plt.ylabel(etiqueta_y)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.xlim(caja[0], caja[1])
        plt.ylim(caja[2], caja[3])
        plt.legend(loc="upper left", fontsize="small")

    plt.figure("Resultados del perceptron", figsize=(15, 5))
    plt.suptitle(
        "Activacion: %s   |   sesgo = %g, pesos = %s   |   aciertos: %d/%d"
        % (activacion["nombre"], sesgo,
           "[" + ", ".join("%g" % p for p in pesos) + "]",
           aciertos, len(coincidencias))
    )

    plt.subplot(1, 3, 1)
    plt.scatter(xs, ys, c=[color_de(v) for v in esperados],
                edgecolors="black", s=90)
    for i in range(len(distintos)):
        plt.scatter([], [], c=paleta[i % len(paleta)], edgecolors="black",
                    s=90, label="esperado = %g" % distintos[i])
    ajustar("1. Valor esperado")

    plt.subplot(1, 3, 2)
    plt.scatter(xs, ys, c=[color_de(v) for v in predichos],
                edgecolors="black", s=90)
    for i in range(len(distintos)):
        plt.scatter([], [], c=paleta[i % len(paleta)], edgecolors="black",
                    s=90, label="predicho = %g" % distintos[i])
    ajustar("2. Valor predicho por el perceptron")

    plt.subplot(1, 3, 3)
    colores = []
    for acierto in coincidencias:
        if acierto:
            colores.append("green")
        else:
            colores.append("red")
    plt.scatter(xs, ys, c=colores, edgecolors="black", s=90)
    plt.scatter([], [], c="green", edgecolors="black", s=90, label="coincide")
    plt.scatter([], [], c="red", edgecolors="black", s=90, label="no coincide")
    ajustar("3. Esperado vs. predicho")

    print("Cierre la ventana del grafico para continuar...")
    plt.show()


## programa principal
def main():
    print("=" * 78)
    print("TAREA 1 - PERCEPTRON")
    print("Computacion Emergente (FPTSP25)")
    print("=" * 78)

    entradas = None
    while entradas is None:
        ruta = input("\nRuta del archivo CSV: ").strip().strip('"')
        entradas, esperados = leer_csv(ruta)

    cantidad_entradas = len(entradas[0])
    n = cantidad_entradas + 1
    print("\nSe cargaron %d vectores de %d columnas (n = %d)."
          % (len(entradas), n, n))
    print("Columnas de entrada: %d   |   Columna de salida esperada: 1"
          % cantidad_entradas)
    if cantidad_entradas > 2:
        print("Nota: n > 3, por lo que solo se graficaran las dos primeras "
              "dimensiones del vector de entrada.")

    etiquetas = etiquetas_del_conjunto(esperados)
    print("Clases encontradas en la salida esperada: %g y %g"
          % (etiquetas[0], etiquetas[1]))

    graficar_datos(entradas, esperados, ruta)

    while True:
        sesgo, pesos = pedir_pesos(cantidad_entradas)
        activacion = pedir_activacion()

        predichos = ejecutar_perceptron(entradas, sesgo, pesos, activacion,
                                        etiquetas)
        coincidencias = comparar(esperados, predichos)

        graficar_resultados(entradas, esperados, predichos,
                            coincidencias, sesgo, pesos, activacion)

        respuesta = input("\nDesea probar con otros pesos? (s/n): ").strip().lower()
        if respuesta not in ("s", "si", "sí", "y", "yes"):
            break

    print("\nFin del programa.")


if __name__ == "__main__":
    main()

