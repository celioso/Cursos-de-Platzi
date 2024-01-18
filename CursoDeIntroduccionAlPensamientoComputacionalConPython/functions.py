print("Funciones y abstracción")
objetivo = int(input("Escoge un numero: "))
print("Porque metodo te gustaria encontrar la raíz cuadara?: ")
print("1. Enumeracion exhaustiva.")
print("2. Aproximacion")
print("3. Busqueda Binaria")
opcion = int(input("Escribe el numero de la opcion que deseas: "))

def enumeracion_exhaustiva(objetivo):
    respuesta = 0

    while respuesta**2 < objetivo:
        respuesta += 1

    if respuesta**2 == objetivo:
        return print(f"La raíz cuadrada de {objetivo} es {respuesta}")
    else:
        return print(f"{objetivo} no tiene una raiz cuadrada exacta")

def aproximacion(objetivo):
    epsilon = 0.01
    paso = epsilon**2
    respuesta = 0.0

    while abs(respuesta**2 - objetivo) >= epsilon and respuesta <= objetivo:
        respuesta += paso

    if abs(respuesta**2 - objetivo) >= epsilon:
        return print(f"No se encontro la raiz cuadrada de {objetivo}")
    else:
        return print(f"La raíz cuadrada de {objetivo} es {respuesta}")

def busqueda_binaria(objetivo):
    epsilon = 0.001
    bajo = 0.0
    lto = max(1.0, objetivo)
    respuesta = (alto + bajo) / 2

    while abs(respuesta**2 - objetivo) >= epsilon:
        if respuesta**2 < objetivo:
            bajo = respuesta
        else:
            alto = respuesta
    respuesta = (alto + bajo) / 2

    return print(f"La raíz cuadrada de {objetivo} es {respuesta}")

if opcion == 1:
    enumeracion_exhaustiva(objetivo)
elif opcion == 2:
    aproximacion(objetivo)
elif opcion == 3:
    busqueda_binaria(objetivo)
else:
    print("Opción no valida.")