print("Hola mundo")

numero = int(input("Inserte el numero: "))

def es_par(numero):
    return (numero & 1) == 0

print(es_par(numero))
