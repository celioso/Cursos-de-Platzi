squares = [x**4 for x in range(1, 11)]
# print("Cuadrados: ", squares)

celsius = [0, 10, 20, 30, 40]
fahrenheit = [(temp *9/5)*32 for temp in celsius]
# print("Temperatura en F: ", fahrenheit)

# Numeros pares

evens = [x for x in range(1,21) if x%2==0]
# print(evens)

matrix = ([1, 2, 3],
          [4, 5, 6],
          [7, 8, 9])

transposed = [[raw[i] for raw in matrix] for i in range(len(matrix[0]))]
print(matrix)
print(transposed)

# Números primos

def es_primo(numero):
    if numero < 2:
        return False  # Los números menores que 2 no son primos
    for i in range(2, int(numero ** 0.5) + 1):  # Verificamos hasta la raíz cuadrada del número
        if numero % i == 0:
            return False
    return True

# Solicitar número al usuario
numero = int(input("Introduce un número: "))

# Verificar si el número es primo
if es_primo(numero):
    print(f"{numero} es un número primo.")
else:
    print(f"{numero} no es un número primo.")
