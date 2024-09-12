#Leer un archivo línea por línea
"""with open("cuento.txt", "r") as file:
    for lineas in file:
        print(lineas.strip())"""

# Leer todas las líneas en una lista

"""with open("cuento.txt", "r") as file:
    lines = file.readlines()
    print(lines)"""

"""# Añadir texto al final del archivo
with open("cuento.txt", "a") as file:
    file.write("\n\nBy:ChatGPT")"""

"""# Sobreescribir el texto
with open("cuento_1.txt", "w") as file:
    file.write("\n\nBy:ChatGPT")"""

with open("cuento.txt", "r") as file:
    lineas = file.readlines()
    print("El total de lineas que tiene el archivo son: ",len(lineas))