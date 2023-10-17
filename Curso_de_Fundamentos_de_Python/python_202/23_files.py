#para abrir un archivo
file = open('./text.txt')
# leer el archivo
# print(file.read())
# leer las lineas del archivo
# print(file.readline())
# print(file.readline())
# print(file.readline())
# print(file.readline())

for line in file:
  print(line)
# cerrar archivo
file.close()

#abrir , leer y cerrar archivos automaticamente

with open('./text.txt') as file:
  for line in file:
    print(line)