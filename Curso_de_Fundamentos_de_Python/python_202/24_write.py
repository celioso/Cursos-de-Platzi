with open('./text.txt', 'r+') as file:
  for line in file:
    print(line)
  file.write('nuevas cosas en este archivo\n')
  file.write('otra linea\n')
  file.write(' mas linea\n')

#w+ permite leer el archivo, pero al agregar write sobreescribe el archivo
 #r+ agrega el texto al final del archivo