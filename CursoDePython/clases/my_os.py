import os

# Obtener el directorio actual
"""cwd = os.getcwd()
print("Directorio de trabajo actual", cwd)"""

#Listar los archivos .txt
txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
print("Los Archivos txt: ", txt_files)

#Renombrar archivos .txt
os.rename('cuento_1.txt', 'caperucita.txt')
print('Archivo renombrado')

txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
print("Los Archivos txt: ", txt_files)

