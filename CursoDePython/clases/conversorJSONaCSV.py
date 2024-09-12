import json
import csv

def json_a_csv(json_file, csv_file):
    # Leer archivo JSON
    with open(json_file, mode='r') as archivo_json:
        datos = json.load(archivo_json)

    # Obtener los nombres de las columnas
    columnas = datos[0].keys()

    # Escribir archivo CSV
    with open(csv_file, mode='w', newline='') as archivo_csv:
        escritor_csv = csv.DictWriter(archivo_csv, fieldnames=columnas)
        escritor_csv.writeheader()
        escritor_csv.writerows(datos)

# Uso de la función
json_a_csv('archivo.json', 'archivo.csv')
