import csv
import json

def csv_a_json(csv_file, json_file):
    # Leer archivo CSV
    with open(csv_file, mode='r') as archivo_csv:
        lector_csv = csv.DictReader(archivo_csv)
        filas = list(lector_csv)

    # Escribir archivo JSON
    with open(json_file, mode='w') as archivo_json:
        json.dump(filas, archivo_json, indent=4)

# Uso de la función
csv_a_json('archivo.csv', 'archivo.json')