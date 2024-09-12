import csv

file_path = 'products.csv'
updated_file_path = 'products_updated.csv'

with open(file_path, mode='r', newline='') as file:
    csv_reader = csv.DictReader(file)
    #optener el nombre de las columnas existentes
    fieldnames=csv_reader.fieldnames + ['code_product']

    with open(updated_file_path, mode='w', newline='') as updated_file: 
        csv_writer=csv.DictWriter(updated_file, fieldnames=fieldnames)
        csv_writer.writeheader() # Escribir los encabezados

        for row in csv_reader:
            row['code_product'] = float(row['price']) * int(row['quantity'])
            csv_writer.writerow(row)