import statistics
import csv

# Leer los datos de ventas mensuales desde un archivo CSV
monthly_sales = {}
with open('monthly_sales.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        month = row['month']
        sales = int(row['sales'])
        monthly_sales[month] = sales

sales = list(monthly_sales.values())
#print(sales)

# Hallar la Media
mean_sales = statistics.mean(sales)
print(f"La media es: {mean_sales}")

# Hallar la Mediana
median_sales = statistics.median(sales)
print(f"La mediana es: {median_sales}")

# Hallar la Moda
mode_sales = statistics.mode(sales)
print(f"La media es: {mode_sales}")

# Hallar la Desviación Estándar
stdev_sales = statistics.stdev(sales)
print(f"La desviación estándar es: {stdev_sales}")

# Hallar la varianza
variance_sales = statistics.variance(sales)
print(f"La desviación estándar es: {variance_sales}")

# Extraer el máximo de ventas y el mínimo
max_sales = max(sales)
min_sales = min(sales)

range_sales = max_sales - min_sales
print(f'El máximo de ventas {max_sales} y mínimo {min_sales}, El rango de ventas es: {range_sales}')