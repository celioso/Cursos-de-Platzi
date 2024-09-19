# !/bin/bash
# Programa para ejemplificar el uso de la sentencia case
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

option=""

echo "Ejercicio Sentencia Case"
read -n1 -p "Ingrese una opción de la A - Z: " option
echo -e "\n"

case $option in
    "A") echo -e "\nOperación guardar archivo";;
    "B") echo "Operación Eliminar Archivo";;
    [C-E]) echo "No esta implementada la operación";;
    "*") echo "Opción incorrecta"
esac