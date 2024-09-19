# ! /bin/bash
# Programa para ejemplificar el uso de los arreglos
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

arregloNumeros=(1 2 3 4 5 6)
arregloCadenas=(Marco, Antonio, Pedro, Susana)
arreglosRangos=({A..Z} {a..z} {10..20})

# Imprimir todos los valores
echo "Arreglo de Números: ${#arregloNumeros[*]}"
echo "Arreglo de Cadenas: ${#arregloCadenas[*]}"
echo "Arreglo de Números: ${#arreglosRangos[*]}"

#Imprimir los tamaños de los arreglos
echo "Tamaño de Números: ${arregloNumeros[*]}"
echo "Tamaño de Cadenas: ${arregloCadenas[*]}"
echo "Tamaño de Números: ${arreglosRangos[*]}"

#Imprimir la posición 3 del arreglo de números, cadenas de rango
echo "Posición 3 Arreglo de Números: ${arregloNumeros[3]}"
echo "Posición 3 Arreglo de Cadenas: ${arregloCadenas[3]}"
echo "Posición 3 Arreglo de Rangos: ${arreglosRangos[3]}"

#Añadir y eliminar valores en un arreglo
arregloNumeros[7]=20
unset arregloNumeros[0]
echo "Arreglo de Números: ${arregloNumeros[*]}"
echo "Tamaño arreglo de Números: ${#arregloNumeros[*]}"