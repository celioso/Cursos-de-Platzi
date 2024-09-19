# !/bin/bash
# Programa para revisar el declaracion de variables

option=0
nombre=Marco

echo "Optión: $option y nombre: $nombre"

# Exportar la variable nombre para que esta disponible a los demás procesos
export nombre

# Llamar al siguiente script para recuperar la variable
./2_variables_2.sh
