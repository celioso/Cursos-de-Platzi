# !/bin/bash
# Programa para ejemplificar el uso de la sentencia de decisión if, else if, else
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

edad=0

echo "Ejemplo Sentencia If _else"
read -p "Indique cual es su edad: " edad
if [ $edad -le 18 ]; then
	echo "La persona es adolescente"
elif [ $edad -ge 19 ] && [ $edad -le 64 ]; then  
	echo "La persona es adulta"
else
	echo "La persona es adulta mayor"
fi
