# ! /bin/bash
# Programa para ejemplificar el uso de los sentencia de iteración while
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

numero=1

while [ $numero -ne 10 ]
do
	echo "Imprimiendo $numero veces"
	numero=$(( numero + 1 ))
done

