# !/bin/bash
# Programa para ejemplificar el uso de los ifs anidados
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

notaClase=0
continua=""

echo "Ejemplo Sentencia If - Else"
read -n1 -p "Indique cual es su nota (1-9): " notaClase
echo -e "\n"
if [ $notaClase -ge 7 ]; then
	echo "El alumno aprueba la materia"
	read -p "Si va continuar estudiando en el siquiente nivel (s/n): " continua
	if [ $continua = "s" ]; then
		echo "Bienvenido al siguiente nivel"
	else 
		echo "Gracias por trabajar con nosotros, mucha suerte!!"
	fi
else
	echo "El alumno reprueba la materia"

fi