# ! /bin/bash
# Programa que permite manejar las utilidades de Posgres
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

opcion=0

while : 
do
	#Limpiar la pantalla
	clear
	#Desplegar el menú de opciones
	echo "_________________________________________"
	echo "PGUTIL - Programa de Utilidad de Postgres"
	echo "_________________________________________"
	echo "             MENÚ PRINCIPAL              "
	echo "_________________________________________"
	echo "1. Instalar Postgres"
	echo "2. Desinstalar Postgres"
	echo "3. Sacar un respaldo"
	echo "4. Restar respaldo"
	echo "5. Salir"

	# Leer los datos del usuario - Capturar información
	read -n1 -p "Ingrese una función [1-5]: " opcion
	
	#Validar la opción ingresada
	case $opcion in
		1)
			echo -e "\nInstalar postgres......."
			sleep 5
			;;
		2)
			echo -e "\nDesinstalar postgres......."
			sleep 6
			;;
		3)
			echo -e "\nSacar respaldo..."
			sleep 3
			;;
		4)
			echo -e "\nRestaurar respaldo..."
			sleep 2
			;;
		5)
			echo -e "\nSalir del Programa"
			exit 0
			;;
	esac
done
