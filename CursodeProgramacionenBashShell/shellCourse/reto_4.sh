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
	echo "DaksaToolS - Programa de Mantenimiento Linux"
	echo "_________________________________________"
	echo "             MENÚ PRINCIPAL              "
	echo "_________________________________________"
	echo "1. Procesos Actuales"
	echo "2. Memoria Disponible"
	echo "3. Espacio en Disco"
	echo "4. Información de Red"
	echo "5. Variables de Entorno Configuradas"
	echo "6. Información Programa"
	echo "7. Backup información"
	echo "8. Salir"
	# Leer los datos del usuario - Capturar información
	read -n1 -p "Ingrese una función [1-8]: " opcion
	
	#Validar la opción ingresada
	case $opcion in
		1)
			echo -e "\nProcesos Actuales"
			ps aux
			sleep 4
			;;
		2)
			echo -e "\nMemoria Disponible"
			free --giga
			sleep 4
			;;
		3)
			echo -e "\nEspacio en Disco"
			df -h
			sleep 4
			;;
		4)
			echo -e "\nInformación de Red"
			ip addr show
			sleep 4
			;;
		5)
			echo -e "\nVariables de Entorno"
			printenv
			sleep 4
			;;
		6)
			echo -e "\nInformación Programa"
			dpkg --list
			sleep 4
			;;
		7)
			echo -e "\nBackup información"
			deja-dup
			sleep 4
			;;
		8)
			echo -e "\nSalir del Programa"
			exit 0
			;;
	esac
done
