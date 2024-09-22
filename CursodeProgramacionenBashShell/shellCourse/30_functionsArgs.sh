# ! /bin/bash
# Programa que permite manejar las utilidades de 
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

opcion=0

# Funcion Instalar postgres
instalar_postgres () {
	echo "\nInstalar Postgres.."
}

# Función para desinstalar postgres
desinstalar_postgres () {
	echo "\nDesinstalar postgres..."
}

# Función para sacar el respaldo
sacar_respaldo () {	
	echo "\nSacar respaldo..."
	echo "\nDirectorio backup: $1"
}

# Fnción para restaurar respaldo
restaurar_respaldo () {
	echo "\nRestaurar respaldo..."
	echo "\nDirectorio respaldo: $1"
}

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
			instalar_postgres
			sleep 5
			;;
		2)
			desinstalar_postgres
			sleep 6
			;;
		3)
			read -p "Directorio Backup: " directorioBackup
			sacar_respaldo $directorioBackup
			sleep 3
			;;
		4)	
			read -p "Directorio de Respaldos: " directorioRespaldos
			restaurar_respaldo $directorioRespaldos
			sleep 2
			;;
		5)
			echo -e "\nSalir del Programa"
			exit 0
			;;
	esac
done
