# ! /bin/bash
# programa para ejemplificar como capturar la información del usuario utilizando el comando read
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

option=0
backupName=""
Clave=""

echo "Programa utilidades Postgres"
# acepta el ingreso de información de solo un caracter
read -n1 -p "Ingresar una opción:" option
echo -e "\n"
read -n10 -p "Ingresar el nombre del archivo del Backup:" backupName
echo -e "\n"
echo "Optión:$option , bachupName:$backupName"
read -s -p "Clave:" Clave
echo "Clave: $Clave"