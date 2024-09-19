# ! /bin/bash
# programa para ejemplificar como capturar la información del usuario utilizando el comando read
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

option=0
backupName=""

echo "Programa utilidades Postgres"
read -p "Ingresar una opción: " option
read -p "Ingresar el nombre del archivo del Backup: " backupName
echo "Optión:$option , bachupName:$backupName"
