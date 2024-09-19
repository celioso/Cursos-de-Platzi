# ! /bin/bash
# programa para ejemplificar como capturar la información del usuario utilizando el comando echo, read y $REPLY
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

option=0
backupName=""

echo "Programa utilidades Postgres"
echo -n "Ingresar una opción: "
read
option=$REPLY
echo -n "Ingresar el nombre del archivo del Backup: "
read
backupName=$REPLY
echo "Optión:$option , bachupName:$backupName"
