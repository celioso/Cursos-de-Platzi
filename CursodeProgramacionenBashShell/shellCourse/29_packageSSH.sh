# ! /bin/bash
# Programa para ejemplificar ls forma de como transferir por la red utilizando el comando rsync, utilizando las opciones de empaquetamiento para optimizar la velocidad de transferencia
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

echo "Empaquetar todos los scripts de la carpeta shellCourse y transferirlos por la red a otra equipo utilizando en comando rsync"

host=""
usuario=""

read -p "Ingresar el host: " host
read -p "Ingresar el usuario: " usuario
echo -e "\nEn este momento se procedera a empaquetar la carpeta y transferiria según los datos ingresado"
rsync -avz $(pwd) $usuario@$host:/User/martosfre/Downloads/platzi
