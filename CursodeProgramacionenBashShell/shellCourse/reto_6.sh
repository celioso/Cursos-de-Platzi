# ! /bin/bash
# Programa para ejemplificar ls forma de como transferir por la red utilizando el comando rsync, utilizando las opciones de empaquetamiento para optimizar la velocidad de transferencia
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

Modificar programa utilityHost. sh para empaquetar los logs generados utilizando algún formato de compresión,
colocarle una clave y pasarlo a otra máquina a través de SSH, cuando se seleccione la opción 7. Backup de Información
`




# Función para validar entrada con expresión regular
function validar_entrada {
    local entrada
    read -p "$1" entrada
    while [[ ! $entrada =~ $2 ]]; do
        echo "Entrada no válida. $3"
        read -p "$1" entrada
    done
    echo "$entrada"
}

# Expresiones regulares
regex_ano="^[0-9]{4}$"
regex_mes="^[0-9]{2}$"
regex_dia="^(0[1-9]|[1-2][0-9]|3[0-1])$"
regex_hora="^(0[0-9]|1[0-9]|2[0-3])$"
regex_minuto="^[0-5][0-9]$"

# Solicitar y validar la información del usuario
ano=$(validar_entrada "Ingresa el año (YYYY): " "$regex_ano" "Debe tener el formato YYYY (cuatro dígitos).")
mes=$(validar_entrada "Ingresa el mes (MM): " "$regex_mes" "Debe tener el formato MM (dos dígitos).")
dia=$(validar_entrada "Ingresa el día (DD): " "$regex_dia" "Debe estar entre 01 y 31.")
hora=$(validar_entrada "Ingresa la hora (HH): " "$regex_hora" "Debe estar entre 00 y 23.")
minutos=$(validar_entrada "Ingresa los minutos (MM): " "$regex_minuto" "Debe estar entre 00 y 59.")

# Crear el nombre del archivo de log basado en la fecha y hora ingresadas
nombre_log="${ano}_${mes}_${dia}_${hora}_${minutos}_fecha.log"

# Escribir la información en el archivo de log
echo "${ano}:${mes}:${dia} ${hora}:${minutos}" >> "$nombre_log"
gzip -9 $(./{$nombre_log})
archivo_comprimido="$nombre_log.zip"
echo "La información se ha registrado y comprimido en el archivo $archivo_comprimido."

host=""
usuario=""

read -p "Ingresar el host:" host
read -p "Ingresar el usuario:" usuario
echo -e "\nEn este momento se procederá a empaquetar la carpeta y transferirla según los datos ingresados"
rsync -avz $(pwd) $usuario@$host:/Users/martosfre/Downloads/platzi