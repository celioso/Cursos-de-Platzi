# ! /bin/bash
# Programa para ejemplificar como se escribe en un archivo
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

# para realizar la prueba `./21_archivosDirectorios.sh d prueba` d crea carpetas y f archivos

echo "Escribir en un archivo"

echo "Valores escritos con el comando echo" >> $1

# Adición multilíneas
cat <<EOM >>$1
$2
EOM

# para utilizar el programas se usa: `./22_writeFile.sh prueba.txt "Valores con el comando cat"` o `<nombre_del_programa.sh> <archivo a escribir.txt> "<Texto>"`