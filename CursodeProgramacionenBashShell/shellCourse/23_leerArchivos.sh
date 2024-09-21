# ! /bin/bash
# Programa para ejemplificar como se lee en un archivo
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

# para realizar la prueba `./21_archivosDirectorios.sh d prueba` d crea carpetas y f archivos

echo "Leer un archivo"
cat $1
echo -e "\nAlmacenar los valores en una variable"
volorCat=`cat $1`
echo "$valorCat"

# Se utiliza la variable especial IFS (Internal Field Separator) para evitar que los espacios en blanco al inicio al final se recortan
echo -e "\nLeer archivos línea por línea"
while IFS= read linea
do
	echo "$linea"
done < $1 

# Comando de leer el archivo: `/23_leerArchivos.sh prueba.txt`