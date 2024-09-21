# ! /bin/bash
# Programa para ejemplificar la creación de archivos y directorio
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

# para realizar la prueba `./21_archivosDirectorios.sh d prueba` d crea carpetas y f archivos

echo "Archivos - Directorios"

if [ $1 = "d" ]; then
    mkdir -m 755 $2
    echo "Directorio creado correctamente"
    ls -la $2
elif [ $1 == "f" ]; then
    touch $2
    echo "Archivo creado correctamente"
    ls -la $2
else
    echo "No existe esa opción: $1"
fi

