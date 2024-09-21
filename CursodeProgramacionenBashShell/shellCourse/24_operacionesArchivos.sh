# ! /bin/bash
# Programa para ejemplificar las operaciones de un archivo
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

# para realizar la prueba `./21_archivosDirectorios.sh d prueba` d crea carpetas y f archivos

echo "Operaciones de un archivo"
mkdir -m 755 backupScripts

echo -e "\nCopiar los scripts del directorio actual al nuevo directorio backupScripts"
cp *.* backupScripts/
ls -la backupScripts/

echo -e "\nMover el directorio backupScripts a otra ubicación: $HOME"
mv backupScripts $HOME


echo -e "\nEliminar los archivos .txt"
rm *.txt

