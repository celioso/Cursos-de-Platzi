# ! /bin/bash
# Programa para ejemplificar el empaquetamiento con clave utilizando zip
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

echo "Empaquetar todos los scripts de la carpeta shellCourse con zi y asignarle una clave de seguridad"
zip -e shellCourse.zip *.sh
