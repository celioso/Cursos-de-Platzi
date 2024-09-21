# ! /bin/bash
# Reto 5
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

option=1
result=2

# Uso correcto del comando `date`
echo "La opción es: $option y el resultado es: $result" >> log-`date +%Y-%m-%d,%H-%M-%S`.log
