# !/bin/bash
# Reto 3
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

valor=0

echo "Ejercicio reto 3"
read -n1 -p "Indique un numero del 1 al 5: " valor
echo -e "\n"

if [ $valor -le 0 ]; then
        echo "El numero cero no es valida"
elif [ $valor -ge 1 ] && [ $valor -le 4 ]; then
        echo "El numero esta en los rangos pedidos"
else
        echo "El numero es mayor que 5"
fi

