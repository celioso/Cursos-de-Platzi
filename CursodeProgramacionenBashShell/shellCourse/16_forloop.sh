# ! /bin/bash
# Programa para ejemplificar el uso de los sentencia d eiteración for
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

arregloNumeros=(1 2 3 4 5 6)

echo "Interar la Lista de Números"
for num in ${arregloNumeros[*]}
do 
        echo "Número: $num"
done
echo -e "\n"
echo "Iterar en la lista de cadenas"
for nom in "Marco" "Pedro" "Luis" "Daniel"
do
    echo "Nombre: $nom"
done
echo -e "\n"
echo "Iterar en archivos"
for fil in *
do
    echo "Nombre arrchivo: $fil"
done
echo -e "\n"
echo "Iterar utilizando un comando"
for fil in $(ls)
do
    echo "Nombre archivo: $fil"
done
echo -e "\n"
echo "Iterar utilizando el formato tradicional"
for ((i=1; i<10;i++))
do
    echo "Número: $i"
done