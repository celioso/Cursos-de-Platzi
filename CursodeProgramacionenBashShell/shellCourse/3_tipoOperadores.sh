# ! /bin/bash
# programa para revisar tipos de operadores
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

numA=10
numB=4

echo "Operadores Aritméticos"
echo "Números: A= $numA y B=$numB"
echo "Sumar A + B =" $((numA + numB))
echo "Resta A - B =" $((numA - numB))
echo "Multiplicar A * B =" $((numA * numB))
echo "Dividir A / B =" $((numA / numB))
echo "Residuo A % B =" $((numA % numB))


echo -e "\nOperadores Relaciones"
echo "Número:  A=$numA y B=$numB"
echo "A > B =" $((numA > numB))
echo "A < B =" $((numA < numB))
echo "A >= B =" $((numA >= numB))
echo "A <= B =" $((numA <= numB))
echo "A == B =" $((numA < numB))
echo "A != B =" $((numA < numB))


echo -e "\nOperadores Asignados"
echo "Número:  A=$numA y B=$numB"
echo "Suma: A += B" $((numA += numB))
echo "Resta: A -= B" $((numA -= numB))
echo "Multiplicación: A *= B " $((numA *= numB))
echo "Dividir: A /= B" $((numA /= numB))
echo "Residuo A %= B" $((numA %= numB))