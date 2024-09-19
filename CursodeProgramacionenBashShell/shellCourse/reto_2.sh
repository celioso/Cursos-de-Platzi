# !/bin/bash
# Reto 2
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

nameRegex='^[A-Z][a-zA-Z]{2,15}$'
lastnameRegex='^[A-Z][a-zA-Z]{2,15}$'
ageRegex='^[0-9]{1,3}$'
addressRegex='^[A-Za-z0-9\s]{5,25}$'
number_phoneRegex='^[0-9\-\.\s]{7,15}$'

# Lectura de los datos
read -p "Ingrese su nombre: " name
echo -e "\n"
read -p "Ingrese su apellido: " lastname
echo -e "\n"
read -p "Ingrese su edad: " age
echo -e "\n"
read -p "Ingrese su dirección: " address
echo -e "\n"
read -p "Ingrese su teléfono: " number_phone
echo -e "\n"

# Validación del nombre
if [[ $name =~ $nameRegex ]]; then
    echo -e "Su nombre es: $name"
else
    echo -e "\nEse nombre no es válido"
fi

# Validación del apellido
if [[ $lastname =~ $lastnameRegex ]]; then
    echo -e "Su apellido es: $lastname"
else
    echo -e "\nEse apellido no es válido"
fi

# Validación de la edad
if [[ $age =~ $ageRegex ]]; then
    echo -e "Su edad es: $age"
else
    echo -e "\nLa edad ingresada no es válida"
fi

# Validación de la dirección
if [[ $address =~ $addressRegex ]]; then
    echo -e "Su dirección es: $address"
else
    echo -e "\nLa dirección ingresada no es válida"
fi

# Validación del teléfono
if [[ $number_phone =~ $number_phoneRegex ]]; then
    echo -e "Su teléfono es: $number_phone"
else
    echo -e "\nEl número de teléfono ingresado no es válido"
fi

# Resumen final
echo "Hola, $name $lastname, tu edad es $age, la dirección es $address y el número telefónico es $number_phone."