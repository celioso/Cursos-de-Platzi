11_ifElse.sh                                                                                        0000777 0001750 0001750 00000001053 14672660255 012125  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # !/bin/bash
# Programa para ejemplificar el uso de la sentencia de decisión if, else
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

notaClase=0
edad=0

echo "Ejemplo Sentencia If - Else"
read -n1 -p "Indique cual es su nota (1-9): " notaClase
echo -e "\n"
if (( $notaClase >= 7 )); then
	echo "El alumno aprueba la materia"
else
	echo "El alumno reprueba la materia"
fi

read -p "Indique cual es su edad: " edad

if [ $edad -le 18 ]; then
	echo "La persona no puede sufragar"
else
	echo "La persona puede sufragar"
fi
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     11_ifElseIfElse.sh                                                                                  0000777 0001750 0001750 00000000673 14672665072 013226  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # !/bin/bash
# Programa para ejemplificar el uso de la sentencia de decisión if, else if, else
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

edad=0

echo "Ejemplo Sentencia If _else"
read -p "Indique cual es su edad: " edad
if [ $edad -le 18 ]; then
	echo "La persona es adolescente"
elif [ $edad -ge 19 ] && [ $edad -le 64 ]; then  
	echo "La persona es adulta"
else
	echo "La persona es adulta mayor"
fi
                                                                     12_ifAniddos.sh                                                                                     0000777 0001750 0001750 00000001141 14672667651 012624  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # !/bin/bash
# Programa para ejemplificar el uso de los ifs anidados
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

notaClase=0
continua=""

echo "Ejemplo Sentencia If - Else"
read -n1 -p "Indique cual es su nota (1-9): " notaClase
echo -e "\n"
if [ $notaClase -ge 7 ]; then
	echo "El alumno aprueba la materia"
	read -p "Si va continuar estudiando en el siquiente nivel (s/n): " continua
	if [ $continua = "s" ]; then
		echo "Bienvenido al siguiente nivel"
	else 
		echo "Gracias por trabajar con nosotros, mucha suerte!!"
	fi
else
	echo "El alumno reprueba la materia"

fi                                                                                                                                                                                                                                                                                                                                                                                                                               13_expresionesCondicionales.sh                                                                      0000777 0001750 0001750 00000002003 14672675314 015755  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # !/bin/bash
# Programa para ejemplificar el uso de expresiones condicionales
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

edad=0
pais=""
pathArchivo=""

read -p "Ingrese su edad:" edad
read -p "Ingrese su país:" pais
read -p "Ingrese el path de su archivo:" pathArchivo

echo -e "\nExpresiones Condicionales con números"
if [ $edad -lt 10 ]; then
    echo "La persona es un niño o niña"
elif [ $edad -ge 10 ] && [ $edad -le 17 ]; then
    echo "La persona se trata de un adolescente"
else
    echo "La persona es mayor de edad"
fi

echo -e "\nExpresiones Condicionales con cadenas"
if [ $pais = "EEUU" ]; then
    echo "La persona es Americana"
elif [ $pais = "Ecuador" ] || [ $pais = "Colombia" ]; then
    echo "La persona es del Sur de América"
else
    echo "Se desconoce la nacionalidad"
fi



echo -e "\nExpresiones Condicionales con archivos"
if [ -d $pathArchivo ]; then
    echo "El directorio $pathArchivo existe"
else 
    echo "El directorio $pathArchivo no existe"
fi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 14_case.sh                                                                                          0000777 0001750 0001750 00000000726 14672676566 011656  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # !/bin/bash
# Programa para ejemplificar el uso de la sentencia case
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

option=""

echo "Ejercicio Sentencia Case"
read -n1 -p "Ingrese una opción de la A - Z: " option
echo -e "\n"

case $option in
    "A") echo -e "\nOperación guardar archivo";;
    "B") echo "Operación Eliminar Archivo";;
    [C-E]) echo "No esta implementada la operación";;
    "*") echo "Opción incorrecta"
esac                                          15_arreglos.sh                                                                                      0000777 0001750 0001750 00000002057 14673105317 012537  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa para ejemplificar el uso de los arreglos
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

arregloNumeros=(1 2 3 4 5 6)
arregloCadenas=(Marco, Antonio, Pedro, Susana)
arreglosRangos=({A..Z} {a..z} {10..20})

# Imprimir todos los valores
echo "Arreglo de Números: ${#arregloNumeros[*]}"
echo "Arreglo de Cadenas: ${#arregloCadenas[*]}"
echo "Arreglo de Números: ${#arreglosRangos[*]}"

#Imprimir los tamaños de los arreglos
echo "Tamaño de Números: ${arregloNumeros[*]}"
echo "Tamaño de Cadenas: ${arregloCadenas[*]}"
echo "Tamaño de Números: ${arreglosRangos[*]}"

#Imprimir la posición 3 del arreglo de números, cadenas de rango
echo "Posición 3 Arreglo de Números: ${arregloNumeros[3]}"
echo "Posición 3 Arreglo de Cadenas: ${arregloCadenas[3]}"
echo "Posición 3 Arreglo de Rangos: ${arreglosRangos[3]}"

#Añadir y eliminar valores en un arreglo
arregloNumeros[7]=20
unset arregloNumeros[0]
echo "Arreglo de Números: ${arregloNumeros[*]}"
echo "Tamaño arreglo de Números: ${#arregloNumeros[*]}"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 16_forloop.sh                                                                                       0000777 0001750 0001750 00000001344 14673113704 012377  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa para ejemplificar el uso de los sentencia de iteración for
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
done                                                                                                                                                                                                                                                                                            17_whileLoop.sh                                                                                     0000777 0001750 0001750 00000000421 14673114132 012651  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa para ejemplificar el uso de los sentencia de iteración while
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

numero=1

while [ $numero -ne 10 ]
do
	echo "Imprimiendo $numero veces"
	numero=$(( numero + 1 ))
done

                                                                                                                                                                                                                                               18_loopsAnidados.sh                                                                                 0000777 0001750 0001750 00000000422 14673133134 013513  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa para ejemplificar el uso de los loops anidados
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

echo "Loops Anidados"
for fil in $(ls)
do 
	for nombre in {1..4}
	do
		echo "Nombre archivo: $fil _ $nombre"
	done
done
                                                                                                                                                                                                                                              19_breakContinue.sh                                                                                 0000777 0001750 0001750 00000000607 14673135050 013512  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa para ejemplificar el uso de break y Continue
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

echo "Sentencias break y continue"
for fil in $(ls)
do 
	for nombre in {1..4}
	do	
		if [ $fil = "10_download.sh" ]; then
			break;
		elif [[ $fil == 4* ]]; then
			continue;
		else
			echo "Nombre archivo: $fil _ $nombre"
		fi
	done
done
                                                                                                                         1_utilityPostgres.sh                                                                                0000777 0001750 0001750 00000000211 14667700531 014056  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # !/bin/bash
# Programa Para realizar algunas operaciones utilitarias de Postgres

echo "Hola bienvenido al curso de Programación bash"
                                                                                                                                                                                                                                                                                                                                                                                       20_menuOpciones.sh                                                                                  0000777 0001750 0001750 00000002075 14673144350 013361  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa que permite manejar las utilidades de Posgres
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

opcion=0

while : 
do
	#Limpiar la pantalla
	clear
	#Desplegar el menú de opciones
	echo "_________________________________________"
	echo "PGUTIL - Programa de Utilidad de Postgres"
	echo "_________________________________________"
	echo "             MENÚ PRINCIPAL              "
	echo "_________________________________________"
	echo "1. Instalar Postgres"
	echo "2. Desinstalar Postgres"
	echo "3. Sacar un respaldo"
	echo "4. Restar respaldo"
	echo "5. Salir"

	# Leer los datos del usuario - Capturar información
	read -n1 -p "Ingrese una función [1-5]: " opcion
	
	#Validar la opción ingresada
	case $opcion in
		1)
			echo -e "\nInstalar postgres......."
			sleep 5
			;;
		2)
			echo -e "\nDesinstalar postgres......."
			sleep 6
			;;
		3)
			echo -e "\nSacar respaldo..."
			sleep 3
			;;
		4)
			echo -e "\nRestaurar respaldo..."
			sleep 2
			;;
		5)
			echo -e "\nSalir del Programa"
			exit 0
			;;
	esac
done
                                                                                                                                                                                                                                                                                                                                                                                                                                                                   21_archivosDirectorios.sh                                                                           0000777 0001750 0001750 00000001016 14673402245 014735  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
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

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  22_writeFile.sh                                                                                     0000777 0001750 0001750 00000001033 14673404245 012644  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa para ejemplificar como se escribe en un archivo
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

# para realizar la prueba `./21_archivosDirectorios.sh d prueba` d crea carpetas y f archivos

echo "Escribir en un archivo"

echo "Valores escritos con el comando echo" >> $1

# Adición multilíneas
cat <<EOM >>$1
$2
EOM

# para utilizar el programas se usa: `./22_writeFile.sh prueba.txt "Valores con el comando cat"` o `<nombre_del_programa.sh> <archivo a escribir.txt> "<Texto>"`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     23_leerArchivos.sh                                                                                  0000777 0001750 0001750 00000001217 14673406430 013343  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
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

# Comando de leer el archivo: `/23_leerArchivos.sh prueba.txt`                                                                                                                                                                                                                                                                                                                                                                                 24_operacionesArchivos.sh                                                                           0000777 0001750 0001750 00000001114 14673410735 014723  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
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

                                                                                                                                                                                                                                                                                                                                                                                                                                                    25_tar.sh                                                                                           0000777 0001750 0001750 00000000401 14673417322 011501  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa para ejemplificar el empaquetamiento con el comando tar
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

echo "Empaaquetar todos los scripts de la carpeta shellCourse"
tar -cvf shellCourse.tar *.sh

                                                                                                                                                                                                                                                               26_gzip.sh                                                                                          0000777 0001750 0001750 00000000652 14673420225 011671  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa para ejemplificar el empaquetamiento con el comando tar y gzip
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

echo "Empaquetar todos los scripts de la carpeta shellCourse"
tar -cvf shellCourse.tar *.sh

# Cuando se empaqueta con gzip el empaquetamiento anterior se elimina
gzip shellCourse.tar

echo "Empaquetar un solo archivo. con un ratio 9"
gzip -9 9_options.sh
                                                                                      27_pbzip2.sh                                                                                        0000777 0001750 0001750 00000000566 14673420600 012130  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa para ejemplificar el empaquetamiento con el comando pbzip
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

echo "Empaquetar todos los scripts de la carpeta shellCourse"
tar -cvf shellCourse.tar *.sh
pbzip2 -f shellCourse.tar

echo "Empaquetar un directorio con tar y pbzip2"
tar -cf *.sh -c > shellCoursetwo.tar.bz2
                                                                                                                                          28_zipPassword.sh                                                                                   0000777 0001750 0001750 00000000566 14673633437 013266  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa para ejemplificar el empaquetamiento con el comando pbzip
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

echo "Empaquetar todos los scripts de la carpeta shellCourse"
tar -cvf shellCourse.tar *.sh
pbzip2 -f shellCourse.tar

echo "Empaquetar un directorio con tar y pbzip2"
tar -cf *.sh -c > shellCoursetwo.tar.bz2
                                                                                                                                          2_variables.sh                                                                                      0000777 0001750 0001750 00000000453 14667711524 012611  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # !/bin/bash
# Programa para revisar el declaracion de variables

option=0
nombre=Marco

echo "Optión: $option y nombre: $nombre"

# Exportar la variable nombre para que esta disponible a los demás procesos
export nombre

# Llamar al siguiente script para recuperar la variable
./2_variables_2.sh
                                                                                                                                                                                                                     2_variables_2.sh                                                                                    0000777 0001750 0001750 00000000174 14667712342 013031  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # !/bin/bash
# Programa para revisar el declaracion de variables


echo "Optión nombre pasado del scrip anterior: $nombre"
                                                                                                                                                                                                                                                                                                                                                                                                    3_tipoOperadores.sh                                                                                 0000777 0001750 0001750 00000001711 14670146111 013623  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
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
echo "Residuo A %= B" $((numA %= numB))                                                       4_argumentos.sh                                                                                     0000777 0001750 0001750 00000000414 14670147607 013023  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa para ejemplificar el paos de argumentos

nombreCurso=$1
horarioCurso=$2

echo "El nombre del curso es :$nombreCurso dictado en el horario de $horarioCurso"
echo "El número de parametros enviados es: $#"
echo "Los parametros enviados son: $*"
                                                                                                                                                                                                                                                    5_subtitucionComand.sh                                                                              0000777 0001750 0001750 00000000607 14670154217 014331  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# programa para revisar como ejecuatr comandos dentro de un programa y almacenar en una variable para su posterior utilización de operadores
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

ubicacionActual=`pwd`
infoKernel=$(uname -a)

echo "La ubicación actual es la siguiente: $ubicacionActual"
echo "Información del Kernel: $infoKernel"

                                                                                                                         6_readEcho.sh                                                                                       0000777 0001750 0001750 00000000676 14670165017 012360  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# programa para ejemplificar como capturar la información del usuario utilizando el comando echo, read y $REPLY
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

option=0
backupName=""

echo "Programa utilidades Postgres"
echo -n "Ingresar una opción: "
read
option=$REPLY
echo -n "Ingresar el nombre del archivo del Backup: "
read
backupName=$REPLY
echo "Optión:$option , bachupName:$backupName"
                                                                  7_read.sh                                                                                           0000777 0001750 0001750 00000000627 14670165345 011562  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# programa para ejemplificar como capturar la información del usuario utilizando el comando read
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

option=0
backupName=""

echo "Programa utilidades Postgres"
read -p "Ingresar una opción: " option
read -p "Ingresar el nombre del archivo del Backup: " backupName
echo "Optión:$option , bachupName:$backupName"
                                                                                                         8_readValidate.sh                                                                                   0000777 0001750 0001750 00000001047 14671672135 013233  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# programa para ejemplificar como capturar la información del usuario utilizando el comando read
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

option=0
backupName=""
Clave=""

echo "Programa utilidades Postgres"
# acepta el ingreso de información de solo un caracter
read -n1 -p "Ingresar una opción:" option
echo -e "\n"
read -n10 -p "Ingresar el nombre del archivo del Backup:" backupName
echo -e "\n"
echo "Optión:$option , bachupName:$backupName"
read -s -p "Clave:" Clave
echo "Clave: $Clave"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         8_regularExpression.sh                                                                              0000777 0001750 0001750 00000002054 14671673234 014367  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa para ejemplificar como capturar la información del usuario y validarla utilizando expresiones regulares
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

identificacionRegex='^[0-9]{10}$'
paisRegex='^EC|COL|US$'
fechaNacimientoRegex='^19|20[0-8]{2}[1-12][1-31]$'

echo "Expresiones regulares"
read -p "Ingresar una identificacion:" identificacion
read -p "Ingresar las iniciales de un país [EC, COL, US]:" pais
read -p "Ingresar la fecha de nacimiento [yyyyMMdd]:" fechaNacimiento 

#Validación Identificación
if [[ $identificacion =~ $identificacionRegex ]]; then
    echo "Identificación $identificacion válida"
else
    echo "Identificación $identificacion inválida"
fi

#Validación País
if [[ $pais =~ $paisRegex ]]; then
    echo "País $pais válido"
else
    echo "País $pais inválido"
fi

#Validación Fecha Nacimiento
if [[ $fechaNacimiento =~ $fechaNacimientoRegex ]]; then
    echo "Fecha Nacimiento $fechaNacimiento válida"
else
    echo "Fecha Nacimiento $fechaNacimiento inválida"
fi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    9_options.sh                                                                                        0000777 0001750 0001750 00000001013 14673421722 012327  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# programa para ejemplificar como capturar la información del usuario utilizando el comando read
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

echo "Programa Opciones"
echo "Opción 1 enviada: $1"
echo "Opción 2 enviada: $2"
echo "Opción enviadas: $*"
echo -e "\n"
echo "Recuperar valores"
while [ -n "$1" ]
do
case "$1" in
-a) echo "-a option utilizada";;
-b) echo "-b option utilizada";;
-c) echo "-c option utlizada";;
*) echo "$! no es una opción";;
esac
shift
done                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     Reto_5.sh                                                                                           0000777 0001750 0001750 00000000374 14673415657 011564  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Reto 5
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

option=1
result=2

# Uso correcto del comando `date`
echo "La opción es: $option y el resultado es: $result" >> log-`date +%Y-%m-%d,%H-%M-%S`.log
                                                                                                                                                                                                                                                                    reto_2.sh                                                                                           0000777 0001750 0001750 00000002757 14672655612 011624  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # !/bin/bash
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
echo "Hola, $name $lastname, tu edad es $age, la dirección es $address y el número telefónico es $number_phone."                 reto_3.sh                                                                                           0000777 0001750 0001750 00000000644 14672665074 011621  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # !/bin/bash
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

                                                                                            reto_4.sh                                                                                           0000777 0001750 0001750 00000002641 14673150035 011604  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Programa que permite manejar las utilidades de Posgres
# Autor: Mario Celis - https://www.linkedin.com/in/mario-alexander-vargas-celis/

opcion=0

while : 
do
	#Limpiar la pantalla
	clear
	#Desplegar el menú de opciones
	echo "_________________________________________"
	echo "DaksaToolS - Programa de Mantenimiento Linux"
	echo "_________________________________________"
	echo "             MENÚ PRINCIPAL              "
	echo "_________________________________________"
	echo "1. Procesos Actuales"
	echo "2. Memoria Disponible"
	echo "3. Espacio en Disco"
	echo "4. Información de Red"
	echo "5. Variables de Entorno Configuradas"
	echo "6. Información Programa"
	echo "7. Backup información"
	echo "8. Salir"
	# Leer los datos del usuario - Capturar información
	read -n1 -p "Ingrese una función [1-8]: " opcion
	
	#Validar la opción ingresada
	case $opcion in
		1)
			echo -e "\nProcesos Actuales"
			ps aux
			sleep 4
			;;
		2)
			echo -e "\nMemoria Disponible"
			free --giga
			sleep 4
			;;
		3)
			echo -e "\nEspacio en Disco"
			df -h
			sleep 4
			;;
		4)
			echo -e "\nInformación de Red"
			ip addr show
			sleep 4
			;;
		5)
			echo -e "\nVariables de Entorno"
			printenv
			sleep 4
			;;
		6)
			echo -e "\nInformación Programa"
			dpkg --list
			sleep 4
			;;
		7)
			echo -e "\nBackup información"
			deja-dup
			sleep 4
			;;
		8)
			echo -e "\nSalir del Programa"
			exit 0
			;;
	esac
done
                                                                                               utilityHost.sh                                                                                      0000777 0001750 0001750 00000000220 14670162226 012742  0                                                                                                    ustar   celis                           celis                                                                                                                                                                                                                  # ! /bin/bash
# Reto 1
#Mario Celis

option="Entrada :("
result="salida :)"

echo "Llama el valor Option: $option y da como resultado: $result"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                