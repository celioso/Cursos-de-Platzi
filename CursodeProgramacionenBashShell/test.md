### # Curso de Programación en Bash Shell - test

1. **El kernel es la parte fundamental del Sistema Operativo (núcleo) que permite gestionar y administrar los recursos de hardware como son la memoria, el tiempo de procesamiento, el sistema de archivos, las entradas/salidas y es donde se ejecutan las aplicaciones:**

**R/:** Verdadero

2. **La programación bash tiene como objetivo poder ejecutar múltiples comandos de forma secuencial para automatizar una tarea en específico:**

**R/:** Verdadero

3. **¿Una vez creado un script antes de ejecutarlo se tiene que otorgar un permiso de ejecución, con cuál comando se realiza esta acción?**

**R/:** `chmod +x script.sh`

4. **¿Dentro del alcance de una variable, ésta no puede ocuparse en otro script a menos que sea visible a nivel del sistema utilizando el comando EXPORT?**

**R/:** Verdadero

5. **¿Cuáles son los operadores relacionales que se utilizan en la programación bash?**

**R/:** < == !=

6. **¿En el caso de envíe un número de 20 argumentos a mi programa bash y necesita recuperar el número 14, cuál de las siguientes sentencias utilizaría?**

**R/:** ${14}

7. **¿Si se requiere ejecutar un comando dentro de un script y almacenar su respuesta cuál es la sentencia correcta para realizarlo?**

**R/:** variable = $(comando)

8. **¿Cuál es el comando que se utiliza para realizar un debug de un script y que permita diferenciar los comandos de la salida estándar?**

**R/:** `bash -x script.sh`

9. **¿Cuál es el comando que se utiliza para capturar información en un programa shell?**

**R/:** read

10. **Cuando se captura la información ingresada por el usuario y se utiliza validación del tamaño de campo se puede eliminar la información:**

**R/:** Falso

11. **¿Cuándo se utiliza la validación de información utilizando expresiones regulares y se requiere tener 2 ocurrencias de una expresión que sentencia se utiliza para definir la expresión?**

**R/:** `^[0-9]{2}$`

12. **¿Cuál es la forma correcta de pasar opciones a un programa?**

**R/:** `./programa.sh -opt1 -opt2`

13. **¿Cuál es el comando que se utiliza para descargar un programa desde internet?**

**R/:** wget http://www.utilidades.com/programa.zip

14. **En las sentencias de decisión e iteración es necesario respetar los espacios en las condiciones para evitar errores:**

**R/:** Verdadero

15. **En una expresión condicional para comparar números, ¿qué expresión se utiliza?**

**R/:** `[ $variable -eq 10 ]`

16. **La sentencia case puede evaluar un rango de caracteres**

**R/:** Verdadero

17. **¿Cuál de las siguientes declaraciones es correcta para crear un arreglo?**

**R/:** `arregloTmpNumeros=(1 2 3 4)`

18. **Cuando se utiliza la sentencia de iteración for loop se puede iterar arreglos directamente:**

**R/:** Verdadero

19. **¿Cuál es el formato correcto para declarar un while loop? Considerando que todo está en una línea y que las palabras <condition> y <sentences> serán reemplazadas por una condición lógica y sentencias respectivamente.**

**R/:** while <condition>; do <sentences>; done

20. **¿Para qué se utiliza la sentencia break dentro de un loop?**

**R/:** Parar la iteración y salir del loop

21. **Para crear un menú de opciones en un programa bash, ¿qué sentencia de iteración se utiliza?**

**R/:** while loop

22. **¿Qué comando se utiliza para crear un directorio llamado prueba?**

**R/:** mkdir prueba

23. **¿Con cuál de los siguientes comandos se puede escribir en un archivo llamado prueba.txt sin utilizar un programa externo?**

**R/:** `echo texto >> prueba.txt`

24. **¿Cuál sentencia se utiliza para leer el contenido de un archivo llamado prueba.txt dentro de un programa bash?**

**R/:** `cat prueba.txt`

25. **¿Cuál de los siguientes comandos es correcto para copiar todos los archivos de un directorio a otro estando en otro directorio?**

**R/:** `cp -R directorio1/ directorio2/`

26. **¿Con cuál comando se puede empaquetar solamente un archivo simple y no un conjunto de archivos?**

**R/:** gzip

27. **¿Cuál comando para empaquetar soporta poner un password al archivo empaquetado?**

**R/:** Zip

28. **Se requiere realizar un programa en bash que permite sacar respaldos de información y que los transfiera de forma empaquetada por la red a otra computadora en la cual se conoce la IP y el lugar a donde se debe transferir el respaldo. ¿Cuál es la forma más óptima de pasar la información?**

**R/:** Utilizar un comando de transferencia de información con características de empaquetamiento de forma remota desde al origen al destino.

29. **¿Cuál es el formato para declarar una función llamada validarNumeros?**

**R/:** validarNumeros (){ ….}

30. **Para llamar a una función dentro de un programa bash debe crearse antes de realizar la llamada:**

**R/:** Verdadero