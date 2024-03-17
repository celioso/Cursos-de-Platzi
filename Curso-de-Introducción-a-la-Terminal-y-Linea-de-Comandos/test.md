### Curso de Introducción a la Terminal y Línea de Comandos

1. **La shell o línea de comandos es:**

**R/:** Un programa que nos ayuda a comunicarnos con nuestro sistema operativo.

2. **Para crear un archivo usamos el comando:**

**R/:** touch mi_archivo

3. **¿Con cuál comando copiamos un directorio y su contenido? (Esto hace parte de uno de los retos que te dejé)**

**R/:** cp -r mi_directorio ruta_destino

4. **Para leer el manual de usuario de un comando usamos:**

**R/:** man

5. **Las wildcards son caracteres que nos permiten definir patrones avanzados de búsqueda en la línea de comandos, esto es:**

**R/:** Verdadero

6. **Si queremos listar todos los archivos que sean extensión txt podemos usar el comando:**

**R/:** ls *.txt

7. **El file descriptor correspondiente al stderr es:**

**R/:** 2

8. **¿Qué operador nos ayuda a concatenar la salida de un comando a un archivo de texto?**

**R/:** >>

9. **El pipe operator redirecciona la salida de un comando a la entrada de otro comando, esto es:**

**R/:** Verdadero

10. **Si queremos explorar las primeras 100 líneas de un documento de texto lo podemos hacer con:**

**R/:** head -n 100 mi_texto | less

11. **Si deseamos condicionar la ejecución de un comando solo si uno anterior se ejecuto de manera exitosa podemos usar:**

**R/:** &&

12. El comando chmod u=rwx,go=r mi_archivo ¿qué permisos otorga?

**R/:** Otorga permisos de lectura, escritura y ejecución al usuario. Solo otorga permiso de lectura a los grupos y a otros.

13. **Es una mala práctica de seguridad asignar la siguiente configuración de permisos en modo octal a cualquier archivo o directorio.**

**R/:** 777

14. Con el siguiente comando podemos ver la ruta del directorio Home de nuestro usuario:

**R/:** echo $HOME

15. **Para guardar todas nuestras variables de entorno en un archivo de texto podemos ejecutar el comando:**

**R/:** env > environment.txt

16. **Es un comando que nos ayuda a buscar la ruta de binarios o ejecutables en nuestro sistema.**

**R/:** which

17. **Para buscar todas las imágenes png dentro de nuestra computadora podemos ejecutar:**

**R/:** find / -name *.png

18. **Para usar grep sin distinción de mayúsculas o minúsculas usamos:**

**R/:** -i

19. **¿Qué comando nos ayuda consultar la disponibilidad de un equipo en una red?**

**R/:** ping

20. **¿Qué comando muestra los procesos que consumen más recursos en nuestro sistema?**

**R/:** top
