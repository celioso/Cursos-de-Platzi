### Curso de Docker: Fundamentos

1. **¿Cuál de las siguientes palabras reservadas es la que te permite mover un archivo de tu entorno local a tu imagen de Docker?**

**R/:** COPY

2. **¿Cuál es el orden adecuado de los conceptos más relevantes en Docker?**

**R/:** Dockerfile, imagen y contenedor

3. **¿Cómo se llama la herramienta que te permite desplegar todos tus contenedores de manera local con un solo comando?**

**R/:** Docker Compose

4. **¿Qué comando utilizarías para mostrar todos tus contenedores locales?**

**R/:** docker ps

5. **¿Cómo se debe llamar el archivo de Docker Compose para desplegar un conjunto de imágenes?**

**R/:** docker-compose.yml

6. **¿Qué parámetro debo colocar en el comando docker run para que yo me comunique al puerto 5600 y Docker lo haga con el puerto 5300 a la aplicación?**

**R/:** -p 5600:5300

7. **¿Qué comando debes utilizar para conocer toda la información de un contenedor de Docker?**

**R/:** docker inspect

8. **¿Qué palabra reservada debes usar en un Dockerfile para ejecutar un comando de terminal?**

**R/:** RUN

9. **¿Cuál es el parámetro en el comando docker run que reemplaza la palabra VOLUME en un Dockerfile?**

**R/:** -v origen:destino

10. **¿Cuál de las siguientes palabras NO es una palabra reservada de un Dockerfile?**

**R/:** PASTE

11. **¿Cuál es el comando que debes utilizar para generar un archivo comprimido a partir de una imagen de Docker?**

**R/:** docker save

12. **¿Cómo puedo eliminar una imagen de Docker desde la línea de comandos?**

**R/:** docker rmi

13. **Selecciona el fragmento de código que permite a docker compose desplegar un servicio**

**R/:** backend: image: backend build: context: ./backend dockerfile: Dockerfile ports: - "5000:5000"

14. **¿Cuál es la secuencia adecuada de etiquetado de una imagen para poder ser publicada en Docker Hub?**

**R/:** nombreDeUsuario/repositorio:version

15. **¿Cuál es el parámetro dentro del comando docker run que te permite ver lo que la consola de la aplicación está mostrando?**

**R/:** -it

16. **¿Cómo debes escribir la instrucción en tu Dockerfile para montar un volumen en tu contenedor?**

**R/:** VOLUME /var/log/app

17. **¿Qué comando utilizarías para mostrar todas tus imágenes locales?**

**R/:** docker images

18. **¿Qué comando debes utilizar para crear una red personalizada en Docker?**

**R/:** docker network create