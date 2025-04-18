### Curso de Swarm

1. **Indica cúal de las siguientes afirmaciones es verdadera:**

**R/:** En el escalado vertical el límite está dado por la potencia máxima que puedas conseguir en una máquina, mientras que en la escalabilidad horizontal está dado por cuántas máquinas puedas conseguir para correr tu carga, y tu capacidad de distribuir carga entre ellas.

2. **Una manera muy popular para garantizar la disponibilidad de una aplicación es escalarla verticalmente**

**R/:** Falso

3. **Indica la opción que mejor describe qué es Docker Swarm**

**R/:** La solución de clustering nativa de Docker, para escalar aplicaciones y distribuir carga entre múltiples máquinas que corren Docker

4. **Indica cuál de los siguientes no es un tipo de nodo de un cluster de Docker Swarm:**

**R/:** Scheduler

5. **Todos los comandos de Docker que se ejecuten en un nodo manager funcionarán igual en un nodo worker**

**R/:** Falso

6. **Una aplicación que no cumple los 12 factores…**

**R/:** siempre podrá correr en un contenedor, pero cuantos menos factores cumpla más difícil será aprovechar las funcionalidades de Docker y Docker Swarm.

7. **El comando para inicializar un Docker Swarm es:**

**R/:** docker swarm init

8. **El join token de un swarm es el mismo para cualquier tipo de nodo, y para decidir si unirse como manager o worker debe pasarse una opción al comando `docker swarm join`**

**R/:** Falso

9. **Si queremos inicializar un swarm en una máquina que tiene más de una interfaz de red, podemos especificar cuál debe usar para escuchar peticiones de nodos que quieren unirse al swarm usando la opción `--advertise-add`r**

**R/:** Verdadero

10. **Si ejecuto `docker swarm ini`t en un nodo manager de un swarm existente, lo que ocurre es:**

**R/:** El nodo abandona el swarm existente e inicializa un nuevo swarm, del cual es el único nodo, manager y líder

11. **Si quiero inspeccionar el estado del nodo de swarm en el que estoy, sin saber el ID del nodo, el comando que debo usar es:**

**R/:** docker node inspect self

12. **El comando para crear un servicio de Docker Swarm es**

**R/:** docker service create

13. **La principal diferencia entre `docker service ls` y `docker ps` es que el primero nos muestra la lista de servicios en todo el cluster de Docker Swarm y que el segundo nos muestra la lista de contenedores en el nodo donde ejecutamos el comando.**

**R/:** Verdadero

14. **Indicar cuál de las siguientes cosas no suceden cuando se ejecuta un comando `docker service create`**

**R/:** Se construye una imagen para cada tarea del servicio nuevo

15. **¿El comando para ver la lista de tareas de un servicio llamado “un-servicio” es:**

**R/:** docker service ps un-servicio

16. **Al ejecutar el comando docker service rm un-servicio lo que ocurre es:**

**R/:** Se elimina “un-servicio”, y de esta manera el scheduler indica a los nodos que tienen tareas de dicho servicio que pueden terminarlas. Dichos nodos se ocupan de eliminar las tareas y los contenedores asociados.

17. **Si queremos que nuestra terminal no quede bloqueada al ejecutar un comando de swarm que puede necesitar que un número de operaciones converjan, podemos usar la opción `-d` o `--dettach`**

**R/:** Verdadero

18. **El comando para actualizar cualquier aspecto de la configuración de un servicio de Docker Swarm es**

**R/:** docker service update

19. **Los comandos `docker service scale mi-servicio=10` y `docker service update --replicas=10 mi-servicio` son equivalentes**

**R/:** Verdadero

20. **El comando `docker service rollback` nos permite elegir de un número determinado de versiones anteriores del servicio a las cuales podemos volver**

**R/:** Falso

21. **Para garantizar que nunca habrá menos contenedores durante un `service update` o `service rollback` de los que había antes de ejecutar el comando, las opciones indicadas son, respectivamente `--update-order start-first` y `--rollback-order start-first`**

**R/:** Verdadero

22. **En un `docker service update`, combinar las opciones `--update-order stop-first` y `--update-parallelism=0` es una mala idea porque:**

**R/:** Elimina todas las tareas del servicio e intenta crear las nuevas tareas todas en simultáneo, lo cual nos deja durante todo ese tiempo con downtime.

23. **Dado un swarm de 3 nodos y un servicio "un-servicio"con una sóla réplica que publica el puerto 3000, si hacemos `docker service update --replicas=5 un-servicio`, lo que ocurre es:**

**R/:** El servicio escala a 5 réplicas y todas publican su puerto 3000, independientemente de la cantidad de nodos disponibles.

24. **Para crear un servicio de alpine cuyas tareas sólo pueden correr en nodos worker, el comando adecuado es:**

**R/:** docker service create --constraint=node.role==worker alpine

25. **Un nodo cuya disponibilidad está en modo drain puede planificar tareas, sólo que con menos prioridad que el resto de los nodos.**

**R/:** Falso

26. **Para poder intercomunicar contenedores de distintos servicios de un Docker Swarm es necesario conectarlos a al menos una misma red de tipo overlay**

**R/:** Verdadero

27. **Los contenedores de dos servicios distintos, conectados a la misma red overlay, pueden alcanzarse entre sí utilizando el nombre se servicio destino como hostname.**

**R/:** Verdadero

28. **La principal diferencia entre cómo `docker-compose` y `docker stack deploy` tratan a un compose file es:**

**R/:** El atributo deployment sólo funciona para docker stack deploy

29. **Indica cuál de los siguientes no es un número de managers indicado para un swarm productivo:**

**R/:** 1

30. **Si un servicio necesita poder ver el estado del swarm para ofrecer su funcionalidad, es absolutamente necesario restringir dicho servicio a los nodos manager**

**R/:** Verdadero