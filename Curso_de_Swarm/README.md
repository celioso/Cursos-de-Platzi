# Curso de Swarm

## ¿Qué es Swarm?

**Docker Swarm**, lo que hace es, tener un Cluster de muchas maquinas, pero desde afuera, para los usuarios Developers (Administrativos, Operadores) se vea como un Docker Deamon.

Esto hace que parezca que estamos usando Docker Local (en nuestras computadoras) pero tenemos acceso a muchos nodos o maquinas que están corriendo Docker Deamon.

Y Docker Swarm lo que nos facilita es toda la paste de actualización de servicios, rotación, administración, etc. Para lograr una alta disponibilidad de nuestro servicio.

## El problema de la escala: qué pasa cuando una computadora sóla no alcanza

La **escalabilidad** es el poder aumentar la capacidad de potencia de computo para poder servir a más usuarios o a procesos más pesados a medida que la demanda avanza.

A la hora de hablar de escalabilidad encontramos dos tipos de soluciones, *escalabilidad vertical*, que consiste en adquirir un mejor hardware que soporte mi solución o una *escalabilidad horizontal*, en la cual varias máquinas están corriendo el mismo software y por lo tanto es la solución más utilizada en los últimos tiempos.

La **disponibilidad** es la capacidad de una aplicación o un servicio de poder estar siempre disponible (24 horas del día), aún cuando suceda un improvisto.

**Es mayor la disponibilidad cuando se realiza escalabilidad horizontal**

Swarm no ofrece la solución a estos problemas.

- Ejecutar aplicaciones productivas: la aplicación debe estar lista para servir a las usuarios a pesar de situaciones catastróficas, o de alta demanda (carga).
- Escalabilidad: Poder aumentar la potencia de cómputo para poder servir a más usuarios, o a peticiones pesadas.
- Escalabilidad vertical: Más hardware, hay límite físico.
- Escalabilidad horizontal: Distribuir carga entre muchas computadoras. es el más usado.
- Disponibilidad: Es la capacidad de una aplicación o servicio de estar siempre disponible para los usuarios. prevé problemas con servidores, etc.
- La escalabilidad horizontal y la disponibilidad van de la mano.

## Arquitectura de Docker Swarm

La arquitectura de Swarm tiene un esquema de dos tipos de servidores involucrados: los managers y los workers.

- Los managers son encargados de administrar la comunicación entre los contenedores para que sea una unidad homogénea.

- Los workers son nodos donde se van a ejecutar contenedores, funciona como un núcleo, los contenedores estarán corriendo en los workers.

**Todos deben tener Docker Daemon (idealmente la misma versión) y deben ser visibles entre sí.**

## Preparando tus aplicaciones para Docker Swarm: los 12 factores

**¿ Está tu aplicación preparada para Docker Swarm ?**

Para saberlo, necesitas comprobarlo con los 12 factores

1. **Codebase**: el código debe estar en un repositorio
2. **Dependencies**: deben estar declaradas en un archivo de formato versionable, suele ser un archivo de código
3. **Configuration**: debe formar parte de la aplicación cuando esté corriendo, puede ser dentro de un archivo
4. **Backing services**: debe estar conectada a tu aplicación sin que esté dentro, se debe tratar como algo externo
5. **Build, release, run**: deben estar separadas entre sí.
6. **Processes**: todos los procesos los puede hacer como una unidad atómica
7. **Port binding**: tu aplicación debe poder exponerse a sí misma sin necesidad de algo intermediario
8. **Concurrency**: que pueda correr con múltiples instancias en paralelo
9. **Disposabilty**: debe estar diseñada para que sea fácilmente destruible
10. **Dev/Prod parity**: lograr que tu aplicación sea lo más parecido a lo que estará en producción
11. **Logs**: todos los logs deben tratarse como flujos de bytes
12. **Admin processes**: la aplicación tiene que poder ser ejecutable como procesos independientes de la aplicación

[The Twelve-Factor App](https://12factor.net/)

## Tu primer Docker Swarm

- Docker swarm es un modo de usar docker.
- `docker swarm init`: para iniciar docker swarm. Cada vez que se inicia como swarm, el equipo se convierte en manager.
- `docker swarm join-token manager`: para unirse como manager.
- `docker swarm join --token TOKEN server:puerto`: para unir un nodo como worker al server con puerto 2377
- `docker node ls`: para ver los nodos swarm de docker.
El manager cuenta con un certificado, con el cuál encripta la comunicación entre los workers y los managers.
- `docker node inspect --pretty self`: para ver los datos del nodo y del certificado.
- `docker swarm leave`: para salir de swarm de docker.
- `docker swarm leave --force`: para forzar la salida del docker swarm, como cuando eres el último manager.

[Manage swarm security with public key infrastructure (PKI) | Docker Documentation](https://docs.docker.com/engine/swarm/how-swarm-mode-works/pki/)

## Fundamentos de Docker Swarm: servicios

- se inicia docker swarm `docker swarm init`
- se crea un servicio llamado pinger y se usa al imagen alpine con ping de dirrección de google para el ejemplo: `docker service create --name pringer alpine ping www.google.com`
- listamos los servicios activos: `docker service ls`

En swarm siempre tienes que tener algún mecanismo de mantenimiento y limpieza. Yo uso [meltwater/docker-cleanup](https://hub.docker.com/r/meltwater/docker-cleanup "meltwater/docker-cleanup") corriendo como un servicio global de Swarm, lo que me garantiza que corre en todos los nodos, y delego en él la tarea de limpiar todo. Lo hago así:

```bash
docker service create \
  --detach \
  -e CLEAN_PERIOD=900 \
  -e DELAY_TIME=600 \
  --log-driver json-file \
  --log-opt max-size=1m \
  --log-opt max-file=2 \
  --name cleanup \
  --mode global \
  --mount type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock \
  meltwater/docker-cleanup
```

[Deploy services to a swarm | Docker Documentation](https://docs.docker.com/engine/swarm/services/)

## Entendiendo el ciclo de vida de un servicio

Desde el Cliente , ‘docker service create’ le envía al Nodo Manager el servicio: se crea, se verifican cuántas tareas tendrá, se le otorga una IP virtual y asigna tareas a nodos; esta información es recibida por el Nodo Worker, quien prepara la tarea y luego ejecuta los contenedores.

![Docker node](./images/node1.jpg)

comandos

`docker service ps pinger`
`docker service inspect pinger`
`docker service inspect --pretty pinger`
`docker service logs pinger`
`docker service logs -f pinger` para que siga sacando el resultado
`docker service rm pinger`

[How services work | Docker Documentation](https://docs.docker.com/engine/swarm/how-swarm-mode-works/services/)

## Un playground de docker swarm gratuito: play-with-docker

¡Play with docker es una herramienta de otro mundo! Te permitirá colaborar entre distintos usuarios en una mismas sesión de docker, y lo mejor, ¡puedes incluir tu propia terminal!

[Play with Docker](https://labs.play-with-docker.com/)

### Características de play-with-docker:

- Tiempo de duración: 4 horas
- Memoria RAM: 4GB
- Docker & Compose
- Git
- Conexión por SSH
- IP propia.
- Puedes agregar mas instancias (IP)

## Docker Swarm multinodo

Ampliar la terminal: alt + enter

`docker swarm init --advertise-addr IP`: para iniciar docker swarm con ip en específico
`docker swarm join --token TOKEN server:puerto`, para unir un nodo como worker al server con puerto 2377
`docker node ls`: para ver los nodos swarm de docker.

- Creamos el nodo manager

```bash
docker swarm init --advertise-addr <MANAGER-IP>
docker swarm init --advertise-addr 192.168.0.18
```
- Creamos una nueva instancia en play-with-docker (+ ADD NEW INSTANCE)

```bash
docker swarm join --token <TOKEN> <MANAGER-IP>:<PORT>
docker swarm join --token SWMTKN-1-32cege8duoof9cr405bi1fsmcga831l6fcecmznp5cxcfdc3vg-ci6f98tjfy9fzhr2swmmo3ter 192.168.0.18:2377
```
- Nos dirigimos a la terminal del nodo MANAGER, observamos los 3 nodos

```bash
docker node ls
```
- Crear un servicio en este caso multinodo

```bash
docker service create --name pinger alpine ping www.google.com
```
- Ver listado de servicios

```bash
docker service ls
```
- Donde estan asignado las tareas de este servicio, nos indica que esta en el nodo 1

```bash
docker service ps pinger
```

- Podemos ver el container

```bash
docker ps
```

## Administrando servicios en escala

Tenemos un swarm con multiples nodos, vamos a tratar de hacer es escalar nuestro servicio de pinger para que tenga muchas replicas

- Ejecutar scale para 5 tareas

```bash
docker service scale <nameService>=<cantidad>
docker service scale pinger=5
```
- Ver el estado de este servicio

```bash
docker service ps pinger
```
- Listamos contenedores para ver que estan corriendo mas de un contenedor
```bash
docker ps
```
- Vamos a ver los logs de este servicio, observamos que todo junto de todas las tareas sin perdernos ningun valor (Ctrl+C) Salir
```bash
docker service logs -f pinger 
```
- Inspeccionamos el servicio para ver cuantas replicas tiene
```bash

docker service inspect pinger
```
- Actualizar en caliente, cambiar configuracion
```bash
docker service update --args "ping www.amazon.com" pinger
docker service logs -f pinger 
```
- Si hay algun problema podemos volver a un estado anterior, es decir hacer un rollback
```bash
docker service rollback pinger
```
- Verificamos
```bash

docker service inspect pinger
```

- ver los servicios
```bash
docker service ls
```

## Controlando el despliegue de servicios

- Actualizar las replicas de un servicio, agregando mas replicas
```bash
docker service update --replicas=20 pinger
docker service update -d --replicas=20 pinger
```
- Actualizar paralelismo y orden de la configuración de update en el servicio pinger (stop-first: Arranca con esta cantidad de tareas, start-first: Creame mas tareas cuando esten listas las nuevas y borramela las viejas)
```bash
docker service update --update-parallelism 4 --update-order start-first pinger
docker service inspect pinger
docker service update --args "ping www.facebook.com" pinger
```


- Actualizar accion en fallo y radio maximo de falla de la configuración de update en el servicio pinger
```bash
docker service update --update-failure-action rollback --update-max-failure-ratio 0.5 pinger
```
- Actualizar paralelismo de la configuracion de rollback en el servicio de pinger
```bash
docker service update --rollback-parallelism 0 pinger
```

### Comandos

`docker service scale pinger=10`
`docker service  update --replicas=20 pinger`
`docker service  update -d --replicas=20 pinger`
`docker service inspect pinger`
`docker service  update --update-parallelism 4 --update-order start-first pinger`
`docker service inspect pinger`
`docker service update --args ping www.facebook.com" pinger`
`docker service ps pinger`
`docker service update --update-failure-action rollback --update-max-failure-ratio 0.5 pinger`
`docker service update --rollback-parallelism 0 pinger`
`docker service update --args "ping www."; pinger`

- `docker service update --replicas=<n> <servicename>`: actualiza el numero de replicas del servicio.
- `docker service update --update-parallelism <n> --update-order <start-first> <servicename>`: configura los n nodos que se actualizaran en paralelo, así como se indica el update order.
- `docker service update --update-failure-action rollback --update-max-failure-ratio <n> <servicename>`: configura el max n de fallas de un update antes de realizar un rollback del servicio.
- `docker service update --rollback-parallelism <n> <servicename>`: configura los n nodos que se haran rollback en paralelo, [0=todos]

## Exponiendo aplicaciones al mundo exterior

exacto solo sigue los mismos comandos de la clase


`git clone git@github.com:platzi/swarm.git`
`cd swarm/hostname`
`docker build -t <your-docker-hub-id>/swarm-hostname`
`docker login --username <your-user-id>`
`docker push <your-docker-hub-id>/swarm-hostname`

y despues en labs.docker


```bash
docker service create -d --name app --publish 3000:3000 --replicas=3 <your-user-id>/swarm-hostname
  ```

  ver los sertvicios `docker service ps app`
  para ver desde el depliegue se utiliza el `curl<el link de lab.swarm>` ejemplo: `curl http://ip172-18-0-26-cqsdt4aim2rg00c1oeag-3000.direct.labs.play-with-docker.com/` y el  responde desde un contenedor distinto.

[Page not found · GitHub](https://github.com/platzi/swarm)

## El Routing Mesh

**Routing Mesh** nos ayuda a que, teniendo un servicio escalado en swarm que tiene mas nodos que servicios y esos servicios están expuestos en un puerto; cuando hacemos un request a un servicio en ese puerto de alguna manera la petición llega y no se pierden en algún nodo que no puede contenerlo en ese puerto o en un contenedor.

Routing Mesh ayuda a llevar la petición y que esta no se pierda si la cantidad de los contenedores es diferente a la cantidad de nodos.

`docker service scale app=6`
`docker service ps app`
`docker ps`
`docker network ls`
`docker service create --name app --publish 3000:3000 --replicas=3 <your-user-id>/swarm-hostname:latest`

ver redes: `docker network ls`

[Use swarm mode routing mesh | Docker Documentation](https://docs.docker.com/engine/swarm/ingress/)

## Restricciones de despliegue

Todo funciono. Una clase excelente y practica.

- Vamos a crear un servicio nuevo
`docker service create --name viz -p 8080:8080 --constraint=node.role==manager --mount=type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock dockersamples/visualizer`

- Verificiamos

`docker service ps viz`

- Haremos que todos nuestros servicios corran en workers

`docker service update --constraint-add node.role==worker --update-parallelism=0 app`

http://192.168.5.151:8080/

## Disponibilidad de nodos

En el visualizer podemos ver que todas las tareas siguen corriendo en el "“worker 1"”, esto sucede porque el planificador de Docker Swarm no va a replanificar o redistribuir la carga de un servicio o de contenedores a menos que tenga que hacerlo; para solucionar esto, debemos forzar un redeployment o una actualización que se logra cambiando el valor de una variable que no sirva para nada.

`docker service update -d --env-add UNA_VARIABLE=de-entorno --update-parallelism=0 app`

`docker service ps app`

- Como hacemos decirle que esto de aqui sacamelo

```bash
docker node ls
docker node inspect --pretty worker2
```
- Le pasamos las replicas a los otros workers, es decir le quitamos replicas nuestro worker2

```bash
docker node update --availability drain worker2
```
- Vamos a nuestro worker1 y observamos que tiene mucha carga
- Vamos a nuestro manager, verificamos que nuetro worker2 esta en modo DRAIN y lo cambiamos a ACTIVE (observamos que no redistribuye las replicas a worker2)
```bash
docker node ls
docker node update --availability active worker2
```
- Con este metodo podemos redistribuir, enviando las tareas a otros workers
```bash
docker service update -d --env-add UNA_VARIABLE=de-entorno --update-parallelism=0 app
docker service ps app
```
`docker node ls`
`docker  node inspect --pretty worker2`
`docker node update --availability drain worker2` deactivar el worker2 
`docker node update --availability active worker2` activar el worker2 
`docker service update -d --env-add UNA_VARIABLE=de-entorno --update-parallelism=0 app`
`docker service rm app` eliminar la app

Disponibilidad de un nodo:

- **active**: El nodo trabaja de forma coordinada con el Swarm y puede recibir nuevas tareas.
**pause**. El nodo deja de recibir nuevas tareas de manera indefinida, pero no desecha las que ya tiene corriendo.
**drain**: Le indica a Swarm que vacíe todas las tareas del nodo y lo mantenga así de manera indefinida. Si un servicio requiere un número definido de réplicas, Swarm redistribuye la respectiva carga del nodo drenado a otro nodo disponible.
Ver más en [Administrar nodos de Swarm.](https://docs.docker.com/engine/swarm/manage-nodes/)

## Networking y service discovery

vamos a ver como es el networking al momento de trabajar con swarm, vamos a poder inspeccionar que tenemos en la red. Todo esto, utilizando el repositorio que encuentras en los enlaces del curso

[https://docs.docker.com/network/overlay/](https://docs.docker.com/network/overlay/ "https://docs.docker.com/network/overlay/")

- Nos situamos en manager1 y creamos una nueva red

```bash
docker service rm app
docker network ls
docker network create --driver overlay app-net
```
- Vamos a usar los archivos de swarm/networking en nuestro local.

```bash
cd swarm/networking
docker build -t borisvargas/swarm-networking .
docker push borisvargas/swarm-networking
```
- Nos situamos en manager. Creamos el servicio de mongo

```bash
docker service create -d --name db --network app-net mongo
docker service ps db
```
- Creamos el servicio de app

```bash
docker service create -d --name app --network app-net -p 3000:3000 <your-user-id>/swarm-networking
docker service ps app
docker service update --env-add MONGO_URL=mongodb://db/test app
docker exec -it <ID_CONTAINER_DB> bash
# mongo
> use test
> db.pings.find()
> db.pings.find()
```
- Esta es la manera como se comunican en un entorno de swarm

```bash
docker network inspect app-net
```
Las redes en docker son muy potentes y simplifican trabajo.

`docker network `
`docker network create --driver overlay app-net`
`docker network inspect app-net`
`docker build -t <your-user-id>/networking .`
`docker login`
`docker push <your-user-id>/networking`
`docker service create -d --name db --network app-net mongo`
`docker service ps db`
`docker service create --name app --network app-net -p 3000:3000 <your-user-id>/networking`
`docker service ps app`
`docker service update --env-add MONGO_URL=mongodb://db/test app`
`docker ps`
`docker exec -it 0491a70c18cf bash`
`mongo`
`use test`
`db.pings.find()`
`docker network inspect app-net`

[Use overlay networks | Docker Documentation](https://docs.docker.com/network/overlay/)

## Docker Swarm stacks

Con Docker Swarm Stacks (un archivo) se puede controlar cómo se van a despliegan los servicios utilizando los stacks. Siempre es bueno utilizar un archivo porque este puede ser versionado (Git) y se tiene un archivo que va a describir la arquitectura de la aplicación.

- Nos dirigimos a nuestro repositorio https://github.com/platzi/swarm, a la carpeta stack.

```bash
cd swarm/stacks
cat stackfile.yml
```

- Nos dirigimos a nuestro manager1
```bash
docker service rm app db
docker network rm app-net
vim stackfile.yml
```
sen el editor escogemos pegar con `:set paste`, para ver el contenido del archivo se utiliza `cat stackfile.yml`

Modo de comandos
i: entrar al modo de inserción
v: entrar al modo visual (selección de palabras)
V: entrar al modo visual (selección de líneas)
w: avanzar al principio de la siguiente palabra
e: avanzar al final de la siguiente palabra
b: retroceder al principio de la palabra anterior
y: copiar (yank) se usa con otros comando para seleccionar texto a copiar
d: borrar (el texto borrado se copia al portapapeles, o sea que también hace la función de cortar)
p: pegar
u: deshacer
:w: guardar fichero actual
:q: salir
:tabe : abrir nueva pestaña con un archivo, si se omite se abre un nuevo buffer vacío
gt: mover a la siguiente pestaña
gT: mover a la anterior pestaña
h, j, k, l: mover cursor una posición a la izquierda, abajo, arriba y a la derecha

```bash
###########stackfile.yml##############
version: "3"

services:
  app:
    image: borisvargas/swarm-networking
    environment:
      MONGO_URL: "mongodb://db:27017/test"
    depends_on:
      - db
    ports:
      - "3000:3000"

  db:
    image: mongo
######################################
```

```bash
docker stack deploy --compose-file stackfile.yml app

docker stack ls

docker stack ps app

docker stack services app
```
- Quiero que los servicios esten en los workers

```bash
vim stackfile.yml
```


```bash
###########stackfile.yml##############
version: "3"

services:
  app:
    image: borisvargas/swarm-networking
    environment:
      MONGO_URL: "mongodb://db:27017/test"
    depends_on:
      - db
    ports:
      - "3000:3000"
    deploy:
      placement:
        constraints: [node.role==worker]

  db:
    image: mongo
######################################
```
`docker stack deploy --compose-file stackfile.yml app`
`docker stack rm app`

comandos usados:

`docker service rm app`
`docker service rm db`
`docker network rm app-net`
`docker stack deploy --compose-file stackfile.yml app`
`docker service ls`
`docker stack ls`
`docker stack ps app`
`docker stack services app`
`docker service scale app_app=3`
`docker stack rm app`

## Reverse proxy: muchas aplicaciones, un sólo dominio

Reverse proxy es una técnica, es un servicio que está escuchando y que toma una decisión con la petición que esta entrando y hace un proxy hacia uno de los servicios que tiene que atender esa petición.

Existe una herramienta llamada traefik, el cual es un intermediario entre las peticiones que vienen del internet a nuestra infraestructura.

Que tal, tuve un pequeño problema. Cuando intente correr el comando, se quedan como en un ciclo, entonces cancele y revise los logs con

```bash
docker service logs proxy 
```

Y me mandaba este mensaje.

2019/10/21 00:26:42 command traefik error: failed to decode configuration from flags: field not found, node: docker

Y para arreglarlo, solo tuve que definir la versión de traefik y cambiar esta parte

```bash
  --docker.swarmMode
```

Por

```bash
--docker.swarmmode
```


El comando final quedo así:

```bash
docker service create --name proxy --constraint=node.role==manager -p 80:80 -p 9090:8080 --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock --network proxy-net traefik:1.5 --docker --docker.swarmmode --docker.domain=domain.ca --docker.watch --api
```

comandos:

- `docker network create --driver overlay proxy-net`
- `docker service create --name proxy --constraint=node.role==manager -p 80:80 -p 9090:8080 --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock --network proxy-net traefik --docker --docker.swarmmode --docker.domain=dbz.com --docker.watch --api`
- `docker service create  --name app1 --network proxy-net --label traefik.port=3000 celismario/swarm-hostname`
- `curl -H 'Host: app1.dbz.com' http://localhost`
- `docker service create  --name app2 --network proxy-net --label traefik.port=3000 celismario/swarm-hostname`
- `curl -H "Host: app2.dbz.com" http://localhost`
- `docker service update --image baezdavidsan/networking app2`
- `curl -H "Host: app1.dbz.com" http://localhost`
- `curl -H "Host: app2.dbz.com" http://localhost`
 
[Traefik - The Cloud Native Edge Router / Reverse Proxy / Load Balancer](https://traefik.io/)

## Arquitectura de un swarm productivo

 Para que Docker Swarm funcione (utilizando buenas practicas), tiene que haber un número impar de Managers. -En la arquitectura de Swarm debe haber un Nodo Manager Leader que toma la decisión final (creación, asignación, destrucción, tareas, servicios, etc.) y los otros replican lo que dice el líder. -Cada cierto tiempo se rota el Status Leader entre los nodos Managers utilizando un algoritmo: [Raft] Utilizado en Clustering.

- Como mínimo necesitamos tres manager.
- Podemos configurar grupos de workers según la necesidad de computo.
- El número de manager debe ser impar. Hay un único líder, y se rotan el liderazgo en un intervalo de tiempo

Si, se usan los labels y el parámetro constrain, aquí tienes un ejemplo [https://success.docker.com/article/using-contraints-and-labels-to-control-the-placement-of-containers](https://success.docker.com/article/using-contraints-and-labels-to-control-the-placement-of-containers)

## Administración remota de swarm productivo

Las herramientas de administración en Docker Swarm deben persistir en disco (su estado interno, la administración) y la mejor manera de almacenar cosas en Docker son los volúmenes.

En esta clase aprenderemos una forma fácil simple e intituiva de administrar nuestro docker swarm de manera remota. No es la única que existe, así que te invitamos a probar y a dejarnos en los comentarios otras formas que encuentres.

Lecturas recomendadas

[Docker for Azure persistent data volumes | Docker Documentation](https://docs.docker.com/docker-for-azure/persistent-data-volumes/)

[REX-Ray](https://rexray.io/)

[Portainer | Simple management UI for Docker](https://portainer.io/)

[GitHub - dockersamples/example-voting-app: Example Docker Compose app](https://github.com/dockersamples/example-voting-app)

**comandos**

- `docker volume create portainer_data`
- `docker volume ls`
- `docker service create --name portainer -p 9000:9000 --constraint node.role==manager --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock --mount type=volume,src=portainer_data,dst=/data portainer/portainer -H unix:///var/run/docker.sock`
- `docker stack ls`
- `docker stack ps voting`
- `docker service ps voting_worker`
http://192.168.5.150:5000/
http://192.168.5.150:5001/

## Consideraciones adicionales para un swarm produtivo

```bash
docker service create -d \
-e CLEAN_PERIOD=900 \
-e DELAY_TIME=600 \
--log-driver json-file \
--log-opt max-size=1m \
--log-opt max-file=2 \
--name=cleanup \
--mode global \
--mount type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock \
meltwater/docker-cleanup
```

`docker service ls`

Lecturas recomendadas
[GitHub - meltwater/docker-cleanup: Automatic Docker image, container and volume cleanup](https://github.com/meltwater/docker-cleanup)

[Configure logging drivers | Docker Documentation](https://docs.docker.com/config/containers/logging/configure/)

[GitHub - stefanprodan/swarmprom: Docker Swarm instrumentation with Prometheus, Grafana, cAdvisor, Node Exporter and Alert Manager](https://github.com/stefanprodan/swarmprom)