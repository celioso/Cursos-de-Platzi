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