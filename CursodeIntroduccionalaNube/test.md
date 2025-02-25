# Curso de Introducción a la Nube

1. **Tú tienes una aplicación ejecutandose en un servidor, sin embargo, te has dado cuenta que por el incremento de usuarios el rendimiento del servidor ha hecho la aplicación más demorada en responderle al usuario. Tú vas a incrementar los recursos del servidor para que pueda soportar la demanda creciente. ¿Qué tipo de escalabilidad estarás aplicando al hacer este cambio?**
   
**R//=** Escalabilidad Vertical, en la cual se aumentan recursos del servidor con un tiempo donde la aplicación estará caída.

2. **¿Cuál consideración deberías tener en cuenta en tu arquitectura para que tu aplicación sea altamente disponible?**
 
**R//=** Desplegar los servicios de la aplicación en al menos 2 zonas de disponibilidad (AZs) y utilizar un balanceador para el tráfico de entrada.

3. **Tienes tu aplicación ejecutándose en una arquitectura con un backend en servidores distribuidos en 2 AZs. Hiciste una campaña de marketing y los usuarios crecieron de forma repentina, sin embargo, ves algunas quejas de los usuarios a los cuales no les sirve la aplicación. Al verificar te diste cuenta que el escalamiento estaba funcionando adecuadamente. ¿Cuál puede ser la causa del problema?**
   
**R//=** En el escalamiento horizontal de los servidores, hay un tiempo mientras un nuevo servidor se crea para soportar más demanda de usuarios. Durante ese tiempo muchos usuarios no pudieron acceder a la aplicación.

4. **De acuerdo a los modelos de responsabilidad compartida de los proveedores de servicios de Cloud al utilizar funciones, ¿qué es responsabilidad del usuario?**
   
**R//=** Proveer el código de la función para ejecutarse y configurar las características de la función.

5. **Al desplegar una base de datos relacional en la nube, ¿cuál característica es la que le agrega alta disponibilidad?**
    
**R//=** Multi-Az

6. **¿Cómo se mide el costo de una función?**
    
**R//=** Por la memoria aprovisionada, cantidad de ejecuciones y por el tiempo de ejecución.

7. **¿Cuál es uno de los mayores retos de trabajar con funciones?**
    
**R//=** Cold Start o tiempo de inicio frío de las funciones.

8. **Al momento de desplegar tu aplicación en la nube, ¿cuál es un criterio para seleccionar la región?**
    
**R//=** Latencia, servicios ofrecidos y precio.

9. **Al diseñar una arquitectura para implementar una aplicación, ¿qué tipos de bases de datos podemos encontrarnos en los proveedores de servicios cloud?**
    
**R//=** Relacional, columnar, llave-valor, memoria, documental, series de tiempo y de grafos.

10. **¿Qué tipos de servicios de contenedores podemos encontrar en los proveedores de servicios cloud?**
    
**R//=** Serverless y basados en servidor

11. **¿Con qué servicio se puede exponer una función?**
    
**R//=** API Gateway

12. **¿Cómo puedo ampliar los límites que tienen los proveedores de servicios cloud en sus servicios?**
    
**R//=** A través de un caso de soporte.

13. **Al crear un servidor en nuestro proveedor de servicios cloud ¿Cómo es el costo del mismo?**
    
**R//=** Usualmente el costo es por minutos o segundos.

14. **Estás creando una aplicación y tienes dudas de donde deberias desplegarla ¿Cuál criterio puede ser útil para desplegarla en una nube pública y no en una nube privada?**
    
**R//=** Escalabilidad, seguridad, flexibilidad, pago por demanda, elasticidad, alta disponibilidad.

15. **Al tener tu aplicación corriendo en una arquitectura de contenedores tuviste un incremento en la demanda y la cantidad de microservicios crecieron. ¿Cómo llamamos a este tipo de escalabilidad?**

**R//=** Horizontal