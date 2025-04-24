# Curso Avanzado de Serverless Framework en AWS

1. **¿Cuáles son las opciones de seguridad que me ofrece el AWS Api Gateway?**
   
**R//=** Apis Keys y Custom Authorizers.

2. **¿Para qué sirven AWS SQS?**
 
**R//=** AWS SQS se utiliza para enviar, almacenar y recibir mensajes de forma asincrónica entre componentes de software, lo que ayuda a desacoplar y escalar las aplicaciones en la nube de AWS.

3. **¿Qué casos de uso puede ser asíncronos e implementados con lambda?**
   
**R//=** Eliminación de cuentas, validación de pagos y procesamiento de imágenes.

4. **Caso de uso: Si requiero gestionar comunicacion entre funciones lambdas, ¿qué servicio debería usar?**
   
**R//=** AWS SQS o SNS

5. **¿Qué es AWS S3?**
    
**R//=** Servicio de almacenamiento de objetos que permite almacenar imágenes, videos, archivos de todo tipo, configurar versionamiento, entre otras funcionalidades.

6. **Los recursos creados usando serverless framework son gestionados por esta herramienta de AWS.**
    
**R//=** AWS Cloudformation

7. **¿Qué eventos de S3 pueden disparar o lanzar una lambda?**
    
**R//=** Creación y eliminación de objetos, eventos de replicación entre buckets, eventos del ciclo de vida de expiración y transición.

8. **¿Cuáles son algunos de los beneficios de usar lambda layers?**
    
**R//=** Se optimiza el tamaño de la lambda al extraer las librerías y dependencias, lo que logra tiempos de inicio y cold-start menores.

9. **¿Cuál de los siguientes NO es un mecanismo que permite asegurar una App Serverless?**
    
**R//=** Random Token

10. **El procesamiento en segundo plano de imágenes permite agregar asincronismo y reducir cuellos de botella en la ejecución de la aplicación. ¿Cuál de los servicios usaría para implementar este caso de uso? Considere usar servicios Serverless.**
    
**R//=** Lambda y S3