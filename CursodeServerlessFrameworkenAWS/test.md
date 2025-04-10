# Curso de Serverless Framework en AWS

1. **¿Qué es AWS Serverless?**
   
**R//=** Es un ecosistema de servicios que provee AWS, en el cual según el modelo de responsabilidad compartido, nosotros como usuarios ejecutamos nuestro código, información y podemos integrar diferentes apps y servicios sin gestionar o mantener ningún servidor.

2. **¿Qué es Serverless Framework?**
 
**R//=** Es un marco de trabajo con el cual podemos construir soluciones usando un serverless.yml. Este archivo nos define un conjunto de recursos como el provider, function y los valores custom.

3. **¿Cuáles de los siguientes servicios de AWS permiten construir soluciones Serverless?**
   
**R//=** AWS Lambda y SQS (Simple Queue Service)

4. **S¿Qué es CloudFormation?**
   
**R//=** Una herramienta propietaria de AWS para hacer IaC y definir servicios.

5. **¿Cual de los siguientes son bloques del serverless.yml cuando usamos Serverless framework?**
    
**R//=** Seleccionar múltiples instancias en diferentes zonas de disponibilidad

6. **¿Cuales de los siguientes son comandos del Serverless Framework?**
    
**R//=** Deploy, Remove, Rollback

7. **Falso o verdadero: ¿Si quiero usar Serverless Framework para mis nanoservicios solo puedo usar AWS como cloud provider?**
    
**R//=** Falso

8. **¿Qué proveedores Cloud serian compatibles con Serverless framework?**
    
**R//=** AWS, Alibaba Cloud, Azure, GCP, Cloudflare

9. **¿Cuáles son los diferentes servicios o triggers que aceptan las funciones lambdas cuando usamos Serverless Framework?**
    
**R//=** Eventos de S3, HTTP (Api Gateway), cloudwatch, Eventbridge, SNS y SQS

10. **¿Cuáles son algunas de las buenas prácticas al momento de usar Serverless Framework?**
    
**R//=** Usar múltiples funciones y separar la lógica de negocio en lambdas independientes, así cuando haya un incidente con algún componente los demás componenters del sistema no se verán afectados.

11. **¿Al mencionar la ventaja de rápida escalabilidad de las Lambdas en AWS, hacemos referencia a que pilar del AWS Well Architected Framework?**
    
**R//=** Reliability

12. **Los recursos creados usando serverless framework son gestionados por cual herramienta en AWS:**
    
**R//=** AWS CloudformationS

13. **¿Para qué es usado el bloque de provider dentro del archivo serverless.yml?**
    
**R//=** Definir todo lo asociado al cloud provider en el cual se alojara la aplicación y los diferente recursos definidos en el serverless.yml

14. **¿Cuál es la funcionalidad de los plugins en el Serverless Framework?**
    
**R//=** Agregar funcionalidades custom creadas por la comunidad que facilitan el trabajo o le agregan super poderes al framework

15. **¿Es la sección del serverless.yml en la cual puedo definir los triggers para cada una de mis funciones?**

**R//=** Dentro del bloque functions en la sección de events, aca se me permite asociar los triggers para cada uno de los handlers que defina en mis funciones lambda
    
16. **Si deseo construir un Rest API, ¿qué tipo de trigger debo usar en mi lambda function?**
    
**R//=** El trigger http o httpApi, permiten hacer llamados usando métodos del protocolo HTTP

17. **¿Qué es DynamoDB?**
    
**R//=** Una base de datos no relacional del ecosistema serverless de AWS

18. **¿Cuáles son los elementos que nos permiten acceder a la información en una base de datos DynamoDB?**
    
**R//=** Partition key y Sort key

19. **Seleccione la afirmación correcta sobre DynamoDB:**
    
**R//=** DynamoDB es un servicio de bases de datos no relacional, el cual utiliza en partition key y sort key para acceder a la información

20. **¿Qué comando nos permite hacer la creación de un proyecto Serverless Framework basado en un template?**
    
**R//=** serverless create --template