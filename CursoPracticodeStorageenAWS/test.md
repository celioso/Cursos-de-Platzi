# Curso Práctico de Storage en AWS

1. **Cuando se usa versionamiento en S3, ¿qué pasa con las versiones anteriores del objeto?**
   
**R//=** Quedan con una marca de eliminación e igualmente visibles con todos los objetos del bucket.

2. **Estás intentando crear un sitio web estático en S3 para conectarlo al dominio corporativo de tu empresa, sin embargo, al momento de crear el bucket te das cuenta que el nombre ya esta siendo utilizado. ¿Qué podrías hacer para solucionar este problema?**
 
**R//=** No se puede crear el sitio web dado que el nombre del bucket está siendo utilizado, se debería utilizar otro dominio verificando antes que el nombre del bucket se encuentre disponible.

3. **El área de auditoria de tu empresa está solicitando un registro de todas las acciones que se ejecutan sobre el bucket en el cual guardan la información crítica, para cumplir con este requerimiento deberías:**
   
**R//=** Configurar el log a nivel de objeto en el bucket.

4. **¿Cuál funcionalidad te permite utilizar los CDN de AWS para mejorar la carga de archivos a S3?**
   
**R//=** Transferencia Acelerada.

5. **Por contingencia deseas tener una copia de uno de tus buckets más importantes, para este objetivo ¿Cuál funcionalidad deberías habilitar en S3?**
    
**R//=** Replicación.

6. **¿Cuál es la disponibilidad de S3 Estándar?**
    
**R//=** 99,99%

7. **Eres el CIO de una empresa y debes guardar la información de los usuarios en S3. Esta información será consultada 1 o 2 veces al mes, así que necesitas seleccionar la clase de almacenamiento más económica que tenga mejor durabilidad y HA, ¿Cuál escogerías?**
    
**R//=** S3 - IA.

8. **En la empresa en la que trabajas te piden una opción para almacenar los archivos históricos, los cuales son consultados una vez cada año. Actualmente hay 5TB de históricos y debes seleccionar el mejor almacenamiento para este fin ¿cuál escogerías?**
    
**R//=** Glacier.

9. **Trabajas como arquitecto de soluciones para una universidad y te piden diseñar un sistema para que las tesis de grado puedan ser configuradas para que después del año de presentación queden guardadas teniendo en cuenta que son poco consultadas, tú quieres automatizar este proceso, ¿Qué funcionalidad usarías?**
    
**R//=** Ciclo de vida de s3.

10. **En tu empresa quieren migrar a la nube de AWS, sin embargo no saben cómo pueden migrar 5TB de información y solo cuentan con un canal de internet de 5Mbps, ¿qué les recomendarías utilizar para cumplir con esta tarea?**
    
**R//=** Usar Snowball.

11. **¿Cuál de los siguientes casos de uso considerarías para usar S3?**
    
**R//=** Almacenar logs para proyectos de BigData.

12. **Por defecto, todos los buckets vienen con encriptación habilitada.**
    
**R//=** Falso.

13. **Por una regulación del gobierno debes encriptar todos los datos que tengas en S3. Ya que quieres manejar totalmente las llaves, tanto su creación, rotación y almacenamiento, ¿cuál sistema de encriptación deberías usar?**
    
**R//=** SSE-C

14. **¿Cuándo usamos SSE-KMS el uso de la llave queda registrado en CloudTrail?**
    
**R//=** Verdadero.

15. **Al usar SSE-S3 la llave que encripta ¿es encriptada y almacenada también en S3?**

**R//=** Verdadero.

16. **Al crear una política ¿cuál de los siguientes componentes es obligatorio?**

**R//=** Statement

17. **¿Cuál componente de una política es una capa adicional de seguridad donde podemos agregar lógica de validaciones?**

**R//=** Condition

18. **Como arquitecto de soluciones te han pedido determinar la mejor forma de darle acceso a un bucket a un usuario de otra cuenta de AWS, ¿qué deberías configurar en S3 para llevar a cabo esta tarea?**

**R//=** La ACL y la Policy.

19. **¿Cuál de los siguientes permite realizar conexiones entre almacenamiento local y almacenamiento en nube?**

**R//=** Storage Gateway

20. **¿Cuál tipo de Storage Gateway seleccionarías para guardar las cintas on-premise en AWS?**

**R//=** Virtual Tape Library

21. **EFS ¿se puede asociar solo a una instancia EC2?**

**R//=** Falso.

22. **¿Con cuál sistema operativo no tiene compatibilidad EFS?**

**R//=** Windows Server

23. **¿A cuántas instancias se puede asociar un volumen EBS?**

**R//=** 1

24. **¿En los volúmenes GP2 cuál es la relación de IOPS y GB?**

**R//=** 3 IOPS por GB

25. **¿Qué tipo de EBS usarías para un servidor que almacenará archivos de BigData, específicamente logs para mejorar el rendimiento?**

**R//=** ST1

26. **¿Puedes encriptar un volumen raíz de una instancia?**

**R//=** Falso