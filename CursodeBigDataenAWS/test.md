# Curso de Big Data en AWS

1. **¿Cómo se llama la arquitectura que no tiene capa Batch?**
   
**R//=** Kinesis o Lambda

2. **¿Cuál servicio en AWS podemos utilizar para IDE?**
 
**R//=** Cloud9

3. **¿Cómo se llama la librería de Python para interactuar con los servicios de AWS ?**
   
**R//=** `Boto3`

4. **¿Cuál servicio de AWS puede actuar como una "Puerta Frontal" para recibir cientos de miles de llamadas ?**
   
**R//=** API Gateway

5. **¿En cuál de los siguientes casos usaría storage gateway?**
    
**R//=** Cuando quiera llevar información de mi app onpremise a la nube.

6. **¿Qué componen una agrupación de data records?**
    
**R//=** Un Shard

7. **Por defecto al configurar kinesis data stream, ¿Cuál es el periodo de retención?**
    
**R//=** 24 Horas

8. **¿Cuál es el principal beneficio de realizar despliegues de infraestructura utilizando CloudFormation?**
    
**R//=** Control de la infraestructura como código.

9. **¿Qué servicio deberías utilizar para alimentar un servicio como ElasticSearch?**
    
**R//=** Kinesis Firehose

10. **¿Cuál servicio utilizarías en un flujo de BigData si cuentas con Kinesis Data Stream y quisieras alimentar Splunk?**
    
**R//=** Kinesis Firehose

11. **Tú cómo arquitecto tienes un clúster de Apache Kafka instalado en tu datacenter, te han pedido llevarlos a AWS ¿Cuál servicio seleccionarías para esta migración?**
    
**R//=** AWS MSK

12. **Al desplegar un clúster de MSK, por defecto crea un nodo de: ______**
    
**R//=** Zookeeper

13. **¿Cuál es la unidad de procesamiento usada por GLUE?**
    
**R//=** DPU

14. **Te han pedido que implementes una solución para probar los ETL de forma gráfica con seguridad e integración a directorio activo. Para cumplir con este requerimiento ¿Qué herramienta utilizarías?**
    
**R//=** Apache Zeppelin en EMR.

15. **Dentro del servicio de Developer Endpoint de Glue , ¿Qué tipos de notebooks puedes crear?**

**R//=** Zeppelin y Sage Maker. O Solo Zepellin.
    
16. **Tu como arquitecto de soluciones en Big Data, te han pedido que ejecute un ETL para convertir información que viene de la app móvil de la empresa y posteriormente puedan analizarla. ¿Qué pasos deberías seguir dentro del servicio Glue para lograr este objetivo?**
    
**R//=** Crawler - ETL - Crawler.

17. **Estas desplegando un Clúster de EMR y necesitas ejecutar un script de inicio con unas configuraciones específicas para el clúster. ¿Qué deberías utilizar para lograr este objetivo?**
    
**R//=** Bootstrap.

18. **¿Qué tipos de nodos conforman un Clúster de EMR ?**
    
**R//=** Master Nodes, Core Nodes y Task Nodes.

19. **Al intentar conectarte a Zeppelin en EMR ves que no aparece habilitada la interfaz web ¿Cuál podría ser la causa del error?**
    
**R//=** Habilitar el puerto 8890 en el grupo de seguridad para el Master Node.

20. **¿En qué servicio de AWS puedo utilizar notebooks de Apache Zeppelin ?**
    
**R//=** EMR y Glue.

21. **¿Cuáles aspectos debes tener en cuenta usando Lambda en proyectos de Big Data?**
    
**R//=** Concurrencia y Errores.

22. **¿Cuál arquitectura es recomendada para proyectos de Big Data con despliegue de servicios en AWS ?**
    
**R//=** Multi-Account.

23. **¿Qué granularidad agrega Lake formation en los permisos sobre tablas?**
    
**R//=** Permisos a nivel de columnas.

24. **¿De qué tamaño es recomendado un shard en ElasticSearch?**
    
**R//=** 50GB - 150GB.

25. **¿Qué debes tener en cuenta en el momento de dimensionar un clúster de ElasticSearch?**
    
**R//=** Cantidad de Shards distribuidos en los nodos.

26. **E¿Cuál alerta utilizarías para identificar que un usuario ha subido información confidencial de tarjetas de crédito a un bucket de S3?**
    
**R//=** Suspicious o Credentia

27. **¿Cómo se llama la estructura básica dentro de Apache Airflow?**
    
**R//=** DAG.

28. **¿Cuál es la forma de establecer las dependencias dentro de un DAG de Cloud Composer?**
    
**R//=** Utilizando "">>""

29. **Se te ha pedido un reporte sobre diferente información que se tiene en formato CSV en S3, debes realizar esta consulta sobre varios archivos y buckets, encuentras que ya el Glue Catalog esta completamente creado ¿Cuál servicio utilizarías de la manera más eficiente para realizar esta consulta en relación al costo?**
    
**R//=** Athena.

30. **¿Cuál servicio en AWS utilizarías para hacer consultas SQL complejas sobre grandes cantidades de datos en S3?**
    
**R//=** RedShift.