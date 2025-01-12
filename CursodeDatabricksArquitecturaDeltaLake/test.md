# Curso de Databricks: Arquitectura Delta Lake

1. **¿Cuál es el propósito principal de Databricks en el ecosistema de Apache Spark?**
   
**R//=** Visualizar y analizar datos de manera colaborativa.

2. **¿Cuál es la ventaja principal de utilizar Databricks para la ejecución de trabajos Spark?**
 
**R//=** Facilita la colaboración entre equipos de datos y ciencia.

3. **¿Qué tipo de entorno de programación ofrece Databricks para trabajar con Apache Spark?**
   
**R//=** Un entorno de cuadernos colaborativos que admite múltiples lenguajes como Python, Scala y SQL.

4. **¿Qué significa la sigla "DBFS" en Databricks?**
   
**R//=** Databricks File System

5. **¿Cuál de las siguientes operaciones en Spark se clasifica como una transformación?**
    
**R//=** filter()

6. **¿Qué acción se utiliza para contar el número de elementos en un RDD o DataFrame en Spark?¿Cuánto espacio ocupa un bloque de datos en Redshift?**
    
**R//=** count()

7. **¿Cuál de las siguientes es una transformación en Spark?**
    
**R//=** map()

8. **¿Qué acción se utiliza para obtener el primer elemento de un RDD o DataFrame en Spark?**
    
**R//=** first()

9. **¿Cuál de las siguientes operaciones en Spark se clasifica como una acción?**
    
**R//=** collect()

10. **¿Qué es Spark UI?**
    
**R//=** Una interfaz de usuario gráfica para monitorear y depurar aplicaciones Spark.

11. **¿Cuál es el propósito principal de Spark UI?**
    
**R//=** Monitorear y analizar el rendimiento de las aplicaciones Spark.

12. **¿Cuál es una de las funciones principales de Spark UI durante la ejecución de una aplicación en Spark?**
    
**R//=** Monitorear el rendimiento de la aplicación y visualizar el plan de ejecución.

13. **¿Qué es Maven en el contexto de Apache Spark?**
    
**R//=** Un sistema de construcción y gestión de dependencias y librerias utilizado para construir proyectos Spark.

14. **¿Cuál de las siguientes afirmaciones describe correctamente Spark SQL?**
    
**R//=** Spark SQL proporciona una interfaz para consultar datos estructurados utilizando SQL y también permite consultas en RDDs de Spark.

15. **¿Cuál de las siguientes afirmaciones es correcta sobre los DataFrames en Spark SQL?**

**R//=** Los DataFrames en Spark SQL son una abstracción distribuida que permite manipular datos estructurados con API en diversos lenguajes.
    
16. **¿Qué significa la sigla "UDF" en el contexto de Apache Spark?**
    
**R//=** User-Defined Function

17. **¿Cuál es el propósito principal de una User-Defined Function (UDF) en Spark?**
    
**R//=** Filtrar datos en un DataFrame según un criterio personalizado

18. **En Spark, ¿qué tipo de datos puede procesar una UDF?**
    
**R//=** Datos de cualquier tipo

19. **¿Cuál es el propósito de "Medallion" en Spark?**
    
**R//=** Facilitar la gestión de recursos y escalabilidad

20. **¿Cuál es el propósito principal de la implementación de un "Delta Lake" en Apache Spark?**
    
**R//=** Gestionar de manera confiable y transaccional los datos en entornos big data.

21. **Estás trabajando en un proyecto de análisis de datos y necesitas una plataforma que permita procesar grandes volúmenes de información de manera eficiente. ¿Cuál de las siguientes opciones sería la más adecuada?**
    
**R//=** Una plataforma de análisis en la nube como Databricks.

22. **Tu empresa está considerando migrar sus flujos de trabajo de análisis de datos a la nube y está evaluando diferentes opciones. ¿Qué ventajas podría ofrecer Databricks en este escenario?**
    
**R//=** Integración con servicios en la nube como AWS, GCP y Azure.

23. **Tu equipo de análisis de datos necesita realizar consultas SQL sobre grandes conjuntos de datos distribuidos. ¿Qué herramienta sería la más apropiada para este escenario?**
    
**R//=** Apache Spark SQL.

24. **Tu equipo necesita ejecutar procesos ETL (Extract, Transform, Load) de manera regular para preparar los datos antes de su análisis. ¿Qué componente de Spark sería el más adecuado para automatizar este proceso?**
    
**R//=** Spark DataFrames.

25. **Estás trabajando en un proyecto que requiere procesamiento en tiempo real de datos de sensores. ¿Qué componente de Spark sería más útil para este caso?**
    
**R//=** Spark Streaming.

26. **Estás trabajando en un proyecto de análisis de datos en Apache Spark y necesitas realizar operaciones de transformación sobre un conjunto de datos distribuido. ¿Cuál de las siguientes opciones describe mejor qué es un RDD en este contexto?**
    
**R//=** Un RDD es un conjunto de datos distribuido e inmutable que permite realizar operaciones de transformación en paralelo en un clúster de Spark.

27. **Estás desarrollando un flujo de trabajo en Spark y necesitas entender la diferencia entre transformaciones y acciones para optimizar el procesamiento de datos. ¿Cuál de las siguientes afirmaciones describe correctamente la diferencia entre transformaciones y acciones en Spark?**
    
**R//=** Las transformaciones son operaciones perezosas que definen un nuevo conjunto de datos a partir de uno existente, mientras que las acciones son operaciones que desencadenan la ejecución real de las transformaciones y devuelven resultados al programa del conductor.

28. **Tu empresa está considerando migrar sus flujos de trabajo de análisis de datos a la nube y está evaluando diferentes opciones. ¿Cuáles podrían ser las ventajas y desventajas de utilizar un entorno de Spark local en comparación con uno en la nube como Databricks?**
    
**R//=** Ventajas de Databricks: Integración con servicios en la nube como AWS y Azure, mayor escalabilidad y facilidad de gestión. Desventajas: Menor control sobre la infraestructura subyacente y posibles costos adicionales asociados con el uso de servicios en la nube.

29. **Estás trabajando en un proyecto de análisis de datos en Spark y necesitas cargar un archivo CSV que contiene información sobre ventas. ¿Cuál de las siguientes opciones sería la forma adecuada de cargar este archivo en un DataFrame de Spark?**
    
**R//=** Utilizar la función spark.read.csv() para cargar el archivo CSV directamente en un DataFrame.