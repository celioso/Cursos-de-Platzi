# Curso de AWS Redshift para Manejo de Big Data

1. **¿Qué define mejor a un Data Warehouse?**
   
**R//=** Un repositorio unificado de datos con un propósito analítico.

2. **¿Qué tipo de tablas contiene usualmente un modelo orientado a analítica en un Data Warehouse?**
 
**R//=** Tablas de hechos y tablas de dimensiones.

3. **¿Qué objetivo tiene una base de datos orientada a filas?**
   
**R//=** Procesamiento óptimo de consultas complejas en grandes volúmenes de datos. X

4. **¿Qué objetivo tiene una base de datos orientada a columnas?**
   
**R//=** Procesamiento óptimo de consultas complejas en grandes volúmenes de datos.

5. **¿En qué consiste un cluster?**
    
**R//=** Dos o más nodos interconectados procesando tareas en paralelo.

6. **¿Cuánto espacio ocupa un bloque de datos en Redshift?**
    
**R//=** 1MB

7. **Redshift está basado en la arquitectura de:**
    
**R//=** PostgreSQL

8. **El comando para obtener la compresión recomendada por Redshift a una tabla es:**
    
**R//=** analyze compression

9. **El objetivo de comprimir columnas en Redshift es:**
    
**R//=** Reducir el número de bloques de datos que ocupa, para incrementar la velocidad de respuesta.

10. **Consigue comprimir los datos evaluando la diferencia entre un dato y el siguiente dato en la columna.**
    
**R//=** Delta

11. **Consigue comprimir los datos creando otra entidad en donde a cada valor de la columna se le asigna un índice único.**
    
**R//=** Diccionario de bytes

12. **Consigue comprimir los datos agrupando filas con el mismo dato consecutivo en una columna.**
    
**R//=** Run-length

13. **¿Cuáles son los tipos de distribución en Redshift?**
    
**R//=** all, key, even.

14. **Al aplicar un estilo de distribución en Redshift de busca:**
    
**R//=** Los datos deben estar distribuidos equitativamente y con las columnas "join" en los mismos slices para incrementar la velocidad de respuesta.

15. **La distribución Key distribuye los datos en los nodos basados en una columna especifica:**

**R//=** Verdadero
    
16. **Distribuye los datos en los nodos usando round robin.**
    
**R//=** Even

17. **El objetivo de las llaves de ordenamiento es:**
    
**R//=** Reducir tiempos de respuesta en la consultas.

18. **Una de las desventajas del ordenamiento intercalado (Interleaved) es:**
    
**R//=** La carga de datos es más lenta.

19. **Una de las desventajas del ordenamiento compuesto (Compound) es:**
    
**R//=** La consulta pierde efectividad si se filtran los datos por columnas secundarias de la llave.

20. **Los bloques de datos en Redshift guardan el valor mínimo y máximo que contienen, por esta razón usar llaves de ordenamiento permite descartar rápidamente bloques de datos fuera de cláusula WHERE indicada e incrementar la velocidad de respuesta:**
    
**R//=** Verdadero

21. **¿Qué función cumple el parámetro COMPUPDATE en el copy?**
    
**R//=** Crea automáticamente la compresión de columnas recomendada por Redshift a una tabla previamente vacía.

22. **El comando copy realizar el cargue en la tabla usando procesamiento masivo en paralelo (MPP), por esta razón es muy efectivo.**
    
**R//=** Verdadero

23. **Con este parámetro puedes indicar cual es el caracter de separación en tu archivo a cargar en un copy.**
    
**R//=** delimiter

24. **¿Qué se logra con el comando Analyze?**
    
**R//=** Actualizar la metadata estadística buscando que el planificador de consultar conozca cómo resolver estas consultas de la mejor forma.

25. **El comando Vacuum no indispone la tabla para actualización de datos.**
    
**R//=** Falso

26. **¿Qué puedo conseguir aplicando Vacuum a una tabla?**
    
**R//=** Recuperar espacio, ordenar correctamente las columnas en una tabla luego de un cargue.

27. **¿Qué función tiene "explain" en Redshift?**
    
**R//=** Retorna el plan de ejecución y costo de una consulta.

28. **El comando explain retorna el costo de una consulta de la siguiente manera valor1..valor2 a modo de ejemplo 122..234432, ¿qué significan estos valores respectivamente?**
    
**R//=** 
- **valor1:** costo de retornar la primer fila.
- **valor2:** costo de retornar todas las filas.

29. **El comando unload es igual de óptimo que resolver una consulta en el editor SQL, con la única diferencia que guarda los datos en S3.**
    
**R//=** Falso

30. **La siguiente instrucción ejecuta una consulta y envía los resultados a S3 en archivos de 100 mb, separador (;), compresión BZIP2 y encabezado.**
    
**R//=** 
```sql
unload ('select * from unload_test') to 's3://loadfilesredshift/unload/unload_test_other_' 
credentials 'aws_iam_role=aws_arn' 
delimiter ';' 
HEADER 
BZIP2 
MAXFILESIZE 100 mb
```