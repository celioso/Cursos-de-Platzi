# Curso de ETL e ingesta de datos con Python

1. **Si deseas visualizar la distribución de una variable numérica en un conjunto de datos, ¿qué tipo de gráfico sería más adecuado utilizar?**
   
**R//=** Histograma

2. **Al recibir una respuesta en formato JSON de una API, ¿cómo podrías convertirla en un formato más legible para su análisis?**
 
**R//=** Usando la librería pandas para convertirla en un DataFrame.

3. **¿Qué métrica de calidad de datos evaluarías primero para asegurar que un dataset es útil para análisis??**
   
**R//=** Completitud, para verificar que no falten datos críticos.

4. **Para conectarte a una base de datos SQL Lite en Python, ¿qué librería es más adecuada para simplificar el proceso?**
   
**R//=** SQL Alchemy

5. **Si necesitas acceder a un rango de valores en una Panda Series, ¿qué método de indexación deberías usar?**
    
**R//=** Usar el índice personalizado que configuraste

6. **Al convertir una variable categórica como 'género' en una representación numérica, ¿qué técnica podrías usar para asegurar que no se pierda información en un caso de género no binario?**
    
**R//=** Utilizar la técnica de 'get_dummies' para crear variables dummy.

7. **Si tienes una columna de 'edad' con valores nulos y deseas convertirla a un tipo numérico sin errores, ¿qué método de pandas podrías usar?**
    
**R//=** Utilizar 'pd.to_numeric' con el parámetro 'errors=coerce'.

8. **Si deseas calcular el valor máximo y mínimo de una columna al mismo tiempo, ¿qué función de Python deberías utilizar?**
    
**R//=** aggregate

9. **Si deseas calcular el salario anual de los empleados a partir de su salario mensual, ¿qué método de pandas deberías usar para aplicar una función personalizada?**
    
**R//=** apply

10. **Para combinar los datos de empleados y bonificaciones usando el ID de empleado como clave, ¿qué método de pandas es más adecuado?**
    
**R//=** merge

11. **¿Cuál de los siguientes parámetros de la función to_csv permite especificar el delimitador entre los valores de un archivo CSV?**
    
**R//=** Usar el parámetro sep con el valor ,

12. **Al particionar un archivo CSV por año, ¿qué resultado esperarías al visualizar el archivo 'Data 2023.csv'?**
    
**R//=** Solo registros del año 2023

13. **Si una empresa de e-commerce quiere actualizar su base de datos con nuevos registros cada día, ¿qué tipo de carga de datos debería considerar en su proceso ETL?**
    
**R//=** Carga incremental

14. **Si deseas realizar una carga incremental de datos en un archivo Excel existente, ¿qué paso es crucial antes de ejecutar el código?**
    
**R//=** Cerrar el archivo Excel si está abierto

15. **Si necesitas importar un esquema de base de datos en MySQL Workbench, ¿qué opción deberías seleccionar?**

**R//=** Data import
    
16. **Si deseas conectar Python con MySQL, ¿qué librería es esencial instalar?**
    
**R//=** mysql-connector-python

17. **Si necesitas filtrar actores cuyo primer nombre comience con 'A' en Python, ¿qué método deberías usar?**
    
**R//=** startswith

18. **Si deseas convertir los nombres a mayúsculas en Python, ¿qué función deberías aplicar?**
    
**R//=** upper

19. **Al documentar un proceso ETL, ¿qué elemento es crucial para asegurar una correcta gestión de excepciones?**
    
**R//=** Describir claramente el manejo de errores y las excepciones.

20. **Si tienes una versión de Python anterior a la 3.5, ¿qué problema podrías enfrentar al usar la librería json?**
    
**R//=** Problemas de compatibilidad