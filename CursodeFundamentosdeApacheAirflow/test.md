# Curso de Fundamentos de Apache Airflow

1. **¿Cuál de estás son caraterísticas de Apache Airflow?**
   
**R//=** Todas las respuestas son correctas

2. **Un DAG tiene las siguiente propiedadades, es un grafo cíclico y dirigido. ¿Es correcto?**
 
**R//=** FALSE

3. **Un DAG está compuesto por tareas.**
   
**R//=** TRUE

4. **Un operador nos permite:**
   
**R//=** Definir qué van a hacer nuestras tareas

5. **La variable de configuración AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL nos permite:**
    
**R//=** Definir la frecuencia (en segundos) con la que se escanea el directorio DAGs en busca de nuevos archivos. Por defecto, 5 minutos.

6. **Una de las maneras en la que podemos crear variables y conexiones en Apache Airflow es utilizando la interfaz gráfica, pero hay otras alternativas**
    
**R//=** TRUE

7. **¿Cuál de estás es una sintaxis para declarar un DAG?**
    
**R//=** Todas las opciones son válidas

8. **¿El parámetro task_id es obligatorio para utilizar un bash operator?**
    
**R//=** TRUE

9. **¿Para qué sirve el parámetro python_callable de un PythonOperator?**
    
**R//=** Para pasar la función que nos interesa ejecutar

10. **¿Cómo podemos definir dependencias entre tareas? Utilizando:**
    
**R//=** >>, <<, set_downstream(), set_upstream()

11. **Para crear un custom operator nuestra clase tiene que heredar de la clase:**
    
**R//=** BaseOperator

12. **¿Qué significa definir un schedule_interval @hourly?**
    
**R//=** Nuestro DAG se va a ejecutar cada hora

13. **¿Qué significa ejecutar un proceso usando la siguiente sintaxis? 0 7 * * 1**
    
**R//=** Cada lunes a las 7 AM nuestro proceso se va a ejecutar

14. **Las tareas en Apache Airflow sólo tienen 3 posibles estados, RUNNING, FAILED y SUCCESS**
    
**R//=** FALSE

15. **El task action CLEAR nos permite limpiar el estado actual de una tarea**

**R//=** TRUE
    
16. **El trigger rule ALL_DONE nos permite:**
    
**R//=** Ejecutar una tarea cuando las tareas anteriores hayan finalizado sin importar el estado final

17. **Los sensores son un tipo de operador**
    
**R//=** TRUE

18. **Los ExternalTaskSensor nos permiten:**
    
**R//=** Esperar por el estado de una tarea que se encuentra en otro DAG

19. **Un file sensor nos permite esperar por la creación de un fichero específico**
    
**R//=** TRUE

20. **Si utilizamos la variable {{ ds }} podemos utilizar:**
    
**R//=** El logical date de nuestro DAG

21. **
Los XComs nos permiten que las tareas se comuniquen entre sí?**
    
**R//=** TRUE

22. **Un BranchPythonOperator nos permite:**
    
**R//=** Establecer una condición para que cuando se cumpla dicha condición se ejecute una parte de nuestro flujo u otra

23. **¿Dónde tenemos que guardar nuestros DAGs para que el scheduler sea capaz de leerlos?**
    
**R//=** En la carpeta /dags

24. **Cuando una tarea está de color gris significa que:**
    
**R//=** La tarea está en queued

25. **Cuando una tare está de color verde lima significa que:**
    
**R//=** La tarea está en running

26. **Las tareas por defecto tienen el parámetro depend_on_past en:**
    
**R//=** FALSE

27. **El parámetro max_active_runs sirve para::**
    
**R//=** Definir un máximo de instancias concurrentes de nuestro DAG se pueden ejecutar

28. **Airflow se puede utilizar para:**
    
**R//=** Todas las opciones son válidas