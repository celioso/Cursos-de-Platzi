# Curso de Data Warehousing y Modelado OLAP

1. **¿Qué es business intelligence?**
   
**R//=** Tomar información del pasado y del presente, limpiarla, transformarla y analizarla para tomar decisiones basadas en datos.

2. **¿Cuál es la jerarquía de los datos?**
 
**R//=** Datos>Información>Conocimiento>Sabiduría

3. **Si se requiere analizar información del pasado para comprender mejor mi negocio a partir del comportamiento de los datos, ¿Cuál tipo de analítica se estaría aplicando?**
   
**R//=** Analítica descriptiva

4. **¿Cuál es la principal diferencia entre un Data Warehouse y un Data Mart??**
   
**R//=** Un Data Warehouse es una base de datos centralizada que almacena datos de toda la organización, mientras que un Data Mart almacena solo un subconjunto específico de datos relevantes para un área de negocio en particular.

5. **¿Qué es una dimensión y cómo se diferencia de un hecho?**
    
**R//=** La Dimensión es la perspectiva por la cual quiero analizar los datos. Por otro lado, un hecho es un valor numérico que se puede analizar, como una cantidad de ventas o un ingreso neto.

6. **Si estás trabajando en un proyecto de transacciones financieras, ¿utilizarías OLTP o OLAP?**
    
**R//=** Utilizaría OLTP, ya que está diseñado para procesar actualizaciones de datos recurrentemente.

7. **Si se está trabajando en un proyecto de análisis de datos de ventas para los últimos 5 años, ¿se debería utilizar OLTP u OLAP?**
    
**R//=** OLAP, ya que está diseñado para analizar grandes volúmenes de datos de manera eficiente. OLTP, por otro lado, está diseñado para procesar transacciones en tiempo real.

8. **Sin un Data Mart no es posible realizar un proceso de inteligencia de negocios. ¿Esto es verdadero o falso?**
    
**R//=** Falso, no es necesario.

9. **¿Cuáles son metodologías de Data Warehousing?**
    
**R//=** Bill Inmon, Ralph Kimball, Hefesto

10. **Si se trata de almacenar grandes volúmenes de datos no estructurados y se desea una mayor flexibilidad para el análisis posterior, ¿Qué solución de almacenamiento de datos sería la más adecuada?**
    
**R//=** Data lake.

11. **¿Que es un cubo en el ámbito de business intelligence?**
    
**R//=** Una representación de un modelo dimensional.

12. **Si se tiene un modelo donde una dimensión principal se relaciona a otra dimensión a un nivel superior, ¿qué tipo de modelo dimensional es?**
    
**R//=** Copo de nieve

13. **Si un equipo de análisis de negocios necesita diseñar una solución de business intelligence para manejar datos que cambian lentamente, ¿qué tipo de dimensión sería la más apropiada si desea mantener la historia en filas?**
    
**R//=** Tipo 2

14. **Si quiero tener información de la fecha y usuario que manipuló un registro en una dimensión, ¿qué tipo de atributos deben usarse?**
    
**R//=** Control

15. **¿Qué tipo de cambios son capturados en una dimensión lentamente cambiante tipo 1 y qué ocurre con los registros históricos?**

**R//=** Los cambios son capturados y sobrescriben directamente los valores antiguos, lo que implica que se pierde la información original.
    
16. **¿Qué tipo de cambios son capturados en una dimensión lentamente cambiante tipo 2 y cómo se manejan los registros históricos?**
    
**R//=** Los cambios se manejan mediante la creación de una nueva fila para cada cambio, lo que implica que se mantienen los registros históricos y se puede ver el historial completo de los cambios.

17. **¿En qué se diferencia una dimensión lentamente cambiante tipo 3 de una dimensión lentamente cambiante tipo 2?**
    
**R//=** Una dimensión lentamente cambiante tipo 3 solo mantiene el registro del valor actual y el valor anterior de un atributo, mientras que una dimensión lentamente cambiante tipo 2 mantiene un historial completo de cambios y agrega una nueva fila para cada cambio con una nueva llave de la dimensión.

18. **¿Cuál es la función principal de una tabla de hechos en un esquema de BI?**
    
**R//=** Almacenar los datos numéricos y métricas de una empresa para su análisis.

19. **Identifica las dimensiones y métricas de la siguiente pregunta de negocio: ¿Cuántos usuarios se registraron para los eventos presenciales en los últimos 3 años, y cuántos de esos efectivamente asistieron?**
    
**R//=** Definiendo una interfaz de listener con un método notify y gestionando los listeners en un manager.

20. **¿Cuál diseño de modelo dimensional permite dar respuesta de la mejor manera las siguientes preguntas de negocio? a. ¿Cuál es el nombre, código y cantidad vendida por línea, talla y color de producto? b. Liste el nombre, dirección y total de ventas ($) por cada tienda. c. ¿Cuáles fueron las ventas en cantidades y total de ventas ($), día a día de los ùltimos 3 años?**
    
**R//=** 
![modelo2](images/modelo2-b0d46010-6057-40f2-9fbe-95b3c7bb5ed2.png)

21. **¿Cuál de las siguientes tablas funcionará como dimensión lentamente cambiante tipo 2 para una dimensión de Empleado?**
    
**R//=** 
```sql
CREATE TABLE Dim_Empleado
(Id_Empleado INTEGER ,
Cod_Empleado VARCHAR(20), 
Nombre_Completo_Empleado VARCHAR(50),
Estado_Civil VARCHAR(50), 
Start_Date TIMESTAMP, 
End_Date TIMESTAMP); 
```

22. **¿Cuál es la forma adecuada de orquestar un flujo para cargar un modelo en el DWH?**
    
**R//=** Primero las dimensiones y luego la tabla de hechos.