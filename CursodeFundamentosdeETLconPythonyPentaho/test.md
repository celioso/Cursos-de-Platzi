# Curso de Fundamentos de ETL con Python y Pentaho

1. **¿Qué significa ETL en ingeniería de datos?**
   
**R//=** ETL es un acrónimo que significa "Extracción, Transformación y Carga", un proceso utilizado para integrar datos de múltiples fuentes en un solo destino.

2. **¿Cuál es la importancia de ETL en ingeniería de datos?**
 
**R//=** ETL es importante en ingeniería de datos porque permite a los profesionales de datos integrar y consolidar datos de múltiples fuentes, lo que mejora la calidad y la eficacia de los análisis y toma de decisiones.

3. **¿Cuál es la diferencia entre el source y el target en una ETL?**
   
**R//=**El source se refiere a la fuente de datos donde se extraen los datos para la transformación y carga en el target, que es el destino final de los datos.

4. **¿Cuál es la diferencia entre un data warehouse y un data lake en términos de ETL?**
   
**R//=** En un data warehouse el proceso de ETL se centra en la integración y transformación de datos estructurados y almacenados en diferentes sistemas, para crear un conjunto de datos coherente y consolidado. En un data lake, el proceso de ETL se enfoca en la ingestión y almacenamiento de datos en su forma más cruda, sin aplicar transformaciones significativas hasta que se requieran para un análisis específico.

5. **¿Siempre es mejor que una ETL se realice en streaming y no en procesos batch?**
    
**R//=** Falso, depende de la misma naturaleza y necesidades del proyecto.

6. **Un ETL netamente desarrollada desde cero en Python u otro lenguaje, ¿de qué tipo se puede considerar?**
    
**R//=** Custom

7. **Es algo a tener en cuenta al momento de usar sources en un proceso de ETL.**
    
**R//=** Todas las opciones se deben considerar.

8. **¿Cómo afecta la frecuencia de extracción de las fuentes en una ETL?**
    
**R//=** Si la frecuencia es muy baja, es posible que se pierdan datos recientes, mientras que una frecuencia demasiado alta puede causar una sobrecarga en el sistema y afectar el rendimiento.

9. **La extracción de datos en Python solo la debo manejar con la librería de Pandas. ¿Esto es verdadero o falso?**
    
**R//=** Falso, si bien es una librería perfecta para la manipulación de datos existen otras librerías que podemos usar.

10. **¿Cuál es la mejor estrategia para manejar duplicados en una ETL?**
    
**R//=** La mejor estrategia para manejar duplicados en una ETL es utilizar una combinación de técnicas como la eliminación de duplicados, la unificación de registros y la consolidación de datos.

11. **¿Qué hace esta línea de código de Pandas?**
```python
df_codes[['clean_code','parent_description']] = df_codes.apply(lambda x : clean_code(x['Code']),axis=1, result_type='expand')
```
    
**R//=** Asigna a dos columnas de un DataFrame (df_codes) los resultados de aplicar una función lambda (clean_code) a la columna "Code" de dicho DataFrame, para cada fila.

12. **¿Qué hace esta línea de código de Pandas en el DataFrame?**

```python
df_countries = df_countries[df_countries['alpha-3'].notnull()]
```
    
**R//=** Filtra el DataFrame 'df_countries' para eliminar todas las filas donde la columna 'alpha-3' tiene un valor nulo.

13. **¿Cuál es la razón de crear esta función en Python para una transformación de datos?**

```python
defcreate_dimension(data, id_name):
    list_keys = []
    value = 1
    for _ in data:
        list_keys.append(value)
        value = value + 1
    return pd.DataFrame({id_name:list_keys, 'values':data})
```
    
**R//=** Una manera eficiente de crear un DataFrame con valores únicos de posibles dimensiones o valores categóricos.

14. **¿¿Cuál es la importancia del formato de los datos en el proceso de carga en una ETL?**
    
**R//=** El formato de los datos es esencial en el proceso de carga de una ETL, ya que determina cómo se pueden manipular y transformar los datos durante la fase de transformación.

15. **¿Cuál es la librería en Python para gestionar el uso de AWS?**

**R//=** boto3
    
16. **Desde Python únicamente se puede gestionar carga a data warehouses de AWS como Redshift. ¿Esto es verdadero o falso?**
    
**R//=** Falso, puede ser a cualquier tipo de data warehouse, sea on-premise o cloud.

17. **¿Qué herramienta de Pentaho debe usarse para leer datos de una tabla en una base de datos?**
    
**R//=** Input Table

18. **¿Cuál es el propósito del paso Select values en Pentaho PDI?**
    
**R//=** Se utiliza para seleccionar y renombrar columnas específicas de un conjunto de datos.

19. **¿Cuál es el propósito del paso Filter rows en Pentaho PDI?**
    
**R//=** Filtrar filas específicas de un conjunto de datos en función de una o más condiciones.

20. **¿En Pentaho solo puedo hacer un cargue de datos a bases de datos relacionales?**
    
**R//=** No, hay múltiples target no relacionales a los que puedo cargar.