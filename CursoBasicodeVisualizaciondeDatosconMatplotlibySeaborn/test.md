# Curso Basico de Visualizacion de Datos con Matplotlib y Seaborn

1. **¿Qué nos permite encontrar la visualización de datos?**
   
**R//=** Todas las opciones son correctas.

2. **Sentencia de Matplotlib para crear un lineplot de x, y con puntos azules:**
 
**R//=** `plt.plot(x, y,'bo')`

3. **¿Con la sentencia plt.subplot(1,2,1) de Matplotlib a qué estoy accediendo?**
   
**R//=** El primer plot de una fila y dos columnas.

4. **¿El método orientado a objetos de Matplotlib es mejor que el método de pyplot?**
   
**R//=** Depende de la necesidad y su uso.

5. **¿En la sentencia de Matplotlib "fig, axes = plt.subplots(nrows=1, ncols=2)" axes de qué tipo se le asigna un objeto?**
    
**R//=** En la sentencia de Matplotlib fig, axes = plt.subplots(nrows=1, ncols=2), la variable axes se le asigna un objeto del tipo array de ejes (de tipo numpy.ndarray) que contiene instancias de AxesSubplot.

6. **¿Para qué sirve el uso de leyendas, etiquetas y títulos en nuestras visualizaciones de datos?**
    
**R//=** Más contexto a nuestros gráficos.

7. **¿Cuál es la propiedad para definir el estilo de la línea en un plot de Matplotlib?**
    
**R//=** linestyle

8. **¿En Matplotlib a diferencia de plt.bar, para qué funciona plt.barh?**
    
**R//=** El mismo gráfico que plt.bar, pero horizontal.

9. **¿Cuál comando de Matplotlib se usa para crear un histograma?**
    
**R//=** `plt.hist`

10. **Este gráfico es muy útil para ver la correlación entre dos variables:**
    
**R//=** scatter

11. **Es posible cambiar la paleta de colores, estilo, fuente y otros parámetros con ese comando de seaborn**
    
**R//=** `sns.set`

12. **¿En el parámetro data de Seaborn que se debe poner?**
    
**R//=** La fuente de datos.

13. **Con este comando de Seaborn se pueden crear distintos tipos de gráficos orientados a distribución**
    
**R//=** `sns.displot`

14. **¿Es posible mezclar dos tipos de gráficos como swarmplot y boxplot en una gráfica de Seaborn?**
    
**R//=** Verdadero

15. **Con este comando de Seaborn se pueden crear distintos tipos de datos orientados a gráficos de relación:**
    
**R//=** `sns.relplot`

16. **Con este parámetro en joinplot de Seaborn se pueden modificar los valores de las gráficas secundarias de los ejes:**
    
**R//=** marginal_kws

17. **¿Con cuál comando de Seaborn se pueden ver múltiples relaciones en todo un dataset?**
    
**R//=** `sns.pairplot`

18. **Este tipo de gráfica muy utilizado en Seaborn es útil para visualizar estructuras matriciales:**
    
**R//=** heatmap