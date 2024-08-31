### Curso de Configuración Profesional de Entorno de Trabajo para Ciencia de Datos

1. **¿Por qué deberías explorar y lidiar con los valores faltantes?**

**R/:** Al encontrar valores faltantes en los datos, es necesario explorarlos y entenderlos para evitar producir sesgos en los resultados e incluso evitar problemas en la creación de modelos.

2. **Responde si la siguiente sentencia es verdadera o falsa:**
**A pesar de que la mejor manera de tratar a los datos que faltan es no tenerlos, esto es pasa con una frecuencia baja. Por lo tanto, deberemos aprender cómo tratarlos apropiadamente.**

**R/:** Verdadera

3. **Responde si la siguiente sentencia es verdadera o false. Ignorar a los valores faltantes no puede producir sesgos en tus análisis.**

**R/:** Falso

4. **A lo largo de tu carrera como científica de datos utilizarás distintas herramientas de software. ¿Qué es algo que siempre debes aprender para cada herramienta al tratar valores faltantes?**

**R/:** Sus operaciones y representaciones de los valores faltantes.

5. **Recolectar datos es una tarea del día a día de una científica de datos. Puedes combinar el poder de interactividad de los jupyter notebooks junto con la versatilidad del uso de la línea de comandos para conseguir hacer este proceso automático en un ambiente familiar. ¿Esto es verdadero o falso?**

**R/:** Verdadero

6. **¿En qué caso podrías hacer uso de extender la API de Pandas?**

**R/:** Todas las respuestas son correctas.

7. **Al trabajar con datos temporales, la pregunta "¿Cuál es mi racha de valores completos y faltantes en una variable?" Puede ser importante por que ___.**

**R/:** Permite encontrar fechas / períodos en los cuales esté acumulando valores faltantes. Lo que permite continuar investigando en esos períodos y entender el por qué de la ausencia de valores.

8. **La siguiente sentencia es verdadera o falsa:**
**Los valores faltantes siempre son representados como NA.**

**R/:** Falsa

9. **¿Cuál de los siguientes valores no puede ser una representación de un valor faltante?**

**R/:** Todos podrían ser representación de valores faltantes. La representación de los valores faltantes siempre dependerá del contexto de trabajo.

10. **¿Qué puedes concluir de la siguiente afirmación? Asumir que los valores faltantes siempre vendrán en un único formato es un error, pero asumir que siempre seremos capaces de detectar la ausencia de valores es un error aún mayor.**

**R/:** Los valores faltantes pueden tomar distintas formas dentro de un conjunto de datos, pero también pueden no estar incluidas de forma directa dentro del espacio observable.

11. **¿Cuál de las siguientes estrategias no corresponde a un método para encontrar valores faltantes implícitos?**

**R/:** 

12. **¿En qué consiste el completar observaciones al exponer combinaciones de n-tuplas o n-variables?**

**R/:** Convierte los valores faltantes implícitos en explicítos a través de completar los datos con combinaciones faltantes de valores provenientes de n-varibles.

13. **`pyjanitor` es una implementación del paquete de R `janitor`, y provee una API para limpieza de datos en Python. ¿Cuál de las siguientes funciones podríamos utilizar para exponer los valores faltantes implícitos?**

**R/:** 

14. **¿Cuál de los siguientes elementos es un mecanismo de acción de los valores faltantes?**

**R/:** Todos son un mecanismo de acción de los valores faltantes.

15. **Llegas a tu trabajo y observas que todos los amigos de Lynn se encuentran ausentes, sin excepción. ¿Qué tipo de mecanismo de valores faltantes podría estar actuando?**

**R/:** Missing Not At Random (MNAR).

16. **¿Cuál no es una característica de la matriz de sombras?**

**R/:** Valores implícitos.

17. **¿Qué es la matriz de sombras?**

**R/:** Es una matriz que tiene las mismas dimensiones que los datos originales, y consiste de indicadores binarios para los valores faltantes, donde los valores faltantes son representados explícitamente.

18. **Una vez que se concatena la matriz de sombras a tu conjunto de datos original, ¿qué se puede realizar con ella?**

**R/:** Ambas opciones son posibles.

19. **Una gráfica de puntos es muy útil para visualizar relaciones entre variables. No obstante, si al menos una de las dos variables tiene valores faltantes, estos puntos no serán dibujados. ¿Cómo podrías solventar este problema para continuar explorando sin agregar ruido a tus datos reales?**

**R/:** Agregar valores dummy a los valores faltantes para visualizarlos en los marginales de la gráfica de puntos y observar su distribución en cada eje.

20. **¿Qué significa cuantificar la correlación de nulidad?**

**R/:** La correlación de nulidad cuantifica qué tan fuerte la presencia o ausencia de una variable afecta la presencia o ausencia de otra.

21. **¿Qué significa que una variable tenga una correlación de nulidad de -1, 0 o 1 con otra variable?**

**R/:** Si la correlación es -1, si una variable aparece la otra seguramente no. Si la correlación es 0, la presencia u ausencia de una variable no afecta la de otra. La correlación de 1 indica que si una variable aparece la otra también lo hará.

22. **¿Qué significa la eliminación de valores faltantes por el método pairwise?**

**R/:** En la eliminación listwise, la eliminación ocurre cuando un procedimiento usa casos que contienen valores faltantes. El procedimiento no incluye una variable particular cuando existe un valor falte, pero puede utilizar otras variable sin valores faltantes.

23. **¿Qué significa la eliminación de valores faltantes por el método listwise?**

**R/:** En la eliminación listwise, un caso se descarta de un análisis porque le falta un valor en al menos una de las variables especificadas. Por lo tanto, el análisis solo se ejecuta en casos que tienen un conjunto completo de datos.

24. **¿Qué es la imputación de valores faltantes?**

**R/:** Estimar los valores ausentes con base a los valores de otras variables o modelos predictivos.

25. **¿Cuál podría ser una desventaja de la imputación de un único valor como la media, mediana o moda?**

**R/:** Puede sesgar los resultados y perder la correlación entre variables.