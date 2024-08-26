### Curso de Python para Ciencia de Datos

1. **Si tienes un DataFrame de Pandas con datos de ventas y necesitas calcular el total de ventas por cliente, ¿qué método utilizarías?**

**R/:** groupby()

2. **Para preprocesar datos que contienen valores nulos antes de un análisis, ¿qué función de Pandas te permite rellenar esos valores de manera eficiente?**

**R/:** fillna()

3. **¿Qué estructura de NumPy utilizarías para representar una imagen en color con altura, ancho y tres canales de color?**

**R/:** Un tensor

4. **¿Cómo crearías un array en NumPy con los números del 1 al 15?**

**R/:** np.arange(1, 16)

5. **¿Cuál es la salida de np.dot(A, B) si A y B son matrices cuadradas de 2x2 en NumPy?**

**R/:** fUna matriz de 2x2

6. **Si necesitas acceder al último elemento de un array desde el final, ¿qué técnica de indexación utilizarías?**

**R/:** Indexación negativa con -1

7. **¿Qué ventaja ofrece el broadcasting en NumPy al realizar operaciones entre arrays de diferentes dimensiones?**

**R/:** Evita la duplicación de datos y reduce el uso de memoria.

8. **¿Qué operación aplicarías en NumPy para convertir un array multidimensional en un array unidimensional?**

**R/:** flatten()

9. **¿Cuál es el método adecuado para leer un archivo Excel en un DataFrame de Pandas?**

**R/:** pd.read_excel("file.xlsx")

10. **¿Qué técnica podrías usar para manejar valores faltantes en un DataFrame sin eliminar información?**

**R/:** Imputación con la media o mediana

11. **¿Qué método utilizarías para obtener un resumen estadístico completo de un DataFrame?**

**R/:** describe()

12. **¿Cómo seleccionarías las primeras 5 filas y las primeras 3 columnas de un DataFrame?**

**R/:** df.iloc[:5, :3]

13. **¿Cómo crearías una nueva columna 'Profit' como la diferencia entre 'TotalPrice' y 'Cost' en un DataFrame?**

**R/:** df['Profit'] = df['TotalPrice'] - df['Cost']

14. **¿Qué método permite aplicar una función personalizada a cada grupo después de agrupar los datos?**

**R/:** apply()

15. **Si deseas calcular la suma total y la cantidad vendida por país y código de producto, ¿cuál función de Pandas usarías?**

**R/:** pivot_table con aggfunc=“sum”

16. **¿Qué tipo de unión devolvería todas las filas del DataFrame izquierdo y solo las filas coincidentes del DataFrame derecho?**

**R/:** Left Join

17. **¿Cuál es la función de Pandas que permite cambiar la frecuencia de los datos temporales?**

**R/:** resample()

18. **¿Qué resultado obtienes al concatenar dos DataFrames verticalmente sin especificar el parámetro axis en concat()?**

**R/:** Las filas de ambos DataFrames se apilan una sobre otra.

19. **Si quisieras identificar los 5 productos más vendidos, ¿cómo ajustarías el código para mostrar solo esos productos?**

**R/:** Utilizar head(5) después de ordenar por ‘Quantity’.

20. **¿Qué función usarías para guardar un array de NumPy en un archivo binario?**

**R/:** np.save()

21. **¿Cómo seleccionarías la tercera fila y segunda columna de un DataFrame usando iloc?**

**R/:** df.iloc[2, 1]