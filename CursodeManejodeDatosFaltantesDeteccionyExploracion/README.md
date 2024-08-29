# Curso de Manejo de Datos Faltantes Detección y Exploración

## Operaciones con valores faltantes

En el manejo de datos, es común encontrarse con valores faltantes (o `NaN` en Pandas y NumPy). Estos valores pueden causar problemas en los análisis si no se manejan adecuadamente. A continuación, te muestro algunas operaciones comunes para trabajar con valores faltantes utilizando **Pandas**.

### 1. **Detectar valores faltantes**

Para identificar los valores faltantes en un `DataFrame` o `Series`, se pueden usar los métodos `isnull()` o `isna()`, que devuelven un DataFrame de booleanos.

```python
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4],
    'C': [1, None, None, 4]
})

# Detectar valores faltantes
df.isnull()
```

Salida:
```plaintext
       A      B      C
0  False   True  False
1  False  False   True
2   True  False   True
3  False  False  False
```

### 2. **Contar valores faltantes**

Puedes contar cuántos valores faltantes hay en cada columna utilizando `isnull().sum()`.

```python
# Contar valores faltantes por columna
df.isnull().sum()
```

Salida:
```plaintext
A    1
B    1
C    2
dtype: int64
```

### 3. **Eliminar valores faltantes**

Para eliminar filas o columnas con valores faltantes, se utiliza `dropna()`.

- **Eliminar filas con valores faltantes**:
  ```python
  df_sin_nan = df.dropna()
  ```

  Esto eliminará cualquier fila que contenga al menos un valor faltante.

- **Eliminar columnas con valores faltantes**:
  ```python
  df_sin_nan_columnas = df.dropna(axis=1)
  ```

  Esto eliminará cualquier columna que tenga al menos un valor faltante.

### 4. **Rellenar valores faltantes**

En lugar de eliminar los valores faltantes, puedes optar por rellenarlos con algún valor. Esto se puede hacer con el método `fillna()`.

- **Rellenar con un valor constante**:
  
  ```python
  df_relleno = df.fillna(0)
  ```

  Aquí, todos los valores `NaN` se reemplazan por `0`.

- **Rellenar con la media, mediana o moda**:
  
  - Media:
    ```python
    df['A'] = df['A'].fillna(df['A'].mean())
    ```
  
  - Mediana:
    ```python
    df['B'] = df['B'].fillna(df['B'].median())
    ```
  
  - Moda:
    ```python
    df['C'] = df['C'].fillna(df['C'].mode()[0])
    ```

- **Rellenar con el valor anterior o siguiente (forward fill / backward fill)**:
  
  - Rellenar con el valor anterior:
    ```python
    df_forward_fill = df.fillna(method='ffill')
    ```
  
  - Rellenar con el valor siguiente:
    ```python
    df_backward_fill = df.fillna(method='bfill')
    ```

### 5. **Interpolar valores faltantes**

Para valores numéricos, puedes usar la interpolación para estimar los valores faltantes basándote en los valores vecinos.

```python
# Interpolación lineal
df_interpolado = df.interpolate()
```

### 6. **Reemplazar valores específicos**

Si deseas reemplazar un valor específico (como `None`, `NaN`, o un número), puedes usar el método `replace()`.

```python
df_reemplazo = df.replace({None: 0})
```

### 7. **Verificar si hay valores faltantes**

Para saber si hay algún valor faltante en el `DataFrame`, puedes usar `isnull().values.any()`.

```python
# Verificar si hay valores faltantes
df.isnull().values.any()
```

### Ejemplo completo:

```python
import pandas as pd

# Crear un DataFrame con valores faltantes
df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4],
    'C': [1, None, None, 4]
})

# Detectar valores faltantes
print("Valores faltantes:")
print(df.isnull())

# Contar valores faltantes por columna
print("\nConteo de valores faltantes:")
print(df.isnull().sum())

# Eliminar filas con valores faltantes
print("\nDataFrame sin filas con valores faltantes:")
print(df.dropna())

# Rellenar valores faltantes con un valor constante
print("\nRellenar valores faltantes con 0:")
print(df.fillna(0))

# Rellenar con la media de la columna
print("\nRellenar la columna 'A' con la media:")
df['A'] = df['A'].fillna(df['A'].mean())
print(df)

# Interpolación para rellenar valores faltantes
print("\nInterpolación de valores faltantes:")
print(df.interpolate())
```

### Conclusión

Pandas ofrece herramientas muy flexibles para detectar, eliminar, rellenar, y reemplazar valores faltantes en tus datos. Dependiendo del análisis que quieras realizar, podrás elegir la mejor técnica para manejar estos valores.

**Lecturas recomendadas**

[Deepnote](https://deepnote.com/workspace/platzi-escuela-datos-83832097-f136-43ff-b38d-abaa022e8ec7/project/datos-faltantes-694a3d08-7f18-421d-9e2f-c2820a79680e "Deepnote")

## Conociendo datasets para manejo de datos faltantes

```python
https://nrvis.com/data/mldata/pima-indians-diabetes.csv

names=[
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree_function",
        "age",
        "outcome",
    ]
```

Para practicar y aprender cómo manejar datos faltantes, puedes utilizar algunos conjuntos de datos (datasets) populares que tienen valores faltantes de manera intencional o realista. Estos datasets se utilizan frecuentemente para entrenar habilidades de preprocesamiento en ciencia de datos.

### 1. **Titanic (Kaggle)**
El famoso conjunto de datos del Titanic incluye información sobre los pasajeros del barco, con características como el nombre, edad, sexo, clase, y si sobrevivieron o no. Algunos campos, como la edad y el puerto de embarque, tienen valores faltantes.

- **Descarga**: [Titanic Dataset - Kaggle](https://www.kaggle.com/c/titanic)
  
  - **Valores faltantes**: `Age`, `Cabin`, `Embarked`.

  ```python
  import pandas as pd
  titanic = pd.read_csv('titanic.csv')
  print(titanic.isnull().sum())
  ```

### 2. **Housing Prices (Kaggle)**
Este dataset proviene de una competencia de Kaggle sobre predicción de precios de viviendas. Contiene información sobre características de las casas, como el tamaño, el número de habitaciones, el año de construcción, entre otros, y tiene valores faltantes en varias columnas.

- **Descarga**: [House Prices - Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

  - **Valores faltantes**: `LotFrontage`, `GarageType`, `GarageYrBlt`, etc.

  ```python
  housing = pd.read_csv('house_prices.csv')
  print(housing.isnull().sum())
  ```

### 3. **Air Quality (UCI Machine Learning Repository)**
Este dataset contiene información sobre la calidad del aire en Milán, Italia, y tiene muchos valores faltantes debido a fallas en los dispositivos de monitoreo. Se utiliza comúnmente para trabajar con interpolación de datos faltantes y análisis de series temporales.

- **Descarga**: [Air Quality Dataset - UCI](https://archive.ics.uci.edu/ml/datasets/Air+quality)

  - **Valores faltantes**: Muchos valores faltantes debido a errores en la captura de datos.

  ```python
  air_quality = pd.read_csv('AirQualityUCI.csv', sep=';')
  print(air_quality.isnull().sum())
  ```

### 4. **Diabetes Dataset (Pima Indians)**
Este dataset contiene información sobre pacientes con antecedentes de diabetes y varias características médicas, como el nivel de glucosa en sangre y la presión arterial. Algunos valores son claramente incorrectos o están ausentes, por lo que se utiliza para preprocesar datos médicos.

- **Descarga**: [Diabetes Dataset - Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

  - **Valores faltantes**: Algunos valores de glucosa, presión arterial y otros son 0, lo cual es un valor incorrecto y representa valores faltantes.

  ```python
  diabetes = pd.read_csv('diabetes.csv')
  diabetes.replace(0, np.nan, inplace=True)
  print(diabetes.isnull().sum())
  ```

### 5. **World Happiness Report**
Este conjunto de datos contiene indicadores de felicidad mundial, medidos por varios factores como el PIB per cápita, la esperanza de vida, el apoyo social y la percepción de corrupción. Algunos de estos indicadores pueden estar faltantes en algunos países.

- **Descarga**: [World Happiness Report - Kaggle](https://www.kaggle.com/unsdsn/world-happiness)

  - **Valores faltantes**: Algunos datos para países están ausentes en columnas como `Social Support`, `Generosity`, etc.

  ```python
  happiness = pd.read_csv('world_happiness.csv')
  print(happiness.isnull().sum())
  ```

### 6. **Medical Cost Personal Dataset**
Este conjunto de datos contiene información sobre los costos médicos personales según varios factores como la edad, el sexo, el IMC, el número de hijos, el hábito de fumar y la región. Algunos de estos campos pueden contener valores faltantes o no válidos.

- **Descarga**: [Medical Cost Personal Dataset - Kaggle](https://www.kaggle.com/mirichoi0218/insurance)

  - **Valores faltantes**: Aunque originalmente no tiene valores faltantes, se puede modificar para introducir valores `NaN` para prácticas de manejo de datos faltantes.

  ```python
  medical = pd.read_csv('insurance.csv')
  medical['bmi'] = medical['bmi'].mask(medical['bmi'] < 18)
  print(medical.isnull().sum())
  ```

### 7. **Adult Dataset (Income Prediction)**
Este dataset contiene información sobre los ingresos de adultos, junto con detalles como edad, ocupación, estado civil, etc. Tiene varios valores faltantes en categorías como la ocupación y el área de trabajo.

- **Descarga**: [Adult Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)

  - **Valores faltantes**: `Workclass`, `Occupation`, `Native-country`.

  ```python
  adult = pd.read_csv('adult.data', header=None)
  print(adult.isnull().sum())
  ```

---

### Técnicas para Manejar Datos Faltantes

Con estos datasets, puedes aplicar técnicas de manejo de valores faltantes como:

- **Eliminar filas o columnas con `dropna()`**.
- **Rellenar con un valor como `fillna()`**.
- **Interpolar valores faltantes usando `interpolate()`**.
- **Reemplazar valores faltantes con medidas estadísticas (media, mediana, moda)**.
  
Estas prácticas te ayudarán a gestionar correctamente los valores faltantes en tus proyectos de análisis de datos.

**Lecturas recomendadas**

[Deepnote](https://deepnote.com/workspace/platzi-escuela-datos-83832097-f136-43ff-b38d-abaa022e8ec7/project/datos-faltantes-694a3d08-7f18-421d-9e2f-c2820a79680e "Deepnote")

[GitHub - njtierney/naniar: Tidy data structures, summaries, and visualisations for missing data](https://github.com/njtierney/naniar "GitHub - njtierney/naniar: Tidy data structures, summaries, and visualisations for missing data")

[Getting Started with naniar](https://cran.r-project.org/web/packages/naniar/vignettes/getting-started-w-naniar.html "Getting Started with naniar")

[Pima Indians Diabetes Database | Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database "Pima Indians Diabetes Database | Kaggle")

## Correr una notebook dentro de otra en Deepnote

¡Hola! Te doy la bienvenida a este pequeño tutorial para conocer cómo ejecutar las funciones de una Jupyter Notebook dentro de otra en **Deepnote**.

Por una actualización de Deepnote notarás que ahora **las Notebooks están separadas de los archivos (Files)**.

![notebooks_deepnote.png](./images/notebooks_deepnote.png "notebooks_deepnote.png")

Por esta actualización tendrás que subir la notebook al sistema de archivos de Deepnote, para poder ejecutarla dentro de otra notebook utilizando la magia %run.

![pandas_missing_run.png](./images/pandas_missing_run.png "pandas_missing_run.png")

Tendrás que hacer lo anterior en tu proyecto de Deepnote del curso, ya que utilizaremos una notebook llamada pandas-missing-extension.ipynb dentro de nuestra notebook principal para utilizar métodos predefinidos en ella. Por ahora, no te preocupes por el contenido de esa notebook, en la siguiente clase hablaremos de ello. 🤗

Sigue estos pasos para **subir tu notebook como archivo (File)** a Deepnote:

1. Ve a la sección de **NOTEBOOKS** del proyecto de Deepnote.

![notebooks](./images/notebooks.png "notebooks")

2. Da clic en los tres puntos sobre la notebook `pandas-missing-extension` y da clic sobre Export as .ipynb. Esto descargará la notebook a tu computadora.

![export](./images/export.png "export")

3. Después da clic en en el signo de + en la sección **FILES** y sube la notebook **pandas-missing-extension.ipynb** que descargaste en el paso anterior en la opción `Upload File`.

![files](./images/files.png "files")

4. Repite los pasos 1-3 cada vez que desees subir la notebook `pandas-missing-extension.ipynb` a la sección **FILES**. dentro de tu proyecto en Deepnote.

5. Para terminar, ejecuta la siguiente línea dentro de la notebook `exploration-missing-values` o `live-exploration-missing-values` para cargar las funciones de la notebbok `pandas-missing-extension.ipynb` y poder utilizarlas.

![run](./images/run.png "run")

¡Nos vemos en la próxima clase! Conoceremos cómo funciona la notebook donde extendemos Pandas para manejar valores faltantes!

## Extendiendo la API de Pandas

Extender la API de **Pandas** es útil cuando deseas agregar funcionalidades personalizadas o simplificar tareas recurrentes que no están directamente soportadas por la biblioteca. A continuación, se presentan diversas formas de extender la API de Pandas.

### 1. **Métodos personalizados con `@pd.api.extensions.register_dataframe_accessor`**

Puedes crear **accesores** personalizados que te permiten extender la funcionalidad de un DataFrame o Series. Esto se hace decorando clases con `@pd.api.extensions.register_dataframe_accessor`.

#### Ejemplo:

```python
import pandas as pd

# Crear un accesor personalizado
@pd.api.extensions.register_dataframe_accessor("analytics")
class AnalyticsAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    # Método personalizado para obtener estadísticas resumidas
    def summary(self):
        return {
            "mean": self._obj.mean(),
            "median": self._obj.median(),
            "max": self._obj.max(),
            "min": self._obj.min(),
        }

# Uso del accesor
df = pd.DataFrame({
    'A': [10, 20, 30],
    'B': [1, 2, 3],
    'C': [4, 5, 6]
})

print(df.analytics.summary())
```

#### Explicación:
- Se crea un accesor llamado `analytics` que puedes utilizar directamente sobre el DataFrame.
- El método `summary()` devuelve estadísticas clave como media, mediana, máximo y mínimo.

### 2. **Agregando métodos a objetos existentes (monkey patching)**

Puedes agregar métodos directamente al DataFrame o Series utilizando **monkey patching**, aunque no es recomendable para proyectos a gran escala debido a posibles problemas de compatibilidad y mantenibilidad.

#### Ejemplo:

```python
import pandas as pd

def highlight_max(df):
    return df.style.apply(lambda x: ['background: yellow' if v == x.max() else '' for v in x], axis=1)

# Agregar método personalizado
pd.DataFrame.highlight_max = highlight_max

# Crear un DataFrame
df = pd.DataFrame({
    'A': [10, 20, 30],
    'B': [1, 50, 3],
    'C': [4, 5, 6]
})

# Usar el nuevo método
df.highlight_max()
```

#### Explicación:
- Se añade el método `highlight_max()` al objeto DataFrame.
- Este método resalta en amarillo los valores máximos en cada fila del DataFrame.

### 3. **Crear una clase heredada de DataFrame**

Una forma más estructurada de extender Pandas es creando clases que hereden de `pd.DataFrame` para agregar funcionalidad adicional.

#### Ejemplo:

```python
import pandas as pd

class MyDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return MyDataFrame

    # Método personalizado para normalizar los datos
    def normalize(self):
        return (self - self.mean()) / self.std()

# Crear un DataFrame personalizado
df = MyDataFrame({
    'A': [10, 20, 30],
    'B': [1, 50, 3],
    'C': [4, 5, 6]
})

# Usar el nuevo método
print(df.normalize())
```

#### Explicación:
- Se hereda de `pd.DataFrame` para crear una nueva clase `MyDataFrame`.
- Se añade un método `normalize()` para normalizar los valores de las columnas.
- Al sobrescribir `_constructor`, se asegura que los métodos devuelvan objetos de la misma clase.

### 4. **Funciones UDF para operaciones en DataFrame**

Las **funciones definidas por el usuario (UDF)** te permiten aplicar operaciones personalizadas sobre filas o columnas de un DataFrame.

#### Ejemplo:

```python
import pandas as pd

df = pd.DataFrame({
    'A': [10, 20, 30],
    'B': [1, 50, 3],
    'C': [4, 5, 6]
})

# Función personalizada
def custom_function(row):
    return row['A'] + row['B'] + row['C']

# Aplicar la función personalizada a cada fila
df['Sum'] = df.apply(custom_function, axis=1)

print(df)
```

#### Explicación:
- Se define una función personalizada `custom_function()` que suma los valores de las columnas `A`, `B` y `C`.
- Se aplica la función a cada fila usando `apply()`.

### 5. **Métodos personalizados con `pipe()`**

El método `pipe()` permite aplicar funciones personalizadas a DataFrames, lo que puede ser útil cuando se encadenan múltiples operaciones.

#### Ejemplo:

```python
import pandas as pd

# Crear un DataFrame
df = pd.DataFrame({
    'A': [10, 20, 30],
    'B': [1, 50, 3],
    'C': [4, 5, 6]
})

# Función personalizada
def add_columns(df, col1, col2):
    df['Sum'] = df[col1] + df[col2]
    return df

# Aplicar la función usando pipe
df = df.pipe(add_columns, 'A', 'B')

print(df)
```

#### Explicación:
- `pipe()` permite encadenar la función `add_columns()` al DataFrame, lo que facilita la legibilidad del código.

### 6. **Extender el comportamiento de las Series**

Similar a los DataFrames, también puedes extender las Series usando `@pd.api.extensions.register_series_accessor`.

#### Ejemplo:

```python
@pd.api.extensions.register_series_accessor("stats")
class StatsAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def range(self):
        return self._obj.max() - self._obj.min()

# Crear una Serie
s = pd.Series([1, 2, 3, 4, 5])

# Usar el accesor personalizado
print(s.stats.range())  # Salida: 4
```

### Resumen

1. **Accesores personalizados** (`@pd.api.extensions.register_dataframe_accessor`): Añadir métodos especializados a DataFrames y Series.
2. **Monkey patching**: Modificar objetos de Pandas directamente (aunque no siempre recomendado).
3. **Herencia de clases**: Crear clases personalizadas a partir de DataFrames o Series.
4. **Funciones UDF**: Aplicar funciones definidas por el usuario.
5. **Encadenamiento con `pipe()`**: Facilita la lectura y escritura de código con operaciones encadenadas.

Estas técnicas te permiten hacer que Pandas sea aún más flexible y personalizado según las necesidades de tus proyectos.

## Tabulación de valores faltantes

La **tabulación de valores faltantes** en un conjunto de datos es una práctica esencial en el preprocesamiento para la limpieza y preparación de los datos. En **Pandas**, podemos realizar esta tabulación para obtener un resumen claro de cuántos valores faltantes tiene cada columna, lo que nos permite entender mejor la calidad de los datos y tomar decisiones informadas sobre su manejo.

### 1. **Tabulación básica de valores faltantes**

Pandas ofrece el método `isnull()` que genera un DataFrame de booleanos (True si el valor es nulo y False en caso contrario), y `sum()` para contar el número de valores nulos por columna.

#### Ejemplo:

```python
import pandas as pd

# Crear un DataFrame con algunos valores faltantes
data = {
    'Nombre': ['Ana', 'Luis', None, 'Carlos', 'Pedro'],
    'Edad': [29, None, 22, None, 35],
    'Ciudad': ['Madrid', 'Barcelona', None, 'Sevilla', 'Valencia']
}

df = pd.DataFrame(data)

# Contar los valores faltantes por columna
faltantes = df.isnull().sum()

print(f"Valores faltantes por columna:\n{faltantes}")
```

#### Salida:

```
Valores faltantes por columna:
Nombre    1
Edad      2
Ciudad    1
dtype: int64
```

En este ejemplo, la columna `Edad` tiene 2 valores faltantes, mientras que las columnas `Nombre` y `Ciudad` tienen 1 valor faltante cada una.

### 2. **Tabulación de valores faltantes en porcentaje**

También es útil obtener el porcentaje de valores faltantes por columna para tener una idea de la magnitud de los datos faltantes.

#### Ejemplo:

```python
# Porcentaje de valores faltantes por columna
porcentaje_faltantes = df.isnull().mean() * 100

print(f"Porcentaje de valores faltantes por columna:\n{porcentaje_faltantes}")
```

#### Salida:

```
Porcentaje de valores faltantes por columna:
Nombre    20.0
Edad      40.0
Ciudad    20.0
dtype: float64
```

### 3. **Tabulación de valores faltantes por fila**

Si deseas tabular los valores faltantes por cada fila, puedes utilizar el mismo enfoque pero con el parámetro `axis=1` en la función `sum()`.

#### Ejemplo:

```python
# Contar los valores faltantes por fila
faltantes_filas = df.isnull().sum(axis=1)

print(f"Valores faltantes por fila:\n{faltantes_filas}")
```

#### Salida:

```
Valores faltantes por fila:
0    0
1    1
2    2
3    1
4    0
dtype: int64
```

Esto muestra cuántos valores faltan en cada fila del DataFrame.

### 4. **Visualización de valores faltantes**

La librería **seaborn** tiene una función `heatmap()` que permite visualizar los valores faltantes de forma gráfica.

#### Ejemplo:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Crear un mapa de calor de los valores faltantes
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Valores Faltantes")
plt.show()
```

Esto genera un gráfico de calor donde los valores nulos aparecen en un color distinto, facilitando la identificación de patrones en los datos faltantes.

### 5. **Resumen de valores faltantes en múltiples datasets**

Si tienes varios DataFrames y deseas obtener un resumen de los valores faltantes en todos ellos, puedes escribir una función que procese cada uno y devuelva un resumen.

#### Ejemplo:

```python
# Supongamos que tenemos varios DataFrames
df1 = df
df2 = pd.DataFrame({
    'Producto': ['A', 'B', None, 'C', 'D'],
    'Precio': [100, None, 150, 200, 250],
    'Stock': [None, 50, 60, None, 30]
})

datasets = {'df1': df1, 'df2': df2}

# Función para obtener el resumen de valores faltantes
def resumen_valores_faltantes(datasets):
    for name, dataset in datasets.items():
        print(f"Dataset: {name}")
        print(dataset.isnull().sum(), "\n")

# Llamar la función
resumen_valores_faltantes(datasets)
```

#### Salida:

```
Dataset: df1
Nombre    1
Edad      2
Ciudad    1
dtype: int64 

Dataset: df2
Producto    1
Precio      1
Stock       2
dtype: int64 
```

### Conclusión

La tabulación de valores faltantes es un paso crucial para evaluar la integridad de los datos. En **Pandas**, puedes hacerlo fácilmente utilizando métodos como `isnull()` y `sum()` para obtener conteos o porcentajes de valores faltantes por columna o fila, y con herramientas de visualización como **seaborn**, puedes obtener un análisis más visual de los datos incompletos. Esto te permitirá decidir si debes rellenar, eliminar o imputar estos valores antes de realizar análisis adicionales.

El error que estás viendo (`ModuleNotFoundError: No module named 'nbformat'`) se debe a que te falta el módulo `nbformat`, que es necesario para ejecutar notebooks desde otro notebook utilizando la magia `%run`.

Para resolverlo, simplemente instala el paquete `nbformat` usando pip:

### Solución:

1. Abre una celda en tu Jupyter Notebook o tu terminal y ejecuta el siguiente comando:

   ```bash
   pip install nbformat
   ```

2. Luego, intenta de nuevo ejecutar el notebook con el comando `%run`:

   ```python
   %run pandas-missing-extension.ipynb
   ```

Eso debería solucionar el problema y permitirte ejecutar el notebook correctamente.