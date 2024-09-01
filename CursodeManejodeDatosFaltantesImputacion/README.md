# Curso de Manejo de Datos Faltantes: Imputación

## El problema de trabajar con valores faltantes

El manejo de datos faltantes a través de la imputación es un proceso crucial en el análisis de datos para mejorar la calidad y la utilidad del conjunto de datos. La imputación implica estimar y reemplazar los valores faltantes con valores calculados o predichos para que el análisis y los modelos sean más precisos. Aquí tienes un resumen de técnicas comunes de imputación:

### 1. **Imputación con la Media, Mediana o Moda**
   - **Media**: Sustituye los valores faltantes por el promedio de los valores presentes en esa columna. Útil para datos numéricos que no tienen muchos valores atípicos.
   - **Mediana**: Sustituye los valores faltantes por el valor central cuando los datos están ordenados. Es menos sensible a los valores atípicos que la media.
   - **Moda**: Sustituye los valores faltantes por el valor más frecuente en la columna. Utilizado para datos categóricos.

   ```python
   import pandas as pd

   # Imputación con la media
   df['column_name'].fillna(df['column_name'].mean(), inplace=True)

   # Imputación con la mediana
   df['column_name'].fillna(df['column_name'].median(), inplace=True)

   # Imputación con la moda
   df['column_name'].fillna(df['column_name'].mode()[0], inplace=True)
   ```

### 2. **Imputación Basada en Modelos**
   - **Regresión**: Usa una variable dependiente para predecir el valor faltante basado en otras variables independientes.
   - **k-Nearest Neighbors (k-NN)**: Imputa valores basándose en la similitud entre los datos. Busca los k vecinos más cercanos y utiliza sus valores para la imputación.

   ```python
   from sklearn.impute import KNNImputer

   # Imputación con k-NN
   imputer = KNNImputer(n_neighbors=5)
   df_imputed = imputer.fit_transform(df)
   ```

### 3. **Imputación por Interpolación**
   - **Lineal**: Interpola los valores faltantes usando una función lineal entre los valores existentes.
   - **Polinómica**: Utiliza polinomios para la interpolación.

   ```python
   # Imputación lineal
   df['column_name'] = df['column_name'].interpolate(method='linear')
   ```

### 4. **Imputación por Valores Predeterminados**
   - Sustituye los valores faltantes con un valor específico que tenga sentido en el contexto del conjunto de datos (por ejemplo, 0, 'desconocido').

   ```python
   # Imputación con un valor específico
   df['column_name'].fillna('Unknown', inplace=True)
   ```

### 5. **Imputación con Datos de Vecinos**
   - Utiliza datos similares de otras observaciones para imputar los valores faltantes.

   ```python
   from sklearn.impute import SimpleImputer

   # Imputación con la mediana por defecto
   imputer = SimpleImputer(strategy='median')
   df_imputed = imputer.fit_transform(df)
   ```

### 6. **Múltiples Imputaciones**
   - **Multiple Imputation by Chained Equations (MICE)**: Imputa los valores faltantes múltiples veces y combina los resultados para tener en cuenta la incertidumbre en la imputación.

   ```python
   from miceforest import MultipleImputedData
   mice_data = MultipleImputedData(df)
   df_imputed = mice_data.complete_data()
   ```

Cada técnica tiene sus ventajas y desventajas, y la elección del método adecuado dependerá de la naturaleza de los datos y del contexto del análisis. La imputación adecuada puede mejorar la precisión del análisis y la calidad de los modelos predictivos.

**Lecturas recomendadas**

[Deepnote](https://deepnote.com/workspace/platzi-escuela-datos-83832097-f136-43ff-b38d-abaa022e8ec7/project/datos-faltantes-imputacion-bdf84ff9-f66c-44c7-a67c-ca1115a2b683 "Deepnote")

[GitHub - platzi/curso-datos-faltantes-imputacion: Repositorio del proyecto del Curso de Manejo de Datos Faltantes: Imputación](https://github.com/platzi/curso-datos-faltantes-imputacion "GitHub - platzi/curso-datos-faltantes-imputacion: Repositorio del proyecto del Curso de Manejo de Datos Faltantes: Imputación")

## Proceso de análisis y limpieza de datos

El proceso de análisis y limpieza de datos es crucial en el trabajo con datos para garantizar que los datos sean precisos, consistentes y listos para el análisis. Aquí te presento un proceso general que puedes seguir para el análisis y limpieza de datos:

### 1. **Recolección de Datos**

   - **Fuentes de Datos:** Identifica y recolecta datos de diversas fuentes como bases de datos, archivos CSV, API, etc.
   - **Formato y Estructura:** Asegúrate de que los datos estén en un formato estructurado adecuado (p. ej., CSV, Excel, JSON).

### 2. **Exploración de Datos**

   - **Carga de Datos:** Utiliza herramientas como Pandas para cargar los datos en un DataFrame.
     ```python
     import pandas as pd
     df = pd.read_csv('datos.csv')
     ```
   - **Resumen Inicial:** Examina un resumen general de los datos.
     ```python
     print(df.head())
     print(df.info())
     print(df.describe())
     ```
   - **Identificación de Valores Faltantes:** Revisa la presencia de valores faltantes.
     ```python
     print(df.isnull().sum())
     ```

### 3. **Limpieza de Datos**

   - **Manejo de Valores Faltantes:**
     - **Eliminación:** Elimina filas o columnas con valores faltantes si es aceptable.
       ```python
       df.dropna()  # Elimina filas con valores faltantes
       ```
     - **Imputación:** Reemplaza valores faltantes con valores estadísticos como la media, mediana, moda o utilizando técnicas más avanzadas.
       ```python
       df.fillna(df.mean())  # Reemplaza valores faltantes con la media
       ```
   - **Corrección de Errores:**
     - **Valores Incorrectos:** Identifica y corrige errores en los datos.
       ```python
       df['column'] = df['column'].replace('incorrect_value', 'correct_value')
       ```
     - **Duplicados:** Elimina filas duplicadas si es necesario.
       ```python
       df.drop_duplicates()
       ```
   - **Normalización y Transformación:**
     - **Escalado:** Normaliza o escala los datos si es necesario.
       ```python
       from sklearn.preprocessing import StandardScaler
       scaler = StandardScaler()
       df[['column']] = scaler.fit_transform(df[['column']])
       ```
     - **Codificación:** Codifica variables categóricas si es necesario.
       ```python
       df = pd.get_dummies(df, columns=['categorical_column'])
       ```

### 4. **Visualización de Datos**

   - **Gráficos Básicos:** Utiliza gráficos básicos para explorar la distribución y las relaciones en los datos.
     ```python
     import matplotlib.pyplot as plt
     df['column'].hist()
     plt.show()
     ```
   - **Visualización de Valores Faltantes:** Utiliza gráficos para visualizar los patrones de los valores faltantes.
     ```python
     import seaborn as sns
     sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
     plt.show()
     ```

### 5. **Análisis de Datos**

   - **Estadísticas Descriptivas:** Calcula estadísticas descriptivas y explora la distribución de los datos.
     ```python
     print(df.describe())
     ```
   - **Correlación:** Examina la correlación entre variables.
     ```python
     print(df.corr())
     sns.heatmap(df.corr(), annot=True)
     plt.show()
     ```

### 6. **Documentación y Reporte**

   - **Documentación:** Documenta todos los pasos tomados, decisiones y transformaciones realizadas.
   - **Reporte:** Genera informes que resuman el estado de los datos y los resultados del análisis.

### Herramientas Comunes

- **Python Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
- **Software de BI:** Power BI, Tableau.
- **Jupyter Notebooks:** Para documentar y ejecutar el análisis de manera interactiva.

Este proceso puede variar según el contexto y los requisitos específicos del proyecto, pero estos pasos proporcionan una guía general para el análisis y la limpieza de datos.

**Lecturas recomendadas**

[NHANES Tutorials](https://wwwn.cdc.gov/nchs/nhanes/tutorials/default.aspx "NHANES Tutorials")

## Visualizar y eliminar valores faltantes

Para visualizar y eliminar valores faltantes en un conjunto de datos usando **Python** y **Pandas**, puedes seguir estos pasos:

### 1. **Visualización de valores faltantes:**

#### a) Usar `isnull()` o `isna()` para detectar valores faltantes:
```python
import pandas as pd

# Crear un DataFrame de ejemplo
df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4],
    'C': [1, None, None, 4]
})

# Visualizar dónde hay valores faltantes (True si falta el valor)
print(df.isnull())
```

#### b) Contar valores faltantes por columna:
```python
# Contar los valores faltantes por columna
print(df.isnull().sum())
```

#### c) Visualización gráfica con **Seaborn**:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualizar el mapa de calor de los valores faltantes
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.show()
```

### 2. **Eliminar valores faltantes:**

#### a) Eliminar filas con valores faltantes:
```python
# Eliminar filas que tengan algún valor faltante
df_cleaned = df.dropna()
print(df_cleaned)
```

#### b) Eliminar columnas con valores faltantes:
```python
# Eliminar columnas que tengan algún valor faltante
df_cleaned_columns = df.dropna(axis=1)
print(df_cleaned_columns)
```

### 3. **Rellenar valores faltantes (opcional):**
Si en lugar de eliminar, prefieres rellenar los valores faltantes con una estrategia de imputación:

#### a) Rellenar con un valor constante (ej. 0 o la media):
```python
# Rellenar valores faltantes con 0
df_filled = df.fillna(0)
print(df_filled)

# Rellenar con la media de la columna
df_filled_mean = df.fillna(df.mean())
print(df_filled_mean)
```

Este proceso te ayudará a manejar los valores faltantes en tu análisis de datos.

## Implicaciones de los distintos tipos de valores faltantes

Existen tres tipos principales de mecanismos de valores faltantes que tienen implicaciones diferentes en los análisis de datos: **MCAR (Missing Completely at Random)**, **MAR (Missing at Random)** y **MNAR (Missing Not at Random)**. Cada uno de estos mecanismos impacta la forma en que se manejan los datos y puede influir en los resultados de los análisis y modelos.

### 1. **Missing Completely at Random (MCAR)**

**Definición**: Los valores faltantes son completamente aleatorios y no dependen de los valores de otras variables ni de la propia variable que contiene el valor faltante.

**Implicaciones**:
- **No sesgo**: Cuando los datos faltan de manera completamente aleatoria, no introducen sesgo en el análisis. La eliminación de datos (listwise o pairwise) no afectará la validez de las conclusiones.
- **Métodos**: Es el caso más fácil de tratar. Se pueden eliminar filas o imputar sin riesgo de sesgo.
  
**Ejemplo**: Una encuesta en la que algunas personas no respondieron debido a que la pregunta fue omitida por error de diseño, sin que esto dependa de ninguna característica de las personas.

### 2. **Missing at Random (MAR)**

**Definición**: La probabilidad de que falte un dato depende de otras variables en el conjunto de datos, pero no del valor de la variable que falta en sí.

**Implicaciones**:
- **Sesgo moderado**: Si los valores faltantes están relacionados con otras variables observadas, ignorar o eliminar datos podría sesgar los resultados.
- **Métodos**: El sesgo puede minimizarse utilizando técnicas de **imputación múltiple** o modelos predictivos que tengan en cuenta las otras variables relacionadas.

**Ejemplo**: En un estudio médico, es posible que las personas mayores tiendan a omitir respuestas a preguntas sobre el uso de tecnología. Aunque la edad no afecta directamente la pregunta faltante, es un factor asociado con la ausencia de datos.

### 3. **Missing Not at Random (MNAR)**

**Definición**: La falta de datos está directamente relacionada con el valor de la variable que falta. Es decir, el hecho de que falte un dato depende del propio valor que no se observó.

**Implicaciones**:
- **Sesgo significativo**: Si los valores faltan sistemáticamente debido a su valor real, eliminar o ignorar estos datos sin una estrategia adecuada puede producir sesgos severos en el análisis.
- **Métodos**: Este tipo de valores faltantes es el más difícil de tratar. Puede ser necesario realizar estudios adicionales o recoger más datos. Las imputaciones basadas en modelos pueden intentar estimar los valores faltantes, pero el sesgo es difícil de eliminar completamente.

**Ejemplo**: En un estudio sobre ingresos, las personas con ingresos más altos podrían ser menos propensas a revelar esa información, haciendo que los ingresos altos estén sistemáticamente ausentes del conjunto de datos.

### **Resumen de implicaciones y estrategias**:

| Tipo de faltante | Implicación | Estrategia de manejo |
|------------------|-------------|----------------------|
| **MCAR** | No hay sesgo si los datos se eliminan. | Se pueden eliminar filas o columnas sin afectar el análisis. |
| **MAR**  | Existe sesgo si no se maneja adecuadamente. | Imputación múltiple o modelado considerando variables relacionadas. |
| **MNAR** | Sesgo significativo, más difícil de tratar. | Recoger más datos o usar técnicas avanzadas de imputación/modelado. |

### Elección de método:
- Para **MCAR**, la eliminación de datos suele ser adecuada.
- Para **MAR**, la **imputación múltiple** o los **modelos predictivos** pueden reducir el sesgo.
- Para **MNAR**, es crucial entender la causa de los valores faltantes. Los métodos de imputación predictiva pueden ayudar, pero es posible que se necesiten técnicas especializadas.

Conocer el mecanismo de los valores faltantes es fundamental para elegir la estrategia de limpieza de datos más adecuada y minimizar el impacto en los análisis.

## Amplía tu conjunto de herramientas para explorar valores faltantes

Para explorar y manejar valores faltantes de manera efectiva, es crucial ampliar las herramientas y técnicas que tienes a tu disposición. Aquí te dejo algunas herramientas y enfoques útiles que puedes implementar:

### 1. **Visualización de valores faltantes**:
   Existen varias librerías en Python que permiten visualizar valores faltantes de manera intuitiva:

   - **Matplotlib y Seaborn**: Úsalos para crear gráficos de calor o gráficos de barras que te permitan identificar la cantidad de valores faltantes.
     
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt
     
     sns.heatmap(df.isnull(), cbar=False)
     plt.show()
     ```

   - **Missingno**: Una librería especializada en la visualización de datos faltantes.

     ```python
     import missingno as msno
     
     # Visualización básica
     msno.matrix(df)
     
     # Visualización de dendograma para identificar patrones de valores faltantes correlacionados
     msno.dendrogram(df)
     ```

### 2. **Matriz de sombras**:
   Una técnica avanzada que permite crear una matriz binaria donde 1 indica la presencia de un valor y 0 indica su ausencia. Esto permite analizar las correlaciones entre la presencia de valores faltantes en diferentes variables.

   ```python
   import pandas as pd
   
   # Matriz de sombras
   shadow_matrix = df.isnull().astype(int)
   ```

   Esta técnica puede ser útil para identificar si la falta de datos en una columna está correlacionada con la falta de datos en otra.

### 3. **Correlación de nulidad**:
   Explorar la relación entre valores faltantes en diferentes columnas es clave para entender el origen de los valores faltantes. Esto se puede hacer usando la función `pandas.DataFrame.corr()` para calcular la correlación entre la matriz de nulidad.

   ```python
   # Calcular la correlación entre valores faltantes
   nullity_corr = df.isnull().corr()
   ```

   También puedes calcular correlaciones específicas con valores faltantes usando el método `pairwise deletion` o la imputación de valores como último recurso.

### 4. **Imputación de valores faltantes**:
   Hay múltiples enfoques que puedes usar para imputar valores faltantes:
   
   - **Imputación por la media/mediana/moda**: Fácil de implementar, pero puede sesgar los resultados.
   
     ```python
     df['column'].fillna(df['column'].mean(), inplace=True)
     ```

   - **Imputación usando métodos más avanzados**:
     - **K-Nearest Neighbors (KNN)**: Este método utiliza las observaciones más cercanas para imputar valores faltantes.
     - **Iterative Imputer**: Realiza imputaciones iterativas en las columnas del conjunto de datos utilizando un modelo de regresión.
     
     ```python
     from sklearn.impute import KNNImputer

     imputer = KNNImputer(n_neighbors=5)
     df_imputed = imputer.fit_transform(df)
     ```

### 5. **Análisis de patrones de valores faltantes**:
   Identificar patrones en los datos faltantes te ayudará a elegir la estrategia adecuada. Existen tres tipos de valores faltantes:
   - **MCAR (Missing Completely at Random)**: No hay patrón.
   - **MAR (Missing at Random)**: Los valores faltantes dependen de otras variables.
   - **MNAR (Missing Not at Random)**: Los valores faltantes dependen de la propia variable con valores ausentes.

   Usando herramientas de visualización como **Missingno** y análisis estadístico, puedes descubrir estos patrones.

### 6. **Funciones adicionales de `pyjanitor`**:
   `pyjanitor` expande las capacidades de `pandas` con funciones específicas para limpieza, incluidas aquellas que exponen valores faltantes implícitos:

   ```python
   import janitor
   df = df.complete()
   ```

   Puedes utilizar `janitor` para completar los valores implícitos y exponer relaciones entre variables.

### 7. **Uso de máscaras para filtrar y analizar valores faltantes**:
   Puedes utilizar máscaras booleanas para filtrar filas y columnas con valores faltantes y hacer un análisis más detallado.

   ```python
   missing_mask = df.isnull()
   missing_columns = df.columns[missing_mask.any()]
   ```

### Conclusión
Ampliar tu conjunto de herramientas para manejar valores faltantes implica una combinación de **visualización**, **análisis estadístico** y **métodos de imputación** avanzados. Utilizar bibliotecas como **Seaborn**, **Missingno**, **pyjanitor**, junto con las funciones integradas de **Pandas**, te permitirá explorar los datos de manera más efectiva y tomar decisiones informadas sobre cómo manejar valores faltantes.

## Tratamiento de variables categóricas para imputación: codificación ordinal

El tratamiento de variables categóricas para imputación es un paso importante en la limpieza de datos, ya que la mayoría de los métodos de imputación (como la media, la mediana o incluso modelos más avanzados como KNN o regresiones) requieren que las variables categóricas estén codificadas numéricamente. 

La **codificación ordinal** es una técnica que convierte las categorías en números de manera que las categorías tengan un orden lógico, pero sin implicar una relación de magnitud precisa entre las categorías.

### ¿Qué es la codificación ordinal?

La **codificación ordinal** se utiliza cuando las categorías de una variable tienen un orden inherente. Un ejemplo común es una variable de satisfacción con las opciones: "Bajo", "Medio", "Alto". El orden es claro, y se puede representar con números como 1, 2, y 3, donde 1 representa "Bajo" y 3 representa "Alto". 

Este tipo de codificación es ideal para variables categóricas ordinales, en las que las categorías pueden ordenarse de manera lógica.

### Pasos para la imputación con codificación ordinal:

1. **Identificación de las variables ordinales**: Primero, debes identificar las variables categóricas que tienen un orden lógico entre sus categorías. Por ejemplo:
   - Nivel educativo (Primaria, Secundaria, Universitaria).
   - Tamaño de una empresa (Pequeña, Mediana, Grande).

2. **Codificación ordinal**: Convertir las categorías en valores numéricos que respeten el orden inherente. Para hacer esto, puedes usar la función `OrdinalEncoder` de `sklearn` o codificar manualmente.

   ```python
   from sklearn.preprocessing import OrdinalEncoder
   import pandas as pd

   # Ejemplo de datos categóricos
   data = {'Nivel_Educativo': ['Primaria', 'Secundaria', 'Universitaria', None]}
   df = pd.DataFrame(data)
   
   # Definir el orden de las categorías
   encoder = OrdinalEncoder(categories=[['Primaria', 'Secundaria', 'Universitaria']])
   
   # Codificar
   df['Nivel_Educativo_Cod'] = encoder.fit_transform(df[['Nivel_Educativo']])
   print(df)
   ```

   Salida:
   ```
     Nivel_Educativo  Nivel_Educativo_Cod
   0        Primaria                  0.0
   1      Secundaria                  1.0
   2   Universitaria                  2.0
   3            None                  NaN
   ```

3. **Imputación**: Una vez que la variable ha sido codificada ordinalmente, puedes aplicar métodos de imputación estándar. Por ejemplo, imputar los valores faltantes utilizando la media o mediana en lugar de dejar los valores como `NaN`. 

   ```python
   from sklearn.impute import SimpleImputer

   # Imputar la mediana en los valores faltantes
   imputer = SimpleImputer(strategy='median')
   df['Nivel_Educativo_Imputado'] = imputer.fit_transform(df[['Nivel_Educativo_Cod']])
   print(df)
   ```

   Salida:
   ```
     Nivel_Educativo  Nivel_Educativo_Cod  Nivel_Educativo_Imputado
   0        Primaria                  0.0                       0.0
   1      Secundaria                  1.0                       1.0
   2   Universitaria                  2.0                       2.0
   3            None                  NaN                       1.0
   ```

   En este caso, el valor faltante se ha reemplazado por la mediana.

### Beneficios de la codificación ordinal

1. **Imputación más precisa**: Al conservar el orden inherente de las categorías, la imputación de los valores faltantes será más precisa, ya que métodos como la media o la mediana pueden mantener la lógica de las relaciones entre las categorías.
   
2. **Modelos más robustos**: Los modelos de Machine Learning y las técnicas estadísticas suelen funcionar mejor con datos numéricos, y la codificación ordinal facilita el uso de estos modelos para variables categóricas.

3. **Evita la codificación arbitraria**: En lugar de usar números sin sentido o hacer una codificación "one-hot" que puede introducir complejidad innecesaria, la codificación ordinal refleja el orden real de las categorías.

### Cuándo no usar codificación ordinal

No todas las variables categóricas deben ser codificadas ordinalmente. **Solo** debes aplicar esta técnica si las categorías tienen un **orden lógico**. Si las categorías no tienen un orden, deberías usar otras técnicas como la **codificación "one-hot"** o **frecuencial**.

### Conclusión

La codificación ordinal es una técnica efectiva cuando necesitas imputar valores faltantes en variables categóricas con un orden implícito. La clave es respetar el orden de las categorías, para luego poder aplicar métodos de imputación y análisis que te proporcionen resultados precisos y consistentes.

## Tratamiento de variables categóricas para imputación: one-hot encoding

El tratamiento de variables categóricas mediante **One-Hot Encoding** es una técnica común en el preprocesamiento de datos. Convierte las variables categóricas en varias columnas binarias, donde cada columna representa una categoría única, asignando un valor de `1` cuando la categoría está presente y `0` cuando no lo está. Este método evita el problema de orden implícito que puede ocurrir al usar codificación ordinal en variables categóricas sin una jerarquía natural.

### Proceso de One-Hot Encoding

1. **Selecciona las variables categóricas** que deseas codificar.
2. **Aplica One-Hot Encoding** para convertirlas en columnas binarias.
3. **Asegúrate de que el resto de las columnas no categóricas** (si existen) pasen sin modificaciones o se procesen adecuadamente.

### Ejemplo en Python usando `pandas` y `sklearn`

#### Paso 1: Importar bibliotecas

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
```

#### Paso 2: Crear un conjunto de datos de ejemplo

```python
# Crear un DataFrame con una variable categórica
data = {
    'gender': ['Male', 'Female', 'Female', 'Male'],
    'age': [23, 45, 31, 22],
    'income': [50000, 60000, 65000, 45000]
}

df = pd.DataFrame(data)
```

#### Paso 3: Aplicar One-Hot Encoding a la variable categórica

```python
# Definir la transformación con OneHotEncoder
categorical_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['gender'])  # Aplicar One-Hot Encoding a la columna 'gender'
    ],
    remainder='passthrough'  # El resto de las columnas permanecen sin cambios
)

# Transformar el DataFrame
transformed_df = pd.DataFrame(categorical_transformer.fit_transform(df))

# Cambiar los nombres de las columnas para que sean más comprensibles
transformed_df.columns = ['gender_Female', 'gender_Male', 'age', 'income']

print(transformed_df)
```

### Salida esperada:

| gender_Female | gender_Male | age | income |
|:-------------:|:-----------:|:---:|:------:|
|       0       |      1      |  23 | 50000  |
|       1       |      0      |  45 | 60000  |
|       1       |      0      |  31 | 65000  |
|       0       |      1      |  22 | 45000  |

### Consideraciones adicionales:
- **One-Hot Encoding** es particularmente útil cuando no existe una relación de orden entre las categorías.
- Si tienes muchas categorías, este método puede aumentar significativamente el número de columnas, lo cual es algo a tener en cuenta al trabajar con grandes volúmenes de datos.
