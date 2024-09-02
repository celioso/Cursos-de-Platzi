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

## Métodos de imputación de valores faltantes

La **imputación de valores faltantes** consiste en estimar los valores ausentes en un conjunto de datos utilizando diversos métodos. Estos métodos pueden ser simples o complejos, dependiendo de la cantidad de datos faltantes y de la naturaleza del problema. Aquí se describen los métodos más comunes para la imputación de valores faltantes.

### 1. **Imputación Simple**
Estos métodos son fáciles de implementar y son adecuados cuando la cantidad de datos faltantes es pequeña.

#### a. **Imputación con la Media**
Se reemplazan los valores faltantes por el valor medio de la variable. Es útil cuando los datos están distribuidos de manera uniforme y no hay grandes outliers.
```python
df['column'] = df['column'].fillna(df['column'].mean())
```

#### b. **Imputación con la Mediana**
Utiliza la mediana para reemplazar los valores faltantes. Es más robusta que la media en presencia de outliers.
```python
df['column'] = df['column'].fillna(df['column'].median())
```

#### c. **Imputación con la Moda**
Para variables categóricas, la imputación con la moda reemplaza los valores faltantes con la categoría más frecuente.
```python
df['column'] = df['column'].fillna(df['column'].mode()[0])
```

### 2. **Imputación Basada en Reglas**
Se puede usar información contextual o reglas específicas del dominio para imputar valores faltantes.

#### a. **Imputación Condicional**
Si una variable tiene una relación lógica con otra, se pueden utilizar reglas basadas en esta relación.
```python
df.loc[(df['edad'] > 18) & (df['estado_civil'].isnull()), 'estado_civil'] = 'Soltero'
```

### 3. **Imputación Multivariante**
Este tipo de imputación usa información de múltiples variables para estimar los valores faltantes. Es más precisa, pero más compleja.

#### a. **Imputación por Regresión**
Utiliza un modelo de regresión (lineal o no lineal) para predecir los valores faltantes a partir de otras variables en el conjunto de datos.

```python
from sklearn.linear_model import LinearRegression

# Selección de datos sin valores nulos
train_data = df.dropna(subset=['column_with_missing_values'])
train_X = train_data[['predictor_1', 'predictor_2']]
train_y = train_data['column_with_missing_values']

# Entrenamiento del modelo
model = LinearRegression()
model.fit(train_X, train_y)

# Predicción de los valores faltantes
df.loc[df['column_with_missing_values'].isnull(), 'column_with_missing_values'] = model.predict(df[['predictor_1', 'predictor_2']])
```

#### b. **Imputación Múltiple (Multiple Imputation by Chained Equations, MICE)**
Este método realiza imputaciones múltiples utilizando una secuencia de modelos de regresión, para luego promediar los resultados. Es robusto y maneja la incertidumbre en los valores imputados.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Imputación iterativa
imputer = IterativeImputer()
df_imputed = pd.DataFrame(imputer.fit_transform(df))
```

### 4. **Imputación con K-Nearest Neighbors (KNN)**
Este método utiliza los valores de las observaciones más cercanas (k-nearest neighbors) para imputar los valores faltantes. Funciona bien cuando los datos tienen relaciones no lineales entre variables.

```python
from sklearn.impute import KNNImputer

# Imputación con KNN
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

### 5. **Imputación por Forward y Backward Filling**
Este método llena los valores faltantes utilizando los valores no nulos anteriores o posteriores. Es común en datos de series temporales.

#### a. **Forward Fill**
Rellena los valores faltantes con el último valor no nulo anterior.
```python
df['column'] = df['column'].fillna(method='ffill')
```

#### b. **Backward Fill**
Rellena los valores faltantes con el siguiente valor no nulo.
```python
df['column'] = df['column'].fillna(method='bfill')
```

### Comparación de Métodos

| **Método**             | **Ventajas**                                           | **Desventajas**                                                |
|------------------------|--------------------------------------------------------|---------------------------------------------------------------|
| Media, Mediana, Moda    | Rápido y fácil de implementar                          | Puede introducir sesgos o distorsionar la distribución         |
| Regresión              | Usa la relación entre variables                        | Complejidad, depende de la calidad del modelo                  |
| KNN                    | Captura relaciones no lineales                         | Computacionalmente costoso para grandes conjuntos de datos     |
| MICE                   | Imputación robusta y manejo de incertidumbre           | Complejo de implementar y costoso en tiempo de cálculo         |
| Forward/Backward Fill   | Útil en datos de series temporales                     | Puede generar datos poco realistas en ciertos escenarios       |

### Conclusión
Elegir el método adecuado de imputación depende del tipo de datos, el contexto y la cantidad de datos faltantes. Para conjuntos de datos pequeños o problemas simples, los métodos como la media o la mediana pueden ser suficientes, mientras que para datos más complejos se recomienda el uso de técnicas multivariantes como la imputación por regresión o MICE.

## Imputación por media, mediana y moda

La **imputación por media, mediana y moda** es uno de los métodos más simples y comunes para manejar valores faltantes. Estos métodos son fáciles de implementar y proporcionan una solución rápida, especialmente cuando los valores faltantes son pocos. A continuación, te explico cada uno de estos enfoques:

### 1. **Imputación por Media**
La imputación por media reemplaza los valores faltantes de una variable numérica por el promedio de todos los valores no faltantes de esa variable. 

- **Ventajas**: Es fácil de calcular e implementar.
- **Desventajas**: Puede distorsionar la distribución de los datos, especialmente si hay outliers, y puede subestimar la varianza.

```python
# Imputación por media
df['columna'] = df['columna'].fillna(df['columna'].mean())
```

### 2. **Imputación por Mediana**
La imputación por mediana utiliza el valor central de los datos para reemplazar los valores faltantes. Es más robusta que la media en presencia de valores atípicos.

- **Ventajas**: La mediana es menos sensible a outliers, por lo que es más adecuada para datos sesgados.
- **Desventajas**: Al igual que con la media, puede reducir la variabilidad en los datos.

```python
# Imputación por mediana
df['columna'] = df['columna'].fillna(df['columna'].median())
```

### 3. **Imputación por Moda**
Para variables categóricas, la imputación por moda reemplaza los valores faltantes con la categoría más frecuente en los datos.

- **Ventajas**: Es útil para variables categóricas.
- **Desventajas**: Si hay varias categorías con frecuencias similares, puede no ser representativo imputar con la moda.

```python
# Imputación por moda (valores categóricos)
df['columna'] = df['columna'].fillna(df['columna'].mode()[0])
```

### Comparación y Aplicaciones
| **Método**  | **Aplicación**                          | **Ventajas**                                           | **Desventajas**                                         |
|-------------|-----------------------------------------|--------------------------------------------------------|--------------------------------------------------------|
| **Media**   | Variables numéricas sin muchos outliers | Fácil de implementar, rápida                           | Sensible a outliers, puede alterar la distribución      |
| **Mediana** | Variables numéricas con outliers        | Menos afectada por outliers                            | Puede no capturar la variabilidad original de los datos |
| **Moda**    | Variables categóricas                   | Útil para variables con categorías repetitivas          | No funciona bien si no hay una categoría dominante      |

### Consideraciones
- **Reducción de la varianza**: Al utilizar la media, mediana o moda, se reduce la variabilidad en los datos, lo que puede ser perjudicial en algunos análisis.
- **Sesgo**: Estos métodos suponen que los valores faltantes son aleatorios. Si los valores faltantes tienen un patrón, la imputación por media, mediana o moda puede introducir sesgos.

Este enfoque es más adecuado cuando hay pocos valores faltantes y no se requiere una alta precisión. Para conjuntos de datos con muchas variables o relaciones complejas, se pueden usar métodos más avanzados como la imputación multivariante o la regresión.

## Imputación por llenado hacia atrás y hacia adelante

La **imputación por llenado hacia atrás y hacia adelante** es una técnica comúnmente utilizada para manejar valores faltantes en series temporales. Estos métodos aprovechan la continuidad temporal de los datos para rellenar los valores faltantes con el valor más cercano disponible en el tiempo.

### 1. **Llenado hacia adelante (Forward Fill)**
Este método rellena los valores faltantes con el último valor conocido anterior. Es decir, se "arrastra" el último valor observado hacia adelante para completar los valores faltantes.

- **Aplicación**: Útil cuando se asume que los valores no cambian drásticamente en cortos períodos de tiempo o cuando un valor faltante puede ser razonablemente estimado como el mismo que el anterior.
- **Ventajas**: Es fácil de aplicar y respeta la estructura temporal de los datos.
- **Desventajas**: Si hay grandes cambios entre períodos, este método puede introducir sesgos.

```python
# Imputación hacia adelante (forward fill)
df['columna'] = df['columna'].fillna(method='ffill')
```

### 2. **Llenado hacia atrás (Backward Fill)**
Este método rellena los valores faltantes con el siguiente valor disponible, es decir, "arrastra" el siguiente valor conocido hacia atrás.

- **Aplicación**: Se usa cuando el valor futuro es una buena aproximación del valor faltante o cuando los datos faltantes deberían parecerse a los valores que siguen.
- **Ventajas**: Es útil cuando los valores futuros se pueden asumir similares a los faltantes.
- **Desventajas**: Similar al llenado hacia adelante, puede introducir sesgos si hay fluctuaciones grandes en los datos.

```python
# Imputación hacia atrás (backward fill)
df['columna'] = df['columna'].fillna(method='bfill')
```

### 3. **Combinación de ambos métodos**
En algunas situaciones, puedes combinar ambos métodos, primero llenando hacia adelante y luego hacia atrás, para asegurar que los valores faltantes en el medio de los datos se llenen de alguna manera.

```python
# Primero hacia adelante, luego hacia atrás
df['columna'] = df['columna'].fillna(method='ffill').fillna(method='bfill')
```

### Ejemplo
Imagina un conjunto de datos de temperaturas diarias donde algunos días no se registraron datos. Si utilizamos **forward fill**, los valores faltantes serán reemplazados por la última temperatura registrada. Si usamos **backward fill**, serán reemplazados por la próxima temperatura conocida.

```python
import pandas as pd
import numpy as np

# Ejemplo de DataFrame con valores faltantes
data = {'fecha': pd.date_range('2023-01-01', periods=10),
        'temperatura': [30, np.nan, np.nan, 35, 33, np.nan, 32, 31, np.nan, 30]}

df = pd.DataFrame(data)

# Llenado hacia adelante
df_ffill = df.fillna(method='ffill')

# Llenado hacia atrás
df_bfill = df.fillna(method='bfill')

print(df_ffill)
print(df_bfill)
```

### Consideraciones
- **Patrones temporales**: Estos métodos son particularmente útiles en series temporales, donde los valores faltantes ocurren de forma secuencial.
- **Sesgo potencial**: Como estos métodos suponen que el valor pasado o futuro es una buena aproximación para el valor faltante, pueden introducir sesgos si hay cambios bruscos en los datos.
- **Casos extremos**: Si los primeros o últimos valores en la serie están faltando, no habrá valores anteriores o futuros con los cuales reemplazarlos, lo que dejará esos valores como `NaN`.

Estos métodos son simples, efectivos y no requieren supuestos complejos, pero es importante evaluar si son adecuados en función de la naturaleza de los datos.

## Imputación por interpolación

La **imputación por interpolación** es un método para estimar valores faltantes en un conjunto de datos, particularmente útil en series temporales o cuando se espera que los datos cambien de manera continua o progresiva. Este método utiliza la tendencia y el comportamiento de los valores circundantes para estimar el valor faltante.

### Tipos de interpolación:

1. **Interpolación lineal**:
   Es el método más básico. Estima los valores faltantes asumiendo que los datos varían de manera lineal entre los puntos observados.

   ```python
   # Interpolación lineal
   df['columna'] = df['columna'].interpolate(method='linear')
   ```

2. **Interpolación polinómica**:
   Utiliza un polinomio de grado `n` para ajustar los valores entre los puntos. Es más flexible que la lineal, pero también puede ser más propensa a oscilar en los extremos.

   ```python
   # Interpolación polinómica (grado 2)
   df['columna'] = df['columna'].interpolate(method='polynomial', order=2)
   ```

3. **Interpolación basada en splines**:
   Utiliza splines cúbicos o de otro grado para suavizar las curvas entre los puntos. Es útil para datos que no siguen un patrón lineal simple, pero que aún deben mantener una curva suave.

   ```python
   # Interpolación cúbica (splines cúbicos)
   df['columna'] = df['columna'].interpolate(method='spline', order=3)
   ```

4. **Interpolación basada en el índice temporal**:
   Si trabajas con series temporales, puedes interpolar usando los índices temporales. Esto es útil cuando la regularidad temporal es más importante que la relación entre los valores de las columnas.

   ```python
   # Interpolación basada en el índice temporal
   df['columna'] = df['columna'].interpolate(method='time')
   ```

### Ejemplo práctico:
Imagina un conjunto de datos que registra la temperatura diaria, pero algunos días faltan registros. Usamos interpolación para estimar esos valores faltantes.

```python
import pandas as pd
import numpy as np

# Crear un DataFrame con fechas y temperaturas
data = {'fecha': pd.date_range('2023-01-01', periods=10),
        'temperatura': [30, np.nan, np.nan, 35, 33, np.nan, 32, 31, np.nan, 30]}

df = pd.DataFrame(data)

# Interpolación lineal
df['temperatura_interpolada'] = df['temperatura'].interpolate(method='linear')

# Mostrar el resultado
print(df)
```

### Ventajas de la interpolación:
- **Aprovecha el patrón de los datos**: Si los datos siguen una tendencia continua, la interpolación proporciona estimaciones razonables.
- **Flexibilidad**: Puedes usar diferentes métodos de interpolación (lineal, polinómica, spline) para ajustar el método a la naturaleza de los datos.
- **Preserva la estructura temporal**: En series temporales, la interpolación basada en el tiempo permite hacer imputaciones manteniendo el orden cronológico de los datos.

### Desventajas de la interpolación:
- **No es adecuada para todos los tipos de datos**: Si los valores faltantes son el resultado de un proceso no continuo o aleatorio, la interpolación puede introducir sesgos.
- **Oscilaciones**: Métodos más complejos como los polinomios pueden producir oscilaciones inesperadas, especialmente en los extremos de los datos.
- **Asume continuidad**: Funciona mejor cuando se puede suponer que los valores entre los puntos siguen un patrón predecible o continuo.

### Consideraciones:
- Si los valores faltantes son numerosos o consecutivos, la interpolación puede generar estimaciones menos fiables.
- La interpolación es más adecuada para datos numéricos y en su mayoría aplicable a series temporales, aunque también se puede usar en otras estructuras siempre que los datos tengan una secuencia o patrón claro.

Este método es útil en muchos casos, pero siempre debes evaluar si las suposiciones de continuidad son razonables para tus datos.

### Pandas.DataFrame.interpolate: Donantes vs. Modelos

En la biblioteca Pandas, la función `DataFrame.interpolate` ofrece diversas opciones para realizar interpolación de valores faltantes en un DataFrame. A continuación se clasifican según su enfoque principal:

**Métodos basados en donantes:**

- `method='linear'`: Interpolación lineal simple entre los dos puntos más cercanos.
- `method='nearest'`: Asigna el valor del punto más cercano al punto con valor faltante.
- `method='quadratic'`: Interpolación cuadrática utilizando los dos puntos más cercanos y el siguiente punto más cercano en la misma dirección.
- `method='cubic'`: Interpolación cúbica utilizando los dos puntos más cercanos y los dos siguientes puntos más cercanos en la misma dirección.
- `method='krogh'`: Interpolación de Akima, que utiliza una función cúbica a trozos con restricciones de monotonía.
- `method='spline'`: Interpolación cúbica con splines de B-spline.

**Métodos basados en modelos:**

- `method='barycentric'`: Interpolación baricéntrica, que utiliza una ponderación basada en la distancia de los puntos vecinos.
- `method='polynomial'`: Interpolación polinomial de orden especificado (parámetro 'order').
- `method='pchip'`: Interpolación cúbica monotónica de Hermite con preservación de la forma local.

**Otros métodos:**

- `method='index'`: Interpolación lineal usando el índice del DataFrame.
- `method='pad'`: Rellena los valores faltantes con el valor del borde más cercano (opción 'ffill' para relleno hacia adelante, 'bfill' para relleno hacia atrás).

Es importante destacar que algunos métodos pueden combinar elementos de ambos enfoques. Por ejemplo, la interpolación cúbica con splines de B-spline (`method='spline'`) se basa en un modelo matemático, pero también utiliza la información de los puntos vecinos.

La elección del método adecuado dependerá de diversos factores, como:

- Tamaño del conjunto de datos: Los métodos basados en modelos pueden ser más precisos para conjuntos de datos grandes, mientras que los métodos basados en donantes pueden ser más eficientes para conjuntos de datos pequeños.
- Patrones en los datos: Los métodos basados en modelos pueden ser más adecuados para conjuntos de datos con patrones complejos, mientras que los métodos basados en donantes pueden ser más robustos para conjuntos de datos con ruido.
- Precisión requerida: Los métodos basados en modelos pueden ofrecer mayor precisión, pero esto puede implicar un mayor costo computacional.

Se recomienda evaluar diferentes métodos y seleccionar el que mejor se adapte a las necesidades específicas de cada caso.

**Lecturas recomendadas**

[pandas.DataFrame.interpolate — pandas 1.5.1 documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html "pandas.DataFrame.interpolate — pandas 1.5.1 documentation")

## Imputación por KNN

La **imputación por K-Nearest Neighbors (KNN)** es una técnica avanzada utilizada para estimar valores faltantes basándose en la similitud de las observaciones con otras observaciones del conjunto de datos. La idea detrás de este enfoque es que los valores faltantes pueden ser aproximados utilizando los valores de las observaciones más cercanas (vecinas) en el espacio de las características.

### Concepto básico de KNN:
El algoritmo de KNN calcula la "distancia" entre las observaciones, donde cada observación es un vector de características (variables). En el caso de imputación, se seleccionan las **K observaciones más cercanas** (vecinos) a la observación con el valor faltante y se utiliza alguna función (promedio, moda, etc.) para estimar el valor faltante con base en los valores de esos vecinos.

### Pasos de la imputación por KNN:

1. **Definir la distancia**: Se elige una métrica de distancia, comúnmente la distancia Euclidiana, para determinar qué observaciones están "cerca" entre sí.
   
2. **Seleccionar K vecinos**: Se selecciona un número \( K \) de vecinos más cercanos a la observación que tiene el valor faltante.

3. **Imputar el valor faltante**: El valor faltante se estima utilizando los valores de los vecinos seleccionados. Si es una variable numérica, se puede usar la media o mediana de los vecinos. Para una variable categórica, se puede utilizar la moda (el valor más frecuente entre los vecinos).

### Implementación en Python:

En Python, la imputación por KNN puede realizarse utilizando la librería `sklearn` y otras herramientas como `fancyimpute` o `KNNImputer`.

#### Usando `KNNImputer` de `scikit-learn`:

```python
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Crear un DataFrame con valores faltantes
data = {'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, np.nan, 3, 2],
        'C': [7, 8, 9, 10, 11]}

df = pd.DataFrame(data)

# Crear un objeto KNNImputer con k=2 vecinos
imputer = KNNImputer(n_neighbors=2)

# Imputar los valores faltantes
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Mostrar el resultado
print(df_imputed)
```

En este ejemplo, el algoritmo KNN selecciona los **2 vecinos más cercanos** para imputar los valores faltantes en las columnas 'A' y 'B'.

### Métricas de distancia comunes:
- **Distancia Euclidiana**: Mide la distancia entre dos puntos en el espacio de múltiples dimensiones. Es la más común para datos numéricos.
  
  \[
  d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
  \]

- **Distancia de Manhattan**: Suma de las diferencias absolutas entre los valores correspondientes de dos puntos.

  \[
  d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
  \]

- **Distancia de Hamming**: Para variables categóricas, mide la diferencia entre dos vectores considerando cada componente de manera independiente.

### Ventajas de la imputación por KNN:

1. **Aprovecha la información global**: Utiliza toda la información disponible en el conjunto de datos para imputar valores, en lugar de limitarse a una sola columna.
   
2. **Flexibilidad**: Funciona tanto para variables numéricas como categóricas (ajustando el método de imputación según el tipo de variable).

3. **No requiere suposiciones fuertes**: A diferencia de la interpolación o métodos paramétricos, no asume una estructura particular de los datos.

### Desventajas de la imputación por KNN:

1. **Costo computacional**: Puede ser costoso en términos de tiempo y recursos computacionales, especialmente en grandes conjuntos de datos, ya que requiere calcular distancias entre todas las observaciones.
   
2. **No siempre es adecuado para datos escasos**: Si los valores faltantes son muchos o si los datos son dispersos, KNN puede no ser eficaz ya que los vecinos podrían no estar suficientemente cerca o ser representativos.

3. **Sensibilidad a la elección de \( K \)**: El número de vecinos (\( K \)) puede afectar significativamente los resultados, y encontrar el \( K \) óptimo puede requerir prueba y error.

4. **Escalado de los datos**: Las diferencias en las escalas de las variables pueden afectar los resultados, por lo que es necesario normalizar o estandarizar los datos antes de aplicar KNN.

### Consideraciones adicionales:

- **Estandarización de los datos**: Es importante que los datos estén en la misma escala, ya que las distancias se ven afectadas por la magnitud de las variables.
  
  ```python
  from sklearn.preprocessing import StandardScaler

  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(df)
  ```

- **Elección de \( K \)**: \( K \) puede determinarse mediante técnicas de validación cruzada o seleccionando el \( K \) que minimice el error de predicción en un conjunto de validación.

### Pasos para imputación por k-Nearest-Neighbors

Para cada observación con valores faltantes:

1. Encuentra otras K observaciones (donadores, vecinos) que sean más similares a esa observación.
2. Reemplaza los valores faltantes con los valores agregados de los K vecinos.

### ¿Cómo determinar cuáles son los vecinos más similares?

Cuantificación de distancia: distancia euclidiana útil para variables numéricas.

Distancia Manhattan útil para variables tipo factor.

Distancia de Hamming útil para variables categóricas

distancia de Gower útil para conjuntos de datos con variables mixtas

- **Euclidiana**: Útil para variables numéricas
- **Manhattan**: Útil paa variables tipo factor
- **Hamming**: Útil para variables categóricas
- **Gower**: Útil para conjuntos de datos con variables mixtas

### Conclusión:

La imputación por KNN es un método poderoso para manejar valores faltantes en conjuntos de datos complejos. Al basarse en las observaciones cercanas, permite realizar imputaciones coherentes con los patrones observados en los datos. Sin embargo, es importante considerar su costo computacional y el impacto de la elección de \( K \) para garantizar buenos resultados.

## Imputación por KNN en Python

La **imputación por KNN** en Python se puede realizar de manera efectiva utilizando la clase `KNNImputer` de la librería `scikit-learn`. Esta herramienta es útil para reemplazar los valores faltantes basándose en las observaciones más cercanas en términos de distancia entre puntos.

### Pasos para implementar KNNImputer en Python:

1. **Instalación de las dependencias necesarias** (si aún no las tienes instaladas):

   ```bash
   pip install scikit-learn pandas
   ```

2. **Imputación por KNN** con un ejemplo práctico.

#### Ejemplo paso a paso:

```python
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Crear un DataFrame con valores faltantes
data = {'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, np.nan, 3, 2],
        'C': [7, 8, 9, 10, 11]}

df = pd.DataFrame(data)

# Mostrar el DataFrame original con valores faltantes
print("DataFrame original:")
print(df)

# Crear un objeto KNNImputer con K=2 (número de vecinos más cercanos)
imputer = KNNImputer(n_neighbors=2)

# Imputar los valores faltantes utilizando KNN
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Mostrar el DataFrame después de la imputación
print("\nDataFrame imputado por KNN:")
print(df_imputed)
```

### Explicación del código:

1. **DataFrame con valores faltantes**: Creamos un `DataFrame` con algunas celdas vacías (representadas por `np.nan`).
2. **KNNImputer**: Inicializamos el objeto `KNNImputer` con 2 vecinos más cercanos (`n_neighbors=2`). Puedes ajustar este valor dependiendo de cuántos vecinos desees utilizar.
3. **Imputación**: Aplicamos el método `fit_transform()` para realizar la imputación de los valores faltantes.
4. **Resultados**: Visualizamos el DataFrame con los valores imputados.

### Salida esperada:

```
DataFrame original:
     A    B   C
0  1.0  5.0   7
1  2.0  NaN   8
2  NaN  NaN   9
3  4.0  3.0  10
4  5.0  2.0  11

DataFrame imputado por KNN:
     A    B     C
0  1.0  5.0   7.0
1  2.0  4.0   8.0
2  3.0  4.0   9.0
3  4.0  3.0  10.0
4  5.0  2.0  11.0
```

### Consideraciones adicionales:

- **Escalado de los datos**: Si los datos tienen escalas muy diferentes, es recomendable normalizarlos antes de aplicar KNNImputer para que las variables no dominen en el cálculo de las distancias.
  
  ```python
  from sklearn.preprocessing import StandardScaler

  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(df)

  # Aplicar KNN después de escalar los datos
  df_imputed_scaled = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df.columns)
  ```

- **Elección del número de vecinos (\( K \))**: El número de vecinos a utilizar puede variar según el tipo de datos y la cantidad de valores faltantes. Generalmente, se prueba con distintos valores de \( K \) y se evalúa cuál proporciona mejores resultados para el conjunto de datos.

La imputación por KNN es útil cuando los valores faltantes están relacionados con otras observaciones cercanas en el espacio de características, proporcionando una forma eficiente de imputar datos faltantes sin introducir sesgos arbitrarios.

**Lecturas recomendadas**

[1.6. Nearest Neighbors — scikit-learn 1.1.2 documentation](https://platzi.com/home/clases/4197-datos-faltantes-imputacion/55407-imputacion-por-knn-en-python/#:~:text=1.6.%20Nearest%20Neighbors%20%E2%80%94%20scikit%2Dlearn%201.1.2%20documentation "1.6. Nearest Neighbors — scikit-learn 1.1.2 documentation")

## Introducción a la imputación basada en modelos

La **imputación basada en modelos** es una técnica avanzada utilizada para tratar valores faltantes en los conjuntos de datos mediante el uso de modelos predictivos. A diferencia de métodos simples como la imputación por media o mediana, los modelos predictivos buscan capturar relaciones complejas entre las variables, utilizando información disponible en otras variables para estimar los valores faltantes de manera más precisa.

### Conceptos clave

1. **Modelo predictivo**: Utiliza un algoritmo de aprendizaje automático para estimar los valores faltantes. Algunos de los modelos más comunes para este propósito incluyen regresión lineal, árboles de decisión, k-vecinos más cercanos (KNN), entre otros.
   
2. **Ventajas de la imputación basada en modelos**:
   - **Mayor precisión**: Puede capturar relaciones complejas entre las variables, lo que resulta en estimaciones más exactas que los métodos tradicionales.
   - **Flexibilidad**: Puede adaptarse a diferentes tipos de datos (numéricos, categóricos).
   
3. **Desventajas**:
   - **Mayor complejidad**: La implementación requiere un mayor conocimiento en machine learning.
   - **Riesgo de sobreajuste**: Si no se gestiona adecuadamente, el modelo puede ajustarse demasiado a los datos de entrenamiento, afectando su capacidad de generalización.

### Métodos comunes de imputación basada en modelos

#### 1. **Regresión Lineal (para variables numéricas)**:
   Este método ajusta un modelo de regresión con las observaciones disponibles y utiliza la relación entre las variables para predecir los valores faltantes.

   - **Ejemplo**: Si tienes un conjunto de datos sobre precios de casas, podrías usar variables como tamaño de la casa, número de habitaciones, y ubicación para predecir los precios faltantes utilizando una regresión lineal.

#### 2. **Árboles de Decisión**:
   Los árboles de decisión pueden manejar tanto variables numéricas como categóricas y encontrar patrones en los datos para imputar los valores faltantes.

#### 3. **Imputación por Random Forest**:
   Utiliza múltiples árboles de decisión para construir un modelo robusto y predecir los valores faltantes. Es particularmente efectivo cuando las relaciones entre las variables son complejas.

#### 4. **Imputación por MICE (Multiple Imputation by Chained Equations)**:
   Este método crea varios modelos iterativos para imputar los valores faltantes de manera secuencial. Cada variable con datos faltantes es imputada basándose en las demás, de manera cíclica. Este proceso se repite varias veces, y el resultado final es la media de las imputaciones.

### Implementación de la imputación basada en modelos en Python

A continuación, te muestro cómo realizar la imputación basada en un modelo de **regresión lineal** utilizando la librería `scikit-learn` y un conjunto de datos con valores faltantes.

#### Paso a paso:

1. **Instalar las dependencias**:

   ```bash
   pip install scikit-learn pandas numpy
   ```

2. **Ejemplo de imputación basada en regresión lineal**:

   ```python
   import pandas as pd
   import numpy as np
   from sklearn.linear_model import LinearRegression
   from sklearn.model_selection import train_test_split

   # Crear un DataFrame con valores faltantes
   data = {'A': [1, 2, 3, 4, 5],
           'B': [2, np.nan, 6, 8, 10],
           'C': [5, 7, np.nan, 9, 11]}

   df = pd.DataFrame(data)

   # Mostrar el DataFrame original
   print("DataFrame original con valores faltantes:")
   print(df)

   # Separar las observaciones con y sin valores faltantes en la columna 'B'
   df_complete = df[df['B'].notna()]
   df_missing = df[df['B'].isna()]

   # Definir las variables predictoras (features) y la variable objetivo (target)
   X_complete = df_complete[['A', 'C']]
   y_complete = df_complete['B']

   X_missing = df_missing[['A', 'C']]

   # Entrenar un modelo de regresión lineal con los datos completos
   model = LinearRegression()
   model.fit(X_complete, y_complete)

   # Predecir los valores faltantes
   df.loc[df['B'].isna(), 'B'] = model.predict(X_missing)

   # Mostrar el DataFrame con los valores imputados
   print("\nDataFrame con los valores imputados:")
   print(df)
   ```

### Explicación del código:
- **DataFrame con valores faltantes**: Se genera un `DataFrame` con algunos valores faltantes en la columna 'B'.
- **Modelo de regresión lineal**: Se entrena un modelo de regresión lineal utilizando las columnas 'A' y 'C' para predecir los valores de la columna 'B'.
- **Imputación**: Se utilizan las predicciones del modelo para reemplazar los valores faltantes en la columna 'B'.

### Salida esperada:

```
DataFrame original con valores faltantes:
   A     B     C
0  1   2.0   5.0
1  2   NaN   7.0
2  3   6.0   NaN
3  4   8.0   9.0
4  5  10.0  11.0

DataFrame con los valores imputados:
   A     B     C
0  1   2.0   5.0
1  2   4.0   7.0
2  3   6.0   NaN
3  4   8.0   9.0
4  5  10.0  11.0
```

### Consideraciones finales:
- Es importante evaluar la calidad de las imputaciones generadas por el modelo para evitar sesgos.
- Si bien la imputación basada en modelos ofrece mayor precisión que los métodos simples, también puede introducir error si el modelo predictivo no captura bien las relaciones subyacentes entre las variables.

### Conclusión:

La **imputación basada en modelos** es una herramienta poderosa para manejar valores faltantes, especialmente en escenarios donde hay relaciones complejas entre las variables. Al aprovechar modelos de machine learning como regresión lineal, árboles de decisión o random forests, se puede mejorar la precisión de los análisis sin descartar datos importantes.

## Imputaciones Múltiples por Ecuaciones Encadenadas (MICE)

La **Imputación Múltiple por Ecuaciones Encadenadas (MICE)** es una técnica avanzada para manejar valores faltantes en conjuntos de datos. En lugar de imputar un único valor para cada valor faltante, MICE genera múltiples imputaciones, proporcionando un rango de posibles valores. Esto es especialmente útil para reflejar la incertidumbre que surge al imputar datos.

### ¿Cómo funciona MICE?

MICE funciona creando varios conjuntos de datos imputados basados en modelos iterativos. El proceso general de MICE sigue estos pasos:

1. **Inicialización**: Se imputan los valores faltantes de forma preliminar con un método sencillo (por ejemplo, la media o la mediana).
2. **Iteración**: Para cada variable con valores faltantes, MICE ajusta un modelo predictivo basado en las demás variables del conjunto de datos. Luego, se imputan los valores faltantes de esa variable con el modelo ajustado.
3. **Repetición**: Este proceso se repite secuencialmente para todas las variables con valores faltantes hasta que las imputaciones convergen (no cambian significativamente entre iteraciones).
4. **Múltiples imputaciones**: Este ciclo se repite varias veces, generando diferentes conjuntos de datos con valores imputados. Al final, se puede analizar cada conjunto imputado por separado y luego combinar los resultados para obtener una estimación robusta.

### Ventajas de MICE

- **Captura la incertidumbre**: Al realizar múltiples imputaciones, MICE incorpora la variabilidad y el posible error en las imputaciones, reflejando una visión más realista de los datos.
- **Usa todas las variables disponibles**: Aprovecha todas las variables presentes en el conjunto de datos para predecir los valores faltantes.
- **Flexibilidad**: MICE puede manejar tanto variables numéricas como categóricas y ajustarse a diferentes tipos de datos.

### Implementación de MICE en Python

La librería `fancyimpute` o el módulo `IterativeImputer` de `scikit-learn` pueden usarse para aplicar MICE. Aquí te muestro cómo hacerlo con `IterativeImputer` de `scikit-learn`.

#### Paso a paso:

1. **Instalación de dependencias**:

   ```bash
   pip install scikit-learn pandas numpy
   ```

2. **Ejemplo de imputación con MICE**:

   ```python
   import pandas as pd
   import numpy as np
   from sklearn.experimental import enable_iterative_imputer
   from sklearn.impute import IterativeImputer

   # Crear un DataFrame con valores faltantes
   data = {'A': [1, 2, np.nan, 4, 5],
           'B': [5, np.nan, 6, 8, 10],
           'C': [7, 8, 9, 10, np.nan]}

   df = pd.DataFrame(data)

   # Mostrar el DataFrame original
   print("DataFrame original con valores faltantes:")
   print(df)

   # Crear el imputador MICE (IterativeImputer)
   imputer = IterativeImputer(max_iter=10, random_state=0)

   # Aplicar imputación MICE
   df_imputed = imputer.fit_transform(df)

   # Convertir el resultado de nuevo a un DataFrame
   df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

   # Mostrar el DataFrame imputado
   print("\nDataFrame con los valores imputados usando MICE:")
   print(df_imputed)
   ```

### Explicación del código:

- **DataFrame original**: El `DataFrame` contiene valores faltantes en varias columnas.
- **IterativeImputer**: Este imputador sigue el principio de MICE, ajustando un modelo para cada variable con valores faltantes de forma iterativa.
- **Transformación**: El método `fit_transform()` genera un nuevo conjunto de datos donde los valores faltantes han sido imputados iterativamente utilizando todas las demás variables.

### Salida esperada:

```
DataFrame original con valores faltantes:
     A     B     C
0  1.0   5.0   7.0
1  2.0   NaN   8.0
2  NaN   6.0   9.0
3  4.0   8.0  10.0
4  5.0  10.0   NaN

DataFrame con los valores imputados usando MICE:
     A     B     C
0  1.0   5.0   7.0
1  2.0   5.7   8.0
2  3.0   6.0   9.0
3  4.0   8.0  10.0
4  5.0  10.0   9.2
```

### Consideraciones:

- **Número de iteraciones**: El parámetro `max_iter` controla cuántas veces se repite el proceso de imputación. Si el número de iteraciones es bajo, las imputaciones pueden ser menos precisas.
- **Repetibilidad**: Se puede establecer un `random_state` para hacer las imputaciones reproducibles.
- **Manejo de incertidumbre**: En algunas implementaciones avanzadas, como `fancyimpute`, puedes generar múltiples conjuntos de datos imputados, reflejando la variabilidad.

### Conclusión:

MICE es una técnica poderosa que ofrece imputaciones precisas al aprovechar toda la información disponible y realizar múltiples imputaciones para reflejar la incertidumbre en los valores faltantes. Es útil en aplicaciones donde es crítico manejar de forma robusta los valores faltantes, como en análisis estadísticos o aprendizaje automático.

**Lecturas recomendadas**

[MICE algorithm to Impute missing values in a dataset](https://www.numpyninja.com/post/mice-algorithm-to-impute-missing-values-in-a-dataset "MICE algorithm to Impute missing values in a dataset")

## Transformación inversa de los datos

La **transformación inversa de los datos** es el proceso de revertir una transformación previamente aplicada a los datos para devolverlos a su escala o formato original. Esto es especialmente importante cuando has aplicado técnicas de preprocesamiento, como la normalización, estandarización, codificación o imputación, y necesitas interpretar los resultados o presentar los datos en su forma original.

### Ejemplos comunes de transformaciones y su inversión:

1. **Escalado y Normalización**:
   - Si aplicaste una transformación de escalado (por ejemplo, `MinMaxScaler` o `StandardScaler` de `scikit-learn`), la transformación inversa deshace el escalado para devolver los datos a su rango original.
   - La inversión de estos métodos es útil, por ejemplo, cuando los modelos de machine learning han sido entrenados en datos escalados, pero quieres interpretar los resultados en la escala original.

2. **Codificación de Variables Categóricas**:
   - Si aplicaste técnicas de codificación como **One-Hot Encoding** o **Label Encoding** para convertir categorías a números, la transformación inversa te permite volver a las etiquetas originales (valores categóricos).
   - Es importante para análisis interpretativo o reportes donde las etiquetas originales tienen más sentido que los números codificados.

3. **Transformaciones logarítmicas o polinómicas**:
   - Si aplicaste transformaciones logarítmicas a tus datos para mejorar la linealidad en modelos de regresión, la transformación inversa devuelve los datos a su forma original aplicando la función exponencial inversa.

4. **Imputación**:
   - En el caso de métodos de imputación, como el **IterativeImputer** o **KNNImputer**, la transformación inversa devuelve los datos imputados a su formato original (por ejemplo, si has imputado valores faltantes y luego deseas evaluar los datos imputados).

### Ejemplo práctico: Transformación inversa después de escalado

Supongamos que has escalado tus datos con `MinMaxScaler` de `scikit-learn` y ahora quieres revertir esta transformación.

#### Paso 1: Aplicar el escalado

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Datos originales
data = np.array([[1, 2], [2, 3], [4, 5], [6, 8]])

# Crear un escalador y ajustarlo a los datos
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

print("Datos escalados:")
print(scaled_data)
```

#### Paso 2: Aplicar la transformación inversa

```python
# Invertir la transformación
original_data = scaler.inverse_transform(scaled_data)

print("Datos originales (después de la transformación inversa):")
print(original_data)
```

### Resultado esperado:

```plaintext
Datos escalados:
[[0.         0.        ]
 [0.2        0.2       ]
 [0.6        0.6       ]
 [1.         1.        ]]

Datos originales (después de la transformación inversa):
[[1. 2.]
 [2. 3.]
 [4. 5.]
 [6. 8.]]
```

### Consideraciones:

- **Transformaciones no reversibles**: No todas las transformaciones son completamente reversibles. Por ejemplo, si eliminas datos o aplicas métodos que modifican los datos de manera no lineal (como algunas transformaciones logarítmicas sobre valores negativos), puede que no se recupere la información original con precisión.
  
- **Exactitud en la inversión**: En transformaciones numéricas, pequeñas diferencias pueden surgir debido a la precisión de los cálculos, pero estas son generalmente insignificantes en la práctica.

### Aplicaciones:

- **Interpretación de modelos**: Después de aplicar transformaciones para entrenar un modelo, las predicciones o los coeficientes del modelo deben ser interpretados en la escala original.
- **Visualización**: A menudo, es más fácil interpretar gráficos y resúmenes en la escala original de los datos.
- **Presentación de resultados**: La presentación de resultados generalmente se hace en términos que sean comprensibles para los no técnicos, por lo que la transformación inversa es esencial.