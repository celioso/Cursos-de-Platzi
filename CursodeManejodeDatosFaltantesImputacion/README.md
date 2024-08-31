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