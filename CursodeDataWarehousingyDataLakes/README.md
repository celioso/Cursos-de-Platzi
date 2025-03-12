# Curso de Data Warehousing y Data Lakes

## Data Warehouse y Data Lake: Gestión Inteligente de Datos

En el mundo del **Big Data** y la **analítica avanzada**, la gestión eficiente de datos es clave para la toma de decisiones empresariales. Dos arquitecturas ampliamente utilizadas para almacenar y analizar datos son el **Data Warehouse (DWH)** y el **Data Lake**. A continuación, exploraremos sus características, diferencias y cuándo utilizarlos.

### **📌 ¿Qué es un Data Warehouse?**  
Un **Data Warehouse** es un sistema de almacenamiento optimizado para **consultas y análisis** de datos estructurados. Se diseñó para **integrar datos de múltiples fuentes**, transformarlos en un formato uniforme y facilitar su análisis mediante herramientas de inteligencia de negocios (**BI**).  

### **🔹 Características principales:**  
✔ **Estructurado**: Los datos se almacenan en tablas con esquemas predefinidos.  
✔ **Optimizado para consultas**: Diseñado para análisis rápidos mediante SQL.  
✔ **Datos históricos**: Permite almacenar grandes volúmenes de información histórica.  
✔ **Procesamiento ETL (Extract, Transform, Load)**: Requiere transformar los datos antes de ingresarlos al almacén.  

### **🔹 Ejemplos de Data Warehouse en la nube:**  
✅ **Amazon Redshift**  
✅ **Google BigQuery**  
✅ **Microsoft Azure Synapse Analytics**  
✅ **Snowflake**

### **📌 ¿Qué es un Data Lake?**  
Un **Data Lake** es un repositorio de almacenamiento que permite guardar grandes volúmenes de datos en su **formato original** (estructurado, semiestructurado y no estructurado). Es ideal para **análisis avanzados, machine learning e inteligencia artificial**.  

### **🔹 Características principales:**  
✔ **Datos en crudo**: No requiere transformación previa (puede almacenar cualquier tipo de dato).  
✔ **Alta escalabilidad**: Permite almacenar grandes cantidades de información de diversas fuentes.  
✔ **Procesamiento ELT (Extract, Load, Transform)**: Los datos se transforman solo cuando se necesitan.  
✔ **Análisis avanzado**: Ideal para ciencia de datos, IA y machine learning.  

### **🔹 Ejemplos de Data Lake en la nube:**  
✅ **Amazon S3 + AWS Lake Formation**  
✅ **Google Cloud Storage + Dataproc**  
✅ **Azure Data Lake Storage**

### **📊 Diferencias clave entre Data Warehouse y Data Lake**  

| **Característica**       | **Data Warehouse** 🏢 | **Data Lake** 🌊 |
|-------------------------|---------------------|------------------|
| **Estructura**          | Altamente estructurado | No estructurado |
| **Tipo de datos**       | Estructurados (tablas SQL) | Cualquier formato (texto, imágenes, videos, JSON, etc.) |
| **Procesamiento**       | ETL (transformación previa) | ELT (transformación bajo demanda) |
| **Costo**               | Alto (almacenamiento optimizado pero costoso) | Bajo (almacenamiento masivo y económico) |
| **Escalabilidad**       | Limitada | Altamente escalable |
| **Casos de uso**        | Reportes, análisis de BI | Big Data, machine learning, análisis en tiempo real |

### **🧐 ¿Cuándo usar Data Warehouse vs. Data Lake?**  

✅ **Usa un Data Warehouse si…**  
🔹 Necesitas informes rápidos y análisis de negocio estructurados.  
🔹 Los datos son limpios, bien organizados y tienen una estructura fija.  
🔹 Requieres optimización para consultas SQL y herramientas de BI.  

✅ **Usa un Data Lake si…**  
🔹 Manejas grandes volúmenes de datos en formatos diversos.  
🔹 Requieres análisis avanzados con inteligencia artificial y machine learning.  
🔹 Necesitas almacenamiento flexible y escalable sin transformación previa.

### **🚀 Conclusión**  
Un **Data Warehouse** y un **Data Lake** no son excluyentes, sino **complementarios**. Muchas empresas implementan un enfoque híbrido llamado **"Lakehouse"**, donde combinan la flexibilidad del Data Lake con la eficiencia de consulta del Data Warehouse.  

📢 **El futuro de la gestión de datos radica en integrar ambas soluciones para maximizar el valor de la información y potenciar la toma de decisiones empresariales.** 🚀💡

### Resumen

### ¿Qué es un Data Warehouse?

En el corazón de la analítica de datos moderna se encuentra el Data Warehouse, un sistema centralizado y estructurado que permite almacenar grandes volúmenes de datos históricos. Estos datos provienen de diversas fuentes dentro de una organización, tales como ventas, inventario y marketing. Sus principales características e incentivos incluyen:

- **Organización Estructurada**: Los datos se disponen en tablas claramente definidas para facilitar el análisis.
- **Optimización**: Diseñado específicamente para realizar consultas rápidas y eficientes.
- **Ejemplos de Uso**: Permite responder a preguntas críticas para el negocio, tales como:
- ¿Cuáles son los productos más vendidos en cada región?
- ¿Qué días del año generamos más ingresos?
- ¿Cómo ha cambiado el comportamiento de nuestros clientes a lo largo del tiempo?

El enfoque estructurado de un Data Warehouse lo convierte en la herramienta ideal para escenarios donde se requiera la generación de informes financieros o análisis de actividades bien definidos.

### ¿Qué papel juega un Data Lake?

Cuando los datos llegan desordenados o en múltiples formatos, el Data Lake surge como la solución ideal. A diferencia del Data Warehouse, un Data Lake almacena datos en bruto, lo cual incluye aquellos estructurados, semiestructurados y no estructurados. Entre sus características más sobresalientes están:

- **Flexibilidad**: Almacena toda la información tal cual llega, sin transformaciones previas.
- Tipos de Datos: Maneja datos de fuentes diversas como archivos de texto, imágenes, videos, sensores en tiempo real, y redes sociales, en formatos como JSON o XML.
- Ideal para: Análisis avanzados como machine learning o proyectos de big data que requieren manipulación y procesamiento avanzado de datos.

Un Data Lake es indispensable para el análisis de contenido multimedia o registros de sensores, especialmente cuando los datos necesitan ser procesados por modelos de inteligencia artificial.

### ¿Cómo elijo entre un Data Warehouse y un Data Lake?

La elección entre un Data Warehouse y un Data Lake no es excluyente; de hecho, ambas soluciones se complementan y participan de forma sinérgica en la infraestructura de datos de las organizaciones. Aquí algunos puntos clave para considerar:

- **Necesidades del Negocio**: Si necesitas decisiones rápidas y reportes predefinidos, el Data Warehouse es lo adecuado. Pero si vives en un mundo de datos complejos y no estructurados, como videos y datos en tiempo real, un Data Lake es la elección correcta.
- **Contexto**: Tu decisión dependerá del tipo de análisis que debes realizar y de la flexibilidad requerida para trabajar con los datos.
- **Propósito Final**: En última instancia, y sin importar la herramienta que elijas, el objetivo siempre es convertir los datos en información valiosa que apoye la toma de decisiones estratégicas.

Este curso te dará las herramientas para distinguir cuándo utilizar cada uno, cómo integrarlos adecuadamente y garantizar la calidad y seguridad de tus datos a través de una sólida gobernanza. Contarás con una guía experta en servicios de RedChip y Amazon S3 de AWS, allanando tu camino hacia un manejo de datos más eficaz y transformador.

Recuerda, almacenar datos es solo el principio. La verdadera ventaja competitiva radica en organizarlos, comprenderlos y usar todo su potencial para agregar valor real a tu organización. ¡Adelante con el aprendizaje!

## Conceptos de Negocios y Stakeholders

En el mundo empresarial, comprender los **conceptos de negocio** y la **gestión de stakeholders** es esencial para el éxito de cualquier organización. En este artículo, exploraremos los principios fundamentales de los negocios y el papel clave de los stakeholders en la toma de decisiones estratégicas.

### **📌 ¿Qué es un Negocio?**  
Un **negocio** es una entidad que ofrece bienes o servicios con el objetivo de generar valor, satisfacer necesidades y obtener beneficios económicos.  

### **🔹 Características de un negocio:**  
✔ **Propósito**: Resuelve problemas o satisface necesidades del mercado.  
✔ **Oferta de valor**: Bienes o servicios que diferencian a la empresa.  
✔ **Clientes**: Personas o empresas que adquieren la oferta de valor.  
✔ **Modelo de ingresos**: Estrategia para generar ganancias.  

### **🔹 Tipos de negocios:**  
✅ **Empresas comerciales**: Venden productos (ej. supermercados, tiendas de ropa).  
✅ **Empresas de servicios**: Ofrecen soluciones intangibles (ej. consultorías, educación).  
✅ **Empresas manufactureras**: Producen bienes físicos (ej. fábricas de autos).  
✅ **Empresas tecnológicas**: Desarrollan software o hardware (ej. startups de IA).

### **📌 ¿Quiénes son los Stakeholders?**  
Los **stakeholders** (partes interesadas) son individuos o grupos que tienen un interés directo o indirecto en una empresa y pueden influir en su desempeño.  

### **🔹 Tipos de Stakeholders:**  

| **Stakeholder**        | **Descripción** | **Ejemplos** |
|------------------------|----------------|--------------|
| **Internos** | Personas dentro de la empresa | Empleados, directivos, accionistas |
| **Externos** | Personas fuera de la empresa pero afectadas por ella | Clientes, proveedores, gobierno, competencia |

### **🔹 Stakeholders Claves en una Empresa:**  
✔ **Accionistas**: Invierten en la empresa y esperan rentabilidad.  
✔ **Clientes**: Compran productos/servicios y determinan el éxito del negocio.  
✔ **Empleados**: Son el motor de la empresa y ejecutan las estrategias.  
✔ **Proveedores**: Suministran insumos esenciales para la operación.  
✔ **Gobierno y reguladores**: Establecen normativas y leyes que afectan la empresa.  
✔ **Competencia**: Influye en el posicionamiento y estrategias del negocio.

### **📊 Importancia de los Stakeholders en la Estrategia Empresarial**  

📢 **Gestionar bien a los stakeholders es clave para el crecimiento del negocio.**  

✅ **Beneficios de una buena relación con los stakeholders:**  
🔹 Mayor confianza y reputación en el mercado.  
🔹 Mejor toma de decisiones estratégicas.  
🔹 Identificación de oportunidades y riesgos en el entorno empresarial.  
🔹 Mayor fidelización de clientes y retención de talento.  

### **🚀 Conclusión**  
Un negocio exitoso no solo se enfoca en vender productos o servicios, sino también en construir relaciones sólidas con sus **stakeholders**. La clave está en **equilibrar los intereses de todas las partes** para generar crecimiento sostenible y valor a largo plazo. 💡📈

## ¿Qué es Data Warehouse?

Un **Data Warehouse** (almacén de datos) es un sistema especializado para almacenar, organizar y analizar grandes volúmenes de información de una empresa. Se utiliza para la **toma de decisiones estratégicas**, permitiendo consolidar datos de diferentes fuentes en un único repositorio estructurado.

### **🔹 Características Claves de un Data Warehouse**  

✔ **Integrado**: Unifica datos de diversas fuentes (ERP, CRM, bases de datos, etc.).  
✔ **Estructurado**: Organiza la información de manera optimizada para análisis.  
✔ **Histórico**: Almacena grandes volúmenes de datos a lo largo del tiempo.  
✔ **Optimizado para consultas**: Diseñado para análisis rápido y eficiente.

### **🔹 ¿Cómo Funciona un Data Warehouse?**  

1️⃣ **Extracción (ETL - Extract, Transform, Load)**: Se extraen datos desde múltiples fuentes.  
2️⃣ **Transformación**: Se procesan y limpian los datos para su integración.  
3️⃣ **Carga**: Los datos estructurados se almacenan en el Data Warehouse.  
4️⃣ **Consulta y análisis**: Se utilizan herramientas de BI (Business Intelligence) para generar reportes y dashboards.

### **🔹 Beneficios del Data Warehouse**  

✅ **Mejora la toma de decisiones** con datos precisos y actualizados.  
✅ **Consolidación de información** en un solo repositorio confiable.  
✅ **Optimización del rendimiento** para consultas complejas.  
✅ **Facilita el análisis de tendencias** y pronósticos estratégicos.  

### **📌 Ejemplos de Data Warehouses en AWS**  

🔹 **Amazon Redshift**: Solución escalable y rápida para análisis de datos masivos.  
🔹 **Google BigQuery**: Plataforma serverless para análisis en la nube.  
🔹 **Snowflake**: Data Warehouse flexible con arquitectura separada de almacenamiento y cómputo.

### **🚀 Conclusión**  
El Data Warehouse es **clave para empresas orientadas a datos**, permitiendo analizar información de manera estructurada y eficiente. Su integración con herramientas de **BI y Machine Learning** lo hace fundamental para la **inteligencia empresarial y la innovación tecnológica**. 💡📊

## Arquitectura de un Data Warehouse

La arquitectura de un **Data Warehouse (DW)** define la forma en que los datos son extraídos, almacenados y analizados para la toma de decisiones estratégicas. Se compone de múltiples capas que garantizan **integración, almacenamiento y acceso eficiente** a los datos.

### **🔹 Capas de un Data Warehouse**  

### **1️⃣ Capa de Fuentes de Datos (Data Sources)**
📌 Contiene los sistemas de origen de donde se extraen los datos. Puede incluir:  
✔ Bases de datos transaccionales (SQL, NoSQL)  
✔ ERP, CRM, sistemas contables  
✔ APIs, archivos CSV, IoT, logs de servidores

### **2️⃣ Capa de Integración y Procesamiento (ETL - Extract, Transform, Load)**
📌 Procesa y transforma los datos antes de almacenarlos en el Data Warehouse.  
✔ **Extracción**: Se obtienen datos desde múltiples fuentes.  
✔ **Transformación**: Se limpia, normaliza y da formato a los datos.  
✔ **Carga**: Los datos transformados se almacenan en el Data Warehouse.  

🔹 Herramientas ETL: **AWS Glue, Apache Nifi, Talend, Informatica PowerCenter**

### **3️⃣ Capa de Almacenamiento (Data Warehouse)**
📌 Aquí se almacenan los datos estructurados de manera optimizada para análisis.  
✔ Se organiza en modelos **estrella o copo de nieve**  
✔ Almacena datos históricos y facilita consultas rápidas  
✔ Utiliza almacenamiento en columnas para mayor rendimiento  

🔹 Ejemplos de DW en la nube:  
✔ **Amazon Redshift**  
✔ **Google BigQuery**  
✔ **Snowflake**

### **4️⃣ Capa de Procesamiento y Análisis (OLAP - Online Analytical Processing)**
📌 Permite realizar análisis avanzados sobre los datos almacenados.  
✔ Motor de consultas optimizado para reportes y dashboards  
✔ Soporta agregaciones, filtros y modelado de datos  
✔ Permite exploración multidimensional de los datos  

🔹 Herramientas OLAP: **Microsoft SSAS, Apache Kylin, SAP BW**

### **5️⃣ Capa de Presentación (BI - Business Intelligence)**
📌 Proporciona visualización y generación de reportes para la toma de decisiones.  
✔ Dashboards interactivos  
✔ Reportes dinámicos  
✔ Análisis de tendencias y predicciones  

🔹 Herramientas BI: **Tableau, Power BI, Looker, AWS QuickSight**

### **🔹 Tipos de Arquitectura de Data Warehouse**  

🔹 **Mononivel (Single-Tier)**: Datos almacenados en una única base, ideal para pequeñas empresas.  
🔹 **Dos niveles (Two-Tier)**: Separa almacenamiento de datos y análisis, usado en medianas empresas.  
🔹 **Tres niveles (Three-Tier)**: Modelo más robusto con **fuentes de datos → Data Warehouse → BI**, ideal para grandes empresas.

### **🚀 Conclusión**  
Una **arquitectura bien diseñada** de Data Warehouse garantiza **integridad, rendimiento y escalabilidad** en el análisis de datos. Se integra con herramientas ETL, OLAP y BI para ofrecer **insights estratégicos** en cualquier organización. 💡📊

## Modelos de Datos en Data Warehouse 

Los modelos de datos en un **Data Warehouse (DW)** definen la manera en que la información se organiza, almacena y accede para facilitar el análisis eficiente. Existen diferentes enfoques para estructurar los datos, dependiendo de las necesidades de la empresa y la complejidad del análisis requerido.

### **🔹 Tipos de Modelos de Datos en Data Warehouse**  

### **1️⃣ Modelo Conceptual**  
📌 Representa una vista de alto nivel de los datos, sin detalles técnicos.  
✔ Enfocado en **entidades** y **relaciones**  
✔ No define estructuras físicas ni tipos de datos  
✔ Utilizado para la planificación y diseño inicial del DW  

🔹 **Ejemplo:** Un banco podría definir las entidades **Cliente**, **Cuenta**, **Transacción**, sin preocuparse por cómo se almacenan.

### **2️⃣ Modelo Lógico**  
📌 Representa la estructura de los datos en términos técnicos pero sin especificar la implementación física.  
✔ Define **tablas, atributos y relaciones**  
✔ Usa conceptos como claves primarias y foráneas  
✔ Puede basarse en modelos **relacionales o multidimensionales**  

🔹 **Ejemplo:**  
✔ Una tabla **Clientes** con campos `ID_Cliente`, `Nombre`, `Fecha_Registro`  
✔ Relación con la tabla **Cuentas** a través del `ID_Cliente`

### **3️⃣ Modelo Físico**  
📌 Representa la implementación real del DW en la base de datos.  
✔ Esquema detallado con **tipos de datos, particionamiento, índices**  
✔ Optimizado para mejorar la velocidad de consultas  
✔ Depende de la tecnología utilizada (Redshift, Snowflake, BigQuery)  

🔹 **Ejemplo:**  
✔ `Clientes(ID_Cliente INT PRIMARY KEY, Nombre VARCHAR(255), Fecha_Registro DATE)`  
✔ Indexación en `ID_Cliente` para mejorar la búsqueda

### **🔹 Modelos de Almacenamiento en Data Warehouse**  

### **1️⃣ Modelo Relacional (OLTP)**  
📌 Basado en bases de datos relacionales tradicionales.  
✔ Usa **tablas normalizadas** para evitar redundancia  
✔ Óptimo para transacciones rápidas, no para análisis  

🔹 **Ejemplo:** Un ERP con tablas altamente normalizadas: `Clientes`, `Pedidos`, `Productos`

### **2️⃣ Modelo Multidimensional (OLAP)**  
📌 Diseñado para análisis eficiente y consultas rápidas.  
✔ Se basa en **cubos de datos** con dimensiones y métricas  
✔ Optimizado para reportes, dashboards y análisis de tendencias  

🔹 **Ejemplo:**  
✔ **Dimensiones**: Tiempo, Producto, Ubicación  
✔ **Métricas**: Ventas Totales, Cantidad Vendida  

📌 **Esquemas comunes en OLAP:**  
✔ **Modelo Estrella**: Una tabla de hechos conectada a múltiples tablas de dimensiones.  
✔ **Modelo Copo de Nieve**: Similar al modelo estrella, pero las dimensiones están normalizadas.  
✔ **Modelo Constelación de Factos**: Múltiples tablas de hechos compartiendo dimensiones.

### **🚀 Conclusión**  
El **modelo de datos en un Data Warehouse** debe elegirse según el propósito del análisis. Para transacciones estructuradas, un modelo **relacional** es ideal. Para análisis avanzados y visualización, un **modelo multidimensional (OLAP)** es la mejor opción. La clave es **optimizar la estructura** para consultas rápidas y eficientes. 🔍📈

### Resumen

### ¿Cuáles son los modelos más populares para la arquitectura de un data warehouse?

Existen diversos modelos de datos que se utilizan para diseñar arquitecturas efectivas en un data warehouse. Cada uno de estos modelos tiene su singularidad y se adapta a diferentes necesidades de almacenamiento y consulta. Aprender sobre estos te dotará de las herramientas necesarias para elegir el que mejor se adapte a tus necesidades. Vamos a explorar algunos de los más comunes: el modelo estrella, el modelo copo de nieve y el modelo de constelación.

### ¿En qué consiste el modelo estrella?

El modelo estrella es uno de los más utilizados a la hora de definir la arquitectura de un data warehouse. Su estructura central la constituye una tabla que puede ser reconocida como la tabla de hechos o tabla puente. Esta tabla central recoge los datos transaccionales y se conecta con varias tablas de dimensiones a través de claves foráneas.

Por ejemplo, podrías tener una tabla de ventas en el centro que se vincule a otras tablas de dimensiones como clientes, productos o países. Este diseño te permite realizar consultas eficientes al concentrar las métricas clave en la tabla central y las cualidades descriptivas en las dimensiones.

### ¿Qué caracteriza al modelo copo de nieve?

El modelo copo de nieve es una extensión del modelo estrella. Mantiene la misma estructura central, pero las tablas de dimensiones están normalizadas. Esto significa que las tablas de dimensiones pueden tener otras dimensiones adicionales, como tablas padre e hijo. Cuando los datos necesitan normalización, el modelo de copo de nieve se convierte en una opción útil.

Este modelo se utiliza comúnmente para reducir la redundancia de datos y simplificar las actualizaciones del sistema cuando hay necesidad de detallar más los atributos alojados en las tablas de dimensión.

### ¿Qué ofrece el modelo de constelación?

El modelo de constelación, también conocido como modelo de galaxia, integra múltiples esquemas estrella dentro de un único modelo. Esto se traduce en que varias tablas de hechos pueden compartir tablas de dimensiones, permitiendo realizar análisis más complejos y multifacéticos. Este modelo es particularmente útil en escenarios donde se requiere analizar múltiples procesos o temas interrelacionados.

Por ejemplo, supón que tienes datos de ventas, clientes y productos junto con detalles de un calendario. Con un modelo de constelación, podrías realizar análisis que crucen información de diferentes áreas, integrando múltiples aspectos de los datos en un solo marco de consulta.

### ¿Cómo modelar tus datos para maximizar el rendimiento?

Modelar correctamente tus datos es crucial para maximizar el rendimiento y la eficiencia en la consulta y procesamiento de datos. Este modelo no solo define la estructura de los datos sino también determina cómo se pueden vincular e interaccionar entre sí.

### ¿Qué factores considerar para un buen modelado de datos?

- **Normalización y Desnormalización**: Clarifica cómo estos conceptos pueden afectar tu estructura de datos y por qué deberías aplicarlos en diferentes escenarios.

- **Identificación de Claves Primarias y Foráneas**: Asegúrate de definir correctamente estas claves para facilitar relaciones claras y eficaces entre distintas tablas.

- **Uso de Índices**: Con un alto volumen de datos, el uso de índices es altamente recomendable para optimizar el rendimiento de las consultas.

- **Granularidad Adecuada**: Determina hasta qué punto normalizar los datos es óptimo para mantener un balance entre el detalle del modelo y la performance del mismo.

- **Evaluación de Dimensiones**: Selecciona adecuadamente las entidades y atributos que integrarán cada dimensión para asegurar que cumplan con los objetivos analíticos deseados.

El construir adecuadamente un data warehouse sólido y funcional no solo facilita un análisis eficiente de los datos, sino que también respalda decisiones estratégicas en tiempo real. Ajusta tu modelado según las necesidades específicas de tu organización para maximizar los beneficios.

**Archivos de la clase**

[ventas.xlsx](https://static.platzi.com/media/public/uploads/ventas_a869ba16-c2b3-4ce0-ab59-445c775fb19d.xlsx)

## Introducción al Modelo de Entidad Relacional (ER)

El **Modelo de Entidad-Relación (ER)** es una técnica utilizada en la **modelación de bases de datos** para representar gráficamente los datos y sus relaciones antes de implementarlos en un sistema de gestión de bases de datos (DBMS).  

Se basa en tres componentes principales: **entidades, atributos y relaciones**, y se representa mediante **diagramas ER**, que ayudan a diseñar bases de datos de manera estructurada y comprensible.

### **🔹 Componentes del Modelo ER**  

### **1️⃣ Entidades**  
📌 Representan objetos o conceptos del mundo real con existencia propia en el modelo de datos.  
✔ Pueden ser **entidades fuertes** (independientes) o **entidades débiles** (dependen de otra entidad).  

🔹 **Ejemplo:**  
✔ Una empresa puede tener entidades como **Empleado**, **Departamento**, **Proyecto**.

### **2️⃣ Atributos**  
📌 Representan las propiedades o características de una entidad.  
✔ Tipos de atributos:  
   - **Simples o atómicos**: No pueden dividirse (Ej. `Nombre`, `Edad`).  
   - **Compuestos**: Se pueden descomponer (Ej. `Nombre Completo → Nombre + Apellido`).  
   - **Multivaluados**: Pueden tener más de un valor (Ej. `Teléfono`).  
   - **Derivados**: Se obtienen de otros atributos (Ej. `Edad` derivada de `Fecha de Nacimiento`).  
   - **Clave primaria**: Un atributo o conjunto de atributos que identifican de manera única a una entidad.  

🔹 **Ejemplo:**  
✔ La entidad **Empleado** podría tener atributos como `ID_Empleado (PK)`, `Nombre`, `Fecha_Nacimiento`, `Salario`.

### **3️⃣ Relaciones**  
📌 Representan asociaciones entre dos o más entidades.  
✔ Se caracterizan por su **cardinalidad**, que define cuántas instancias de una entidad pueden estar relacionadas con otra.  

📌 **Tipos de cardinalidad:**  
✔ **1:1 (Uno a Uno)** → Un empleado tiene un solo puesto y un puesto pertenece a un solo empleado.  
✔ **1:N (Uno a Muchos)** → Un departamento puede tener varios empleados, pero un empleado solo pertenece a un departamento.  
✔ **M:N (Muchos a Muchos)** → Un estudiante puede estar en varios cursos y un curso puede tener varios estudiantes.  

🔹 **Ejemplo:**  
✔ La relación **"Trabaja en"** entre las entidades **Empleado** y **Departamento** sería de tipo **1:N**.

### **🔹 Representación Gráfica (Diagrama ER)**  
Los **diagramas ER** utilizan los siguientes símbolos:  
✔ **Rectángulos** → Representan entidades.  
✔ **Óvalos** → Representan atributos.  
✔ **Rombos** → Representan relaciones.  
✔ **Líneas** → Conectan entidades con relaciones.  

📌 **Ejemplo:**  
Si tenemos un sistema que gestiona empleados y departamentos, podríamos representar:  

🔹 **Empleado** (*ID_Empleado, Nombre, Cargo, Salario*)  
🔹 **Departamento** (*ID_Departamento, Nombre_Departamento*)  
🔹 **Relación:** Un **Empleado** "Trabaja en" un **Departamento**

### **🚀 Conclusión**  
El **Modelo ER** es una herramienta fundamental en el diseño de bases de datos, ya que permite visualizar la estructura y las relaciones antes de su implementación. Facilita la creación de **modelos lógicos** y la transformación hacia **bases de datos relacionales**. 💾📊

### Resumen

### ¿Qué es el modelo entidad-relación?

El modelo entidad-relación es una herramienta fundamental para el diseño de bases de datos, permitiendo esquematizar a través de simbología objetos del mundo real. Este diagrama es esencial para comprender cómo interactúan y se relacionan diversas entidades dentro de un sistema, facilitando el posterior diseño del esquema de una base de datos.

### Simbología básica del modelo

El modelo entidad-relación utiliza una simbología específica para representar distintos elementos:

- **Rectángulo**: Representa una entidad (como una tabla en una base de datos) y es el conjunto que agrupa las características principales de un objeto del mundo real.
- **Elipse**: Indica los atributos o columnas de una entidad, ofreciendo detalles adicionales sobre el objeto representado por el rectángulo.
- **Rombo**: Simboliza las relaciones entre las diferentes entidades, destacando cómo interactúan o se conectan.
- **Línea**: Conecta entidades y atributos, marcando las relaciones finales entre ellos.

### Ejemplos prácticos

Examinar un ejemplo ayuda a clarificar cómo se aplica este modelo. Supongamos una relación donde un autor escribe un libro, un libro tiene ejemplares, y un usuario puede sacar ejemplares. Aquí, los rectángulos marcan entidades como "autor", "libro" y "usuario", mientras que las elipses representan atributos como código, nombre del autor, título del libro, etc.

Los atributos específicos, como los códigos identificatorios marcados, son clave para entender la estructura de la base de datos. Cuando el código de una entidad está subrayado o resaltado, generalmente significa que es su clave primaria, esencial para identificar de manera única cada registro.

### ¿Cómo crear un diagrama entidad-relación?

Para crear un diagrama entidad-relación, es recomendable usar herramientas como app.diagramas.net. Supongamos que queremos diseñar un diagrama para tres entidades: estudiante, asignatura y profesor.

### Pasos para el diseño

1. **Definir las entidades**: Se representan en rectángulos; en este ejemplo, son estudiante, asignatura y profesor.
2. **Establecer relaciones**: Utilizar rombos para describir cómo se conectan las entidades, como que un estudiante cursa una asignatura y un profesor imparte una asignatura.
3. **Agregar atributos**: Cada entidad tiene atributos representados por elipses. En el caso del estudiante, los atributos pueden ser ID, nombre y apellido.

### Claves Primarias y Foráneas

- **Primary key**: Un atributo que identifica de forma única los registros de una entidad. Por ejemplo, el ID del estudiante.
- **Foreign key**: Claves que permiten establecer relaciones entre entidades, como el ID de profesor relacionado con la asignatura que imparte.

En el ejemplo, para la entidad "asignatura", se incluirían tanto el ID único de la asignatura como las claves foráneas que son el ID de estudiante y el ID de profesor, garantizando la vinculación correcta entre entidades y estableciendo un diagrama bien estructurado.

### ¿Cómo aplicar el modelo a una situación real?

Es importante aplicar el conocimiento teórico a casos prácticos para consolidar el aprendizaje. Imagina que trabajas en una tienda que vende productos. El modelo entidad-relación te ayudará a entender y visualizar la interacción entre clientes, productos y ventas.

### Propuesta de ejercicio

- Crea un diagrama entidad-relación para representar la relación entre las entidades "cliente", "producto" y "venta".
- Incluye todos los atributos relevantes y establece las relaciones adecuadas usando simbología correcta.
- Puedes realizar el diagrama con la herramienta que prefieras y compartir un screenshot como parte del proceso de aprendizaje.

Esta práctica no solo reforzará tus habilidades en bases de datos, sino que te proporcionará una comprensión más profunda de cómo modelar datos de manera efectiva y precisa. ¡Buena suerte y sigue aprendiendo con entusiasmo!

**Archivos de la clase**

[ejercicio-diagrama-entidad-relacion.pdf](https://static.platzi.com/media/public/uploads/ejercicio-diagrama-entidad-relacion_9faff035-5252-4c92-b454-412fa5b4711f.pdf)

**Lecturas recomendadas**

[Flowchart Maker & Online Diagram Software](https://app.diagrams.net/)

## ¿Qué es Data Lake?

Un **Data Lake** es un **repositorio centralizado** que permite almacenar grandes volúmenes de datos en su formato original, ya sean **estructurados, semiestructurados o no estructurados**. A diferencia de un **Data Warehouse**, que organiza los datos en estructuras predefinidas, un Data Lake retiene los datos en su forma nativa hasta que se necesiten para análisis. 

### **🔹 Características Claves de un Data Lake**  

✔ **Almacenamiento en formato bruto:** No requiere preprocesamiento o transformación antes de ser almacenado.  
✔ **Alta escalabilidad:** Puede crecer para manejar **petabytes** de datos.  
✔ **Soporte para múltiples formatos:** JSON, CSV, imágenes, videos, logs, etc.  
✔ **Accesibilidad flexible:** Se puede consultar con SQL, Big Data frameworks (Apache Spark, Hadoop) o herramientas de Machine Learning.  
✔ **Integración con análisis avanzado:** Se usa para IA, ML, análisis en tiempo real y reporting.

### **🔹 Diferencia entre Data Lake y Data Warehouse**  

| Característica       | **Data Lake** | **Data Warehouse** |
|----------------------|--------------|--------------------|
| **Tipo de datos**    | Cualquier tipo de datos (estructurados, semiestructurados, no estructurados) | Solo datos estructurados y organizados |
| **Modelo de datos**  | Esquema flexible (Schema-on-read) | Esquema predefinido (Schema-on-write) |
| **Costo**           | Más económico en almacenamiento | Costoso debido al procesamiento y estructura |
| **Uso principal**   | Machine Learning, Big Data, Análisis en tiempo real | BI (Business Intelligence), Reporting |
| **Ejemplo de uso**  | Registro de clics en un sitio web, datos IoT, videos | Informes de ventas, análisis financiero |

### **🔹 Casos de Uso de un Data Lake**  

✅ **Análisis de Big Data:** Permite analizar grandes volúmenes de datos no estructurados.  
✅ **Machine Learning e Inteligencia Artificial:** Facilita el entrenamiento de modelos con datos diversos.  
✅ **Internet de las Cosas (IoT):** Almacena datos de sensores en tiempo real.  
✅ **Análisis de Registros (Logs):** Útil para monitoreo y seguridad cibernética.  
✅ **Almacén de datos históricos:** Guarda información a largo plazo sin transformación.

### **🔹 Ejemplos de Tecnologías Usadas en Data Lakes**  

✔ **AWS S3** (Simple Storage Service)  
✔ **Azure Data Lake Storage**  
✔ **Google Cloud Storage**  
✔ **Apache Hadoop / HDFS**  
✔ **Databricks Delta Lake**  

### **🚀 Conclusión**  
Un **Data Lake** es la solución ideal para organizaciones que manejan **grandes volúmenes de datos** y necesitan **flexibilidad en almacenamiento y análisis**. Permite que científicos de datos y analistas accedan a datos sin restricciones de estructura, fomentando la **innovación en análisis y Machine Learning**. 🚀📊

## ETL, ELT y ETLT

Los procesos **ETL, ELT y ETLT** son estrategias para extraer, transformar y cargar datos en un Data Warehouse o Data Lake. Cada enfoque tiene sus particularidades y se usa en diferentes contextos según la infraestructura, el volumen de datos y los requerimientos analíticos.

### **🔹 1. ¿Qué es ETL? (Extract, Transform, Load)**  

El **ETL (Extracción, Transformación y Carga)** es el método tradicional para integrar datos en un Data Warehouse. Consiste en:  

1️⃣ **Extract (Extraer):** Se obtienen datos desde diversas fuentes (bases de datos, APIs, archivos, etc.).  
2️⃣ **Transform (Transformar):** Se limpian, formatean y estructuran los datos para adecuarlos al análisis.  
3️⃣ **Load (Cargar):** Los datos transformados se almacenan en el Data Warehouse.  

### **📌 Características de ETL:**  
✔ Transformación antes de la carga  
✔ Ideal para Data Warehouses  
✔ Se usa en BI (Business Intelligence)  
✔ Procesamiento por lotes (Batch)  

### **📌 Tecnologías ETL Populares:**  
- Apache NiFi  
- Talend  
- Informatica PowerCenter  
- AWS Glue  
- Microsoft SSIS

### **🔹 2. ¿Qué es ELT? (Extract, Load, Transform)**  

El **ELT (Extracción, Carga y Transformación)** es una evolución de ETL, usada en **Data Lakes y Big Data**, donde la transformación ocurre después de cargar los datos.  

1️⃣ **Extract (Extraer):** Se extraen los datos sin modificaciones.  
2️⃣ **Load (Cargar):** Se almacenan en bruto en un Data Lake o Data Warehouse en la nube.  
3️⃣ **Transform (Transformar):** Se procesan dentro del sistema de destino cuando es necesario.  

### **📌 Características de ELT:**  
✔ Se carga primero, transformación bajo demanda  
✔ Ideal para Data Lakes y análisis de Big Data  
✔ Compatible con tecnologías en la nube  
✔ Usa almacenamiento escalable y barato  

### **📌 Tecnologías ELT Populares:**  
- Snowflake  
- Google BigQuery  
- AWS Redshift  
- Azure Synapse

### **🔹 3. ¿Qué es ETLT? (Extract, Transform, Load, Transform)**  

El **ETLT (Extracción, Transformación, Carga y Transformación adicional)** combina lo mejor de ETL y ELT.  

1️⃣ **Extract (Extraer):** Se obtienen datos de fuentes externas.  
2️⃣ **Transform (Transformar inicial):** Se realiza una limpieza básica antes de cargar.  
3️⃣ **Load (Cargar):** Se almacenan los datos en bruto en un Data Warehouse o Data Lake.  
4️⃣ **Transform (Transformación adicional):** Se realizan transformaciones avanzadas en el sistema de destino.  

### **📌 Características de ETLT:**  
✔ Primera transformación ligera antes de cargar  
✔ Segunda transformación dentro del Data Warehouse o Data Lake  
✔ Combinación ideal para Big Data y análisis en la nube  

### **📌 Tecnologías ETLT Populares:**  
- AWS Glue + Redshift  
- Apache Spark + Snowflake  
- Azure Data Factory + Synapse

### **🔹 Diferencias entre ETL, ELT y ETLT**  

| Característica | **ETL** | **ELT** | **ETLT** |
|--------------|--------|--------|--------|
| **Cuándo se transforman los datos** | Antes de cargar | Después de cargar | Antes y después de cargar |
| **Dónde se transforman los datos** | Servidor ETL | Dentro del Data Warehouse o Data Lake | Ambos (ETL + ELT) |
| **Velocidad** | Más lento | Más rápido (aprovecha la nube) | Equilibrado |
| **Usado en** | Data Warehouses tradicionales | Data Lakes y análisis en la nube | Big Data con necesidades mixtas |
| **Ejemplo de uso** | Informes financieros | Machine Learning, Big Data | Híbrido: BI + ML |

### **🚀 Conclusión**  

✔ **ETL** es ideal para **Data Warehouses**, donde se necesita calidad y estructura en los datos.  
✔ **ELT** es la mejor opción para **Big Data y Data Lakes**, con capacidad de almacenamiento masivo.  
✔ **ETLT** es un modelo híbrido para aprovechar las ventajas de ambos enfoques en entornos modernos.  

📊 **Elegir el modelo correcto depende de las necesidades del negocio, la infraestructura y el volumen de datos.** 🚀

## Data Lakehouse 

El **Data Lakehouse** es un enfoque moderno que combina lo mejor de los **Data Warehouses** y **Data Lakes**. Permite almacenar datos estructurados y no estructurados en un solo lugar, con capacidades avanzadas de procesamiento y análisis.

### **🔹 1. ¿Qué es un Data Lakehouse?**  

Un **Data Lakehouse** es una arquitectura híbrida que une:  
✔ **Data Warehouse** → Procesamiento analítico estructurado  
✔ **Data Lake** → Almacenamiento masivo de datos en bruto  

**Objetivo:** Brindar la escalabilidad y flexibilidad de un Data Lake, junto con el control y la eficiencia de un Data Warehouse.  

### **📌 Características Clave:**  
✅ **Almacena datos estructurados y no estructurados**  
✅ **Separa almacenamiento y cómputo** para mayor eficiencia  
✅ **Permite procesamiento en tiempo real y batch**  
✅ **Optimizado para Machine Learning y BI**  
✅ **Usa formatos abiertos como Parquet y ORC**

### **🔹 2. Comparación entre Data Warehouse, Data Lake y Data Lakehouse**  

| **Característica**  | **Data Warehouse**  | **Data Lake**  | **Data Lakehouse**  |
|----------------|----------------|-------------|---------------|
| **Tipo de datos** | Estructurados | No estructurados y estructurados | Ambos |
| **Costo** | Alto (almacenamiento y cómputo juntos) | Bajo (almacenamiento masivo) | Optimizado |
| **Escalabilidad** | Limitada | Alta | Alta |
| **Tiempo de procesamiento** | Rápido | Lento (procesamiento posterior) | Rápido |
| **Usabilidad** | BI y reportes | Big Data y ML | Ambos |
| **Formato de datos** | Tablas SQL | Archivos (JSON, Parquet, CSV, etc.) | Formatos abiertos (Parquet, Delta Lake) |

### **🔹 3. Beneficios de un Data Lakehouse**  

🚀 **Mayor flexibilidad** → Almacena todo tipo de datos en un solo sistema  
🚀 **Reducción de costos** → Separa almacenamiento y procesamiento  
🚀 **Mejor integración con Machine Learning** → Facilita análisis avanzado  
🚀 **Acceso a datos en tiempo real** → Procesamiento más rápido que en un Data Lake  
🚀 **Formatos abiertos y estandarizados** → Permite mayor interoperabilidad

### **🔹 4. Tecnologías Clave para un Data Lakehouse**  

### **📌 Plataformas Populares:**  
- **Databricks Lakehouse**  
- **Snowflake**  
- **Google BigQuery**  
- **AWS Lake Formation**  
- **Azure Synapse Analytics**  

### **📌 Tecnologías Relacionadas:**  
- **Delta Lake** (Formato transaccional sobre Data Lakes)  
- **Apache Iceberg** (Optimización de tablas en Data Lakes)  
- **Apache Hudi** (Gestión de datos en tiempo real en Data Lakes)

### **🔹 5. Casos de Uso de un Data Lakehouse**  

✔ **Análisis empresarial en tiempo real** → Reportes de BI con datos frescos  
✔ **Machine Learning e IA** → Entrenamiento de modelos directamente en los datos  
✔ **Procesamiento de Big Data** → Integración con streaming y procesamiento batch  
✔ **Gestión de datos en la nube** → Centralización de información para múltiples aplicaciones

### **🚀 Conclusión**  

El **Data Lakehouse** representa el futuro del almacenamiento de datos, uniendo lo mejor de los **Data Warehouses** y **Data Lakes**. Es la solución ideal para organizaciones que buscan escalabilidad, eficiencia y procesamiento avanzado para Business Intelligence y Machine Learning.

## Herramientas y Plataformas de Data Warehouse y Data Lake 

Las empresas utilizan **Data Warehouses (DW)** y **Data Lakes (DL)** para almacenar, procesar y analizar grandes volúmenes de datos. Existen diversas herramientas y plataformas en la nube y on-premise para cada enfoque.

### **📌 1. Herramientas y Plataformas de Data Warehouse**  

Los **Data Warehouses** permiten almacenar y analizar datos estructurados para Business Intelligence (BI) y reportes.  

### **🔹 Plataformas en la nube (Cloud Data Warehouses)**  
💠 **Amazon Redshift** → Data Warehouse en AWS con escalabilidad y alto rendimiento.  
💠 **Google BigQuery** → DW sin servidor, optimizado para análisis en tiempo real.  
💠 **Snowflake** → Plataforma flexible con separación de cómputo y almacenamiento.  
💠 **Microsoft Azure Synapse Analytics** → Análisis de datos empresariales en la nube de Microsoft.  
💠 **IBM Db2 Warehouse** → Optimizado para analítica y basado en IA.  

### **🔹 Plataformas On-Premise y Open Source**  
🔹 **Apache Hive** → Data Warehouse sobre Hadoop para consultas SQL en Big Data.  
🔹 **Greenplum** → DW basado en PostgreSQL para analítica de datos.  
🔹 **ClickHouse** → DW de código abierto optimizado para consultas en tiempo real.  
🔹 **Vertica** → Plataforma de almacenamiento masivo con alto rendimiento.

### **📌 2. Herramientas y Plataformas de Data Lake**  

Los **Data Lakes** almacenan datos en su formato original, estructurado o no estructurado, para Big Data, Machine Learning e IA.  

### **🔹 Plataformas en la nube (Cloud Data Lakes)**  
💠 **AWS Lake Formation** → Crea y administra un Data Lake en AWS.  
💠 **Google Cloud Storage + BigLake** → Unifica almacenamiento y análisis en Google Cloud.  
💠 **Azure Data Lake Storage (ADLS)** → Almacén escalable en la nube de Microsoft.  
💠 **IBM Cloud Object Storage** → Almacén distribuido para datos no estructurados.  

### **🔹 Tecnologías Open Source y Frameworks para Data Lakes**  
🔹 **Apache Hadoop HDFS** → Almacenamiento distribuido escalable para Big Data.  
🔹 **Apache Spark** → Procesamiento de datos distribuido en Data Lakes.  
🔹 **Apache Iceberg** → Gestión de tablas en Data Lakes con optimización transaccional.  
🔹 **Apache Hudi** → Procesamiento de datos en tiempo real en Data Lakes.  
🔹 **Delta Lake** → Extensión de Apache Spark con soporte para transacciones ACID.

### **📌 3. Plataformas Híbridas: Data Lakehouse**  

El **Data Lakehouse** combina lo mejor del Data Warehouse y Data Lake, permitiendo almacenar y procesar datos en un solo entorno.  

💠 **Databricks Lakehouse** → Basado en Apache Spark y Delta Lake.  
💠 **Snowflake** → Admite almacenamiento de datos estructurados y no estructurados.  
💠 **Google BigLake** → Unificación de Data Lakes y Data Warehouses.  
💠 **AWS Lake Formation + Redshift Spectrum** → Combinación de Data Lake y DW en AWS.

### **📌 4. Comparación de Plataformas por Casos de Uso**  

| **Plataforma** | **Tipo** | **Casos de Uso** |
|---------------|---------|----------------|
| Amazon Redshift | Data Warehouse | BI y reportes empresariales |
| Google BigQuery | Data Warehouse | Consultas SQL sobre Big Data |
| Snowflake | Data Warehouse/Lakehouse | Análisis multi-nube |
| Apache Hive | Data Warehouse Open Source | Data Warehousing en Hadoop |
| AWS Lake Formation | Data Lake | Centralización de datos en AWS |
| Apache Spark | Framework para Data Lakes | Procesamiento en tiempo real |
| Delta Lake | Data Lakehouse | Big Data con soporte transaccional |
| Databricks | Data Lakehouse | Machine Learning y Data Science |

### **🚀 Conclusión**  

La elección de la mejor herramienta o plataforma depende de los requerimientos de la empresa:  
✅ **Data Warehouse** → Ideal para BI, reportes y análisis estructurado.  
✅ **Data Lake** → Almacena datos en bruto para Big Data y Machine Learning.  
✅ **Data Lakehouse** → Híbrido, optimizado para análisis avanzado.  

Las empresas suelen combinar varias plataformas para optimizar costos y rendimiento. 🚀

## Business Intelligence (BI) y Niveles de Analítica de Datos 

### **🔹 ¿Qué es Business Intelligence (BI)?**  
**Business Intelligence (BI)** es el conjunto de estrategias, tecnologías y procesos que permiten transformar datos en información útil para la toma de decisiones empresariales.  

🔹 Recopila, almacena y analiza datos desde múltiples fuentes.  
🔹 Ayuda a identificar tendencias, patrones y oportunidades de negocio.  
🔹 Se usa para reportes, dashboards y análisis predictivo.  

🔹 **Ejemplos de herramientas BI:**  
💠 **Power BI** → Análisis interactivo y visualización de datos.  
💠 **Tableau** → Visualización de datos avanzada.  
💠 **Google Data Studio** → Reportes y dashboards en la nube.  
💠 **Qlik Sense** → BI con analítica asociativa.

### **📌 Niveles de Analítica de Datos**  

La analítica de datos en BI se divide en **cuatro niveles**, según la complejidad del análisis y el valor que aporta a la empresa:  

### **1️⃣ Analítica Descriptiva – ¿Qué pasó?**  
🔹 Resume y organiza los datos históricos.  
🔹 Se basa en reportes, dashboards y métricas clave.  
🔹 Identifica patrones y tendencias en datos pasados.  

**📌 Ejemplo:**  
Un dashboard en Power BI muestra las ventas mensuales de una empresa por región.

### **2️⃣ Analítica Diagnóstica – ¿Por qué pasó?**  
🔹 Profundiza en los datos para encontrar causas y relaciones.  
🔹 Usa técnicas de segmentación, correlación y drill-down.  
🔹 Requiere herramientas como SQL, Python o R para análisis más avanzados.  

**📌 Ejemplo:**  
Un análisis muestra que las ventas bajaron en una región debido a problemas en la cadena de suministro.

### **3️⃣ Analítica Predictiva – ¿Qué pasará?**  
🔹 Usa modelos de Machine Learning y estadísticas para predecir eventos futuros.  
🔹 Identifica patrones en datos históricos para hacer pronósticos.  
🔹 Requiere herramientas como **Python (Scikit-learn), AWS Forecast, Azure ML**.  

**📌 Ejemplo:**  
Un modelo de predicción estima la demanda de productos en los próximos 6 meses basado en tendencias pasadas.

### **4️⃣ Analítica Prescriptiva – ¿Qué debo hacer?**  
🔹 Recomienda acciones óptimas basadas en análisis de datos.  
🔹 Utiliza simulaciones, optimización y algoritmos de IA.  
🔹 Requiere modelos avanzados y técnicas de optimización matemática.  

**📌 Ejemplo:**  
Un algoritmo sugiere la mejor estrategia de precios para maximizar ganancias según la demanda del mercado.

### **📊 Comparación de los Niveles de Analítica**  

| **Nivel** | **Pregunta clave** | **Ejemplo** | **Herramientas** |
|-----------|------------------|------------|---------------|
| 📊 **Descriptiva** | ¿Qué pasó? | Reporte de ventas mensuales | Power BI, Tableau |
| 📉 **Diagnóstica** | ¿Por qué pasó? | Análisis de disminución en ventas | SQL, Python |
| 📈 **Predictiva** | ¿Qué pasará? | Pronóstico de demanda de productos | AWS Forecast, Scikit-learn |
| 🤖 **Prescriptiva** | ¿Qué debo hacer? | Recomendaciones de precios dinámicos | Optimización en Python, IA |

### **🚀 Conclusión**  
✅ **BI** ayuda a las empresas a tomar mejores decisiones basadas en datos.  
✅ Cada nivel de analítica aporta más valor a la empresa.  
✅ La combinación de BI con **Machine Learning e IA** permite llegar a la analítica prescriptiva.

## Bases de Datos OLTP y OLAP

Las bases de datos pueden clasificarse según su propósito en **OLTP (Online Transaction Processing)** y **OLAP (Online Analytical Processing)**. Cada una está diseñada para un tipo específico de carga de trabajo en los sistemas de información.

### **🖥️ OLTP (Procesamiento de Transacciones en Línea)**  

🔹 **Definición:** Son bases de datos diseñadas para manejar un gran número de transacciones en tiempo real.  
🔹 **Enfoque:** Procesamiento rápido de pequeñas transacciones.  
🔹 **Objetivo:** Garantizar la integridad y velocidad en la ejecución de transacciones.  

### **🔑 Características de OLTP:**  
✅ Alto volumen de transacciones concurrentes.  
✅ Accesos frecuentes a datos individuales (INSERT, UPDATE, DELETE).  
✅ Integridad de datos con restricciones ACID (Atomicidad, Consistencia, Aislamiento, Durabilidad).  
✅ Consultas cortas y optimizadas para rapidez.  

### **📌 Ejemplo de OLTP:**  
- **Bancos:** Registro de transacciones en cuentas bancarias.  
- **E-commerce:** Gestión de pedidos y pagos en línea.  
- **Sistemas de reservas:** Compra de boletos de avión en tiempo real.  

**🛠️ Herramientas y Tecnologías OLTP:**  
💠 **Bases de datos relacionales:** MySQL, PostgreSQL, SQL Server, Oracle.  
💠 **Sistemas NoSQL para OLTP:** MongoDB, Amazon DynamoDB, Firebase.

### **📊 OLAP (Procesamiento Analítico en Línea)**  

🔹 **Definición:** Son bases de datos optimizadas para la consulta y análisis de grandes volúmenes de datos históricos.  
🔹 **Enfoque:** Análisis de datos en múltiples dimensiones.  
🔹 **Objetivo:** Ayudar en la toma de decisiones estratégicas con información consolidada.  

### **🔑 Características de OLAP:**  
✅ Consultas complejas y agregaciones de datos (SUM, AVG, COUNT).  
✅ Manejo de grandes volúmenes de datos históricos.  
✅ Uso de modelos multidimensionales (cubos OLAP).  
✅ Integración con herramientas de **Business Intelligence (BI)**.  

### **📌 Ejemplo de OLAP:**  
- **Empresas de retail:** Análisis de tendencias de ventas por región.  
- **Finanzas:** Evaluación del desempeño financiero a lo largo del tiempo.  
- **Marketing:** Segmentación de clientes y predicción de comportamiento de compra.  

**🛠️ Herramientas y Tecnologías OLAP:**  
💠 **Data Warehouses:** Amazon Redshift, Google BigQuery, Snowflake.  
💠 **OLAP Engines:** Apache Druid, Microsoft Analysis Services.  
💠 **Herramientas de BI:** Power BI, Tableau, Looker.

### **🔍 Comparación OLTP vs. OLAP**  

| **Característica** | **OLTP** | **OLAP** |
|------------------|----------|----------|
| **Objetivo** | Procesar transacciones en tiempo real | Análisis de datos históricos |
| **Operaciones** | INSERT, UPDATE, DELETE | SELECT con agregaciones |
| **Usuarios** | Usuarios operacionales (cajeros, clientes, empleados) | Analistas, gerentes, directivos |
| **Modelo de Datos** | Normalización (3NF) para evitar redundancia | Desnormalización para optimizar consultas |
| **Tiempo de respuesta** | Milisegundos (rápido) | Segundos o minutos (procesamiento intensivo) |
| **Tamaño de Datos** | Gigabytes | Terabytes o petabytes |
| **Ejemplo** | Base de datos de reservas de vuelos | Data Warehouse de tendencias de ventas |

### **🚀 Conclusión**  

🔹 **OLTP** es ideal para operaciones en tiempo real y transacciones frecuentes.  
🔹 **OLAP** permite analizar grandes volúmenes de datos históricos para la toma de decisiones.  
🔹 Ambas arquitecturas pueden complementarse: **OLTP almacena datos operacionales, que luego se transforman y cargan en OLAP para análisis.**  

📌 **Ejemplo práctico:** Un e-commerce usa **OLTP** para registrar ventas en tiempo real y **OLAP** para analizar tendencias de compra y optimizar estrategias de marketing.

## ¿Cómo crear tu cuenta en AWS?

Para crear una cuenta en AWS, sigue estos pasos:

### **1. Ir al sitio web de AWS**
- Abre tu navegador y ve a [AWS Sign Up](https://aws.amazon.com/).

### **2. Hacer clic en "Crear una cuenta de AWS"**
- Si ya tienes una cuenta de Amazon, puedes iniciar sesión con ella, pero es recomendable crear una cuenta específica para AWS.

### **3. Ingresar información básica**
- **Correo electrónico** (Usa uno que revises con frecuencia).  
- **Nombre de usuario de la cuenta** (puedes usar tu nombre o el de tu empresa).  
- **Contraseña segura** (Debe tener al menos 8 caracteres, combinando mayúsculas, minúsculas, números y símbolos).  

### **4. Elegir tipo de cuenta**
- **Cuenta personal** (para proyectos individuales o aprendizaje).  
- **Cuenta comercial** (si la usarás para una empresa).  

### **5. Proporcionar información de contacto**
- Nombre completo  
- Dirección  
- Número de teléfono  
- País  

### **6. Agregar método de pago**
- AWS requiere una tarjeta de crédito o débito para verificar la cuenta.  
- Aunque AWS tiene una capa gratuita, algunos servicios pueden generar cargos si se exceden los límites gratuitos.  

### **7. Verificación telefónica**
- Recibirás un código de verificación vía SMS o llamada.  
- Ingresa el código para confirmar tu número.  

### **8. Elegir un plan de soporte**
- **Básico (Gratis)**: Ideal para la mayoría de los usuarios nuevos.  
- **Developer, Business o Enterprise**: Son de pago y ofrecen soporte técnico avanzado.  

### **9. Iniciar sesión en la Consola de AWS**
- Una vez creada la cuenta, inicia sesión en la [Consola de AWS](https://aws.amazon.com/console/).  

Después de la activación, puedes comenzar a explorar AWS y configurar tus servicios. 🚀

## ¿Qué es Redshift?

**Amazon Redshift** es un servicio de **almacenamiento de datos (Data Warehouse) en la nube** que permite analizar grandes volúmenes de información de manera rápida y eficiente. Está diseñado para consultas analíticas en bases de datos a gran escala, utilizando procesamiento en paralelo y almacenamiento columnar para optimizar el rendimiento.

### **Características principales de Redshift**  

✅ **Almacenamiento Columnar**: Organiza los datos en columnas en lugar de filas, lo que mejora la velocidad en consultas analíticas.  

✅ **Procesamiento en Paralelo Masivo (MPP)**: Distribuye la carga de trabajo en múltiples nodos para acelerar las consultas.  

✅ **Integración con Herramientas de BI**: Compatible con herramientas como Tableau, QuickSight, Looker y Power BI.  

✅ **Escalabilidad**: Permite aumentar o reducir el número de nodos según las necesidades del negocio.  

✅ **Seguridad**: Soporta cifrado, control de acceso con IAM y aislamiento de redes con Amazon VPC.  

✅ **Bajo Costo**: Ofrece pago por uso y la opción de instancias reservadas para ahorrar costos.

### **Casos de Uso**  

📊 **Análisis de Datos**: Empresas que manejan grandes volúmenes de datos para obtener insights y reportes.  

📈 **Big Data y Machine Learning**: Puede integrarse con servicios como AWS Glue, SageMaker y Data Lake.  

🏢 **Empresas de Retail, Finanzas y Salud**: Se usa para analizar tendencias de clientes, fraudes y reportes en tiempo real.

### **Diferencia entre Redshift y un Data Lake (Ejemplo con S3)**  

| **Característica**      | **Amazon Redshift** | **Amazon S3 (Data Lake)** |
|------------------------|--------------------|--------------------------|
| **Estructura**         | Datos estructurados y optimizados para consultas SQL | Datos en crudo, estructurados y no estructurados |
| **Consultas**          | Alta velocidad con SQL optimizado | Requiere herramientas externas como Athena |
| **Uso principal**      | Análisis de datos empresariales | Almacenamiento masivo y preprocesamiento de datos |

### **Conclusión**  

Amazon Redshift es una excelente opción para empresas que necesitan **consultas rápidas sobre grandes volúmenes de datos**, permitiendo tomar decisiones basadas en análisis de información en tiempo real. 

### Resumen

### ¿Qué es Amazon Redshift y cómo empezar a usarlo?

Amazon Redshift es un poderoso servicio de almacenamiento y análisis de datos en la nube de AWS, similar a un ecosistema de warehouse. Esta herramienta permite crear bases de datos, tablas y conectarse a clústeres, lo que ayuda a gestionar grandes volúmenes de datos de manera eficiente. Redshift es especialmente valorado por su capacidad para manejar cantidades masivas de datos y escalar según las necesidades del usuario.

###¿Cómo configurar tu cuenta de AWS para usar Redshift?

Para iniciar con Amazon Redshift, es esencial familiarizarse con la interfaz de AWS. Después de ingresar a tu cuenta de AWS:

- **Verifica la región**: Asegúrate de estar posicionado en la región de Virginia, ya que es la más económica en AWS para Estados Unidos.
- **Acceso a Redshift**: Escribe "Amazon Redshift" en la barra de búsqueda e ingresa al servicio.
- **Consulta la documentación**: La extensa documentación de Redshift es un recurso valioso que te ayudará a entender a fondo sus beneficios y características.

Es importante revisar aspectos de documentación y costos, ya que Redshift ofrece una parte gratuita basada en las horas de uso del clúster.

### ¿Cómo crear un clúster en Amazon Redshift?

La creación de un clúster es un paso clave para utilizar Redshift. Sigue estos pasos para configurar tu clúster:

- **Elige Redshift Serverless Free Trial**: Aquí podrás configurar las características del clúster.
- **Define el nombre de la instancia**: Por ejemplo, puedes asignarle "red shift curso platzi".
- **Crea una base de datos predeterminada**: Al crear el clúster, se genera automáticamente una base de datos llamada dev.
- **Configura las credenciales IAM**: Personaliza las credenciales de acceso, por ejemplo, con usuario "admin" y una contraseña segura.

### Código de ejemplo para creación de credenciales:

```shell
Usuario: admin
Contraseña: Platzi1234!
```

- **Configura el Work Group y capacidad del clúster**: Define los parámetros de cómputo que mejor se ajusten a tus necesidades. AWS crea automáticamente las configuraciones de red necesarias.

### ¿Cómo optimizar costos usando Amazon Redshift?

Optimizar costos en Amazon Redshift pasa por analizar métricas claves y ajustar configuraciones:

- **Métricas de uso**: Monitorea cuántas horas está activo tu clúster.
- **Performance de queries**: Analiza el rendimiento de tus consultas para identificar áreas de mejora.
- **Alertas de costo y presupuesto**: Establece alertas para no exceder tu presupuesto previsto.

Al estar al tanto de estas métricas, no solo mejorarás la eficiencia de tu clúster, sino que también podrás optimizar el presupuesto dedicado a AWS.

Con lo aprendido, lanza tu propio clúster de Redshift, explora sus capacidades y descubre cómo puede transformar la gestión de datos en tu organización. ¿Qué ventajas podrías encontrar al usar Redshift para tus necesidades de análisis de datos? Comparte tus reflexiones y experiencias para seguir mejorando juntos.

**Lecturas recomendadas**

[AWS | Solución de almacenamiento y análisis de datos en la nube](https://aws.amazon.com/es/redshift/)

## Conociendo Redshit

Amazon Redshift es un servicio de **Data Warehouse en la nube**, diseñado para manejar grandes volúmenes de datos y ejecutar consultas analíticas de manera rápida y eficiente. Es ideal para empresas que necesitan analizar información en tiempo real y tomar decisiones basadas en datos.

### **🔹 Características Principales**  

✅ **Almacenamiento Columnar**  
   - Organiza los datos en columnas en lugar de filas, lo que mejora la velocidad de consulta.  

✅ **Procesamiento en Paralelo Masivo (MPP)**  
   - Divide la carga de trabajo entre múltiples nodos para mejorar el rendimiento.  

✅ **Integración con Herramientas de BI**  
   - Compatible con herramientas como **Tableau, QuickSight, Power BI, Looker**, entre otras.  

✅ **Escalabilidad**  
   - Puedes agregar o reducir nodos fácilmente según las necesidades del negocio.  

✅ **Seguridad Avanzada**  
   - Soporta cifrado, control de acceso con **AWS IAM** y aislamiento de redes con **Amazon VPC**.  

✅ **Costo-Eficiente**  
   - Opciones de **pago por uso** y **instancias reservadas** para reducir costos.

### **🔹 Componentes de Redshift**  

🟠 **Cluster**: Es el conjunto de nodos que ejecuta las consultas y almacena los datos.  

🟠 **Nodos**: Un cluster está compuesto por uno o varios nodos:  
   - **Nodo Líder**: Gestiona la distribución de consultas.  
   - **Nodos de Cómputo**: Procesan y almacenan los datos.  

🟠 **Slices**: Cada nodo de cómputo se divide en "slices", donde se almacenan fragmentos de los datos.  

🟠 **Redshift Spectrum**: Permite ejecutar consultas directamente sobre datos almacenados en **Amazon S3** sin necesidad de cargarlos en Redshift.  

### **🔹 Casos de Uso**  

📊 **Análisis de Datos Empresariales**  
   - Empresas que procesan grandes volúmenes de datos para generar reportes e insights.  

📈 **Big Data y Machine Learning**  
   - Se puede integrar con **AWS Glue, Amazon SageMaker y Data Lake** para análisis avanzados.  

🏢 **Empresas de Retail, Finanzas y Salud**  
   - Análisis de tendencias de clientes, detección de fraudes y reportes en tiempo real.

### **🔹 Diferencia entre Redshift y un Data Lake (Amazon S3)**  

| **Característica** | **Amazon Redshift (Data Warehouse)** | **Amazon S3 (Data Lake)** |
|--------------------|----------------------------------|--------------------------|
| **Estructura** | Datos organizados y optimizados para consultas SQL | Datos en crudo, estructurados y no estructurados |
| **Consultas** | Alta velocidad con SQL optimizado | Requiere herramientas como Athena o Glue |
| **Uso principal** | Análisis de datos estructurados | Almacenamiento masivo y preprocesamiento de datos |

### **🔹 Conexión a Redshift desde la AWS CLI**  

1️⃣ **Configurar AWS CLI**:  
```bash
aws configure
```
2️⃣ **Listar clusters disponibles**:  
```bash
aws redshift describe-clusters
```
3️⃣ **Conectarse desde SQL Client**:  
   - Usar **DBeaver, SQL Workbench o psql** con las credenciales del cluster.

### **🔹 Conclusión**  

Amazon Redshift es una de las soluciones más potentes para **almacenamiento y análisis de datos empresariales**. Su arquitectura optimizada permite **consultas rápidas, escalabilidad y fácil integración** con el ecosistema de AWS.  

Si buscas una solución de **Data Warehouse en la nube**, Redshift es una gran opción. 🚀

### Resumen

### ¿Qué es Amazon Redshift y cómo funciona?

Amazon Redshift es una potente herramienta que permite la gestión de grandes volúmenes de datos mediante la creación de clústeres escalables. A través de su interfaz gráfica, podremos visualizar información relevante, como la cantidad de datos compartidos y las copias realizadas en los servicios. También permite la integración de servicios como Cloudwatch para configurar alarmas y obtener una visión detallada de la capacidad del clúster en tiempo real.

### ¿Cómo se utiliza el Query Editor?

El Query Editor de Amazon Redshift es un componente fundamental para la ejecución de consultas y la creación de elementos de base de datos, como funciones, esquemas y tablas. Este editor es accesible desde un botón en la interfaz o seleccionando la opción "Query Data". A través de él:

- **Creación y gestión**: Puedes crear bases de datos, esquemas, tablas y funciones.
- **Carga de datos**: Permite importar datos desde archivos locales en formatos como CSV, JSON, Parquet, entre otros.
- **Configuración avanzada**: Se pueden establecer delimitadores y configuraciones detalladas al cargar datos.
- **Organización**: Ofrece la capacidad de organizar consultas en carpetas y almacenar o compartir consultas con equipos.

### ¿Cómo se integran otras herramientas con Redshift?

Amazon Redshift ofrece integraciones con herramientas complementarias, como Amazon S3 para el manejo de grandes volúmenes de datos y sistemas de alerta mediante inteligencia artificial. Además, habilita la importación y uso de notebooks compatibles, como aquellos que utilizan Spark o Python, aprovechando el poder de procesamiento del clúster de Redshift.

### ¿Qué características adicionales proporciona Redshift?

Una de las ventajas destacadas de Amazon Redshift es su interfaz rica en funcionalidades adicionales que facilitan el manejo y análisis de datos:

- **Visualización de datos**: Posibilidad de crear visualizaciones de datos para análisis más comprensibles.
- **Historial de consultas**: Un registro donde se pueden seguir las consultas ejecutadas, similar a un historial de transacciones en línea.
- **Configuración personalizada**: Ofrece opciones estéticas (modo oscuro o claro) y configuraciones avanzadas de conexiones y SQL.

### ¿Cómo puedo organizar y almacenar mis consultas?

El sistema de queries de Redshift permite mantener tus consultas bien organizadas:

- **Estructuración en carpetas**: Agrupa consultas para mejorar la organización y acceder a ellas de manera más eficiente.
- **Importación de consultas**: Desde otros sistemas o equipos, para continuar trabajando sin perder información.
- **Acceso compartido**: Puedes ver y trabajar con consultas propias, compartidas o de terceros, lo que facilita el trabajo colaborativo.

A medida que exploramos estas funcionalidades, nos empapamos más del amplio espectro de herramientas que proporciona Amazon Redshift para la administración y análisis de datos a gran escala. Este conocimiento es clave para potenciar nuestras habilidades en gestión de datos, y Redshift se posiciona como un aliado insustituible en este desafío. ¡Continúa explorando y mejorando en este emocionante camino de la tecnología de datos!

## Creando mi DataWarehouse en Redshift

A continuación, te guiaré paso a paso para configurar un **Data Warehouse** en **Amazon Redshift** desde cero. 🚀

### **1️⃣ Prerrequisitos**  

✅ Una cuenta en **AWS** (Si no la tienes, puedes crear una en [aws.amazon.com](https://aws.amazon.com))  
✅ Acceso a **Amazon Redshift**  
✅ AWS CLI configurado (Opcional, pero recomendado)  
✅ Un cliente SQL como **DBeaver, SQL Workbench, o pgAdmin**

### **2️⃣ Creando un Cluster en Amazon Redshift**  

### **🔹 Paso 1: Ir a la consola de AWS**  
1️⃣ Inicia sesión en **AWS Management Console**  
2️⃣ Busca **Amazon Redshift** en la barra de búsqueda  
3️⃣ Haz clic en **Clusters → Crear cluster**

### **🔹 Paso 2: Configurar el Cluster**  
🔹 **Elegir el tipo de implementación**  
   - **Producción**: Clúster con alto rendimiento  
   - **Prueba**: Cluster más pequeño y económico  

🔹 **Nombre del Cluster**: Escribe un nombre, por ejemplo:  
   ```redshift-dw-cluster```  

🔹 **Tipo de nodo**:  
   - Si es **prueba**, usa `dc2.large`  
   - Si es **producción**, usa `ra3.4xlarge` o superior  

🔹 **Cantidad de nodos**:  
   - Para pruebas: **1 nodo**  
   - Para producción: **2 o más nodos**  

🔹 **Credenciales de inicio de sesión**  
   - Usuario: `admin`  
   - Contraseña: `TuContraseñaSegura`

### **🔹 Paso 3: Configurar Acceso y Seguridad**  
🔹 **Habilitar acceso público** si te conectarás desde tu computadora  
🔹 Agregar una regla en el **Security Group** para permitir conexiones desde tu IP

### **🔹 Paso 4: Crear el Cluster**  
✅ Revisa la configuración y haz clic en **Crear Cluster**  
✅ Espera a que el estado cambie a **"Disponible"** (Toma unos minutos)

### **3️⃣ Conectarse a Amazon Redshift**  

### **🔹 Desde la AWS CLI**  
Para verificar el estado del clúster, usa:  
```bash
aws redshift describe-clusters --query "Clusters[*].ClusterStatus"
```
Para obtener el endpoint del clúster:  
```bash
aws redshift describe-clusters --query "Clusters[*].Endpoint.Address"
```

### **🔹 Desde un Cliente SQL (DBeaver, SQL Workbench, etc.)**  
🔹 Descarga e instala un cliente SQL si no lo tienes  
🔹 Conéctate con los siguientes datos:  
   - **Host**: El **endpoint** de Redshift  
   - **Puerto**: `5439`  
   - **Usuario**: `admin`  
   - **Base de datos**: `dev`  

### **4️⃣ Crear una Base de Datos y Tablas**  

Una vez conectado, puedes crear una **Base de Datos** y **Tablas**.  

### **🔹 Crear una Base de Datos**  
```sql
CREATE DATABASE mi_warehouse;
```
Para conectarte a ella:  
```sql
\c mi_warehouse;
```

### **🔹 Crear una Tabla en Redshift**  
```sql
CREATE TABLE ventas (
    id_venta INT PRIMARY KEY,
    fecha DATE,
    producto VARCHAR(100),
    cantidad INT,
    precio DECIMAL(10,2)
);
```

### **🔹 Insertar Datos**  
```sql
INSERT INTO ventas VALUES (1, '2025-03-10', 'Laptop', 2, 1500.00);
```

### **🔹 Consultar los Datos**  
```sql
SELECT * FROM ventas;
```

### **5️⃣ Optimización y Mejores Prácticas**  

✅ **Distribución de Datos**: Usa `DISTSTYLE` para mejorar la eficiencia  
✅ **Compresión de Datos**: Habilita `Columnar Encoding` para reducir tamaño  
✅ **Vacuum & Analyze**: Usa `VACUUM` y `ANALYZE` para mantener el rendimiento

### **✅ Conclusión**  

¡Listo! 🎉 Ahora tienes un **Data Warehouse en Amazon Redshift** configurado. Puedes empezar a **cargar, procesar y analizar datos** de manera eficiente. 🚀  

🔹 **¿Qué sigue?**  
- **Integrar con herramientas de BI** (Power BI, Tableau, AWS QuickSight)  
- **Automatizar la carga de datos con ETL/ELT** (AWS Glue, Lambda)  
- **Mejorar consultas con particiones y distribución de datos**  

### Resumen

### ¿Cómo conectarse a un cluster Redshift?

Conectar un warehouse como Amazon Redshift a tu infraestructura no es tan complicado como podría parecer. El primer paso es establecer una conexión adecuada a tu cluster. Para hacerlo, dentro de la sección del editor, busca la opción "serverless" en tu cluster y haz doble clic. Esto te llevará a la pantalla de conexión donde podrás ingresar con un usuario y contraseña. Estos datos se configuran inicialmente al crear el cluster, por lo tanto, asegúrate de tener el nombre de usuario y la contraseña a mano para establecer la conexión.

### ¿Cómo crear una base de datos en Redshift?

Crear una base de datos es fundamental para gestionar datos de manera eficiente. Una vez que te hayas conectado a tu cluster, irás a ver una pantalla que muestra bases de datos nativas y algunos servicios externos. Para crear tu base de datos, sigue estos pasos:

1. Presiona el botón "Create" y selecciona "Database".
2. Asegúrate de que estás correctamente conectado a tu cluster.
3. Especifica el nombre de la base de datos, por ejemplo, plugtyDB.
4. La creación de la base de datos requiere solo el nombre, pero puedes incluir configuraciones opcionales como usuarios, roles o integración con Amazon Glue si lo deseas.
5. Finaliza presionando "Create Database". Tras un breve momento, deberías recibir un mensaje indicando que se ha creado exitosamente.

### ¿Cómo crear una tabla y definir su esquema?

Una vez que tu base de datos esté lista, el siguiente paso es definir las tablas y su estructura. Siguiendo el ejemplo, crearás una tabla llamada "alumnos". Para hacer esto, asegúrate de estar dentro de la base de datos correcta (platziDB) y el esquema "public", y sigue estos pasos:

1. Presiona el botón "Create" y selecciona "Table".

2. Define el nombre de la tabla, por ejemplo, `alumnos`.

3. Añade columnas de manera manual, lo cual es recomendable para controlar mejor las características de cada campo.

4. Configura las columnas de la siguiente manera:

```sql
CREATE TABLE alumnos ( id_alumno INT PRIMARY KEY NOT NULL, nombre VARCHAR(50) NOT NULL, apellido VARCHAR(50) NOT NULL, pais VARCHAR(50) NOT NULL );
```

5. Presiona "Create Table" para completar el proceso. Puedes optar por usar "Open Query" para generar automáticamente el código SQL correspondiente a través de la interfaz gráfica.

### ¿Cuál es el siguiente paso después de crear la tabla?

Con la tabla creada y su esquema definido, el siguiente paso es realizar las inserciones necesarias para poblarla con datos. Este es un momento crucial, ya que permite verificar que la estructura de la base de datos funciona como esperas y que los datos se manejan correctamente.

Recuerda practicar creando tablas y bases de datos para reforzar el aprendizaje de esta unidad. Trabajar directamente con la interfaz y explorar las opciones disponibles te dará una comprensión más profunda y efectiva de cómo opera Amazon Redshift. Si encuentras dificultades durante el proceso, busca colaboración en foros, deja tus dudas en comentarios o revisa la documentación oficial. ¡Continúa aprendiendo y perfeccionando tus habilidades!

## Creando mi DataWarehouse en Redshift – Parte 2

En esta segunda parte, profundizaremos en **carga de datos, consultas optimizadas y seguridad** en **Amazon Redshift**. 🚀

### **1️⃣ Cargando Datos en Amazon Redshift**  

Para cargar datos en **Redshift**, las opciones más comunes son:  

✅ **COPY desde Amazon S3** (Recomendado 🚀)  
✅ **Insertar manualmente con SQL**  
✅ **Integrar con AWS Glue o DMS**  

### **🔹 Método 1: Cargar datos desde Amazon S3 (Recomendado)**  
**Pasos Previos:**  
🔹 **Subir un archivo CSV a un bucket de S3**  
🔹 Crear un **IAM Role** con permisos de `AmazonS3ReadOnlyAccess`  

```sql
COPY ventas
FROM 's3://mi-bucket/datos/ventas.csv'
IAM_ROLE 'arn:aws:iam::123456789012:role/RedshiftRole'
CSV
IGNOREHEADER 1;
```
🔹 **Verifica los datos** con:  
```sql
SELECT * FROM ventas LIMIT 10;
```

### **🔹 Método 2: Insertar datos manualmente (Para pruebas)**  
```sql
INSERT INTO ventas VALUES (2, '2025-03-11', 'Smartphone', 5, 900.00);
```

### **2️⃣ Optimizando Consultas en Redshift**  

### **🔹 Distribución de Datos (DISTSTYLE)**  
Mejora el rendimiento de consultas:  
```sql
CREATE TABLE ventas (
    id_venta INT PRIMARY KEY,
    fecha DATE,
    producto VARCHAR(100),
    cantidad INT,
    precio DECIMAL(10,2)
)
DISTSTYLE KEY
DISTKEY(id_venta);
```

### **🔹 Compresión de Columnas**  
Redshift usa compresión automática, pero puedes aplicar manualmente:  
```sql
ALTER TABLE ventas
ADD ENCODE ZSTD;
```

### **🔹 Ordenamiento de Datos (SORTKEY)**  
```sql
CREATE TABLE ventas (
    id_venta INT PRIMARY KEY,
    fecha DATE,
    producto VARCHAR(100),
    cantidad INT,
    precio DECIMAL(10,2)
)
SORTKEY (fecha);
```

## **3️⃣ Seguridad en Amazon Redshift**  

### **🔹 Acceso mediante IAM Role**  
Configura permisos en **AWS IAM** para acceso a S3 y otros servicios.

### **🔹 Control de Acceso con Usuarios y Grupos**  
🔹 **Crear un usuario en Redshift**  
```sql
CREATE USER analista PASSWORD 'PasswordSeguro123';
```
🔹 **Asignar permisos**  
```sql
GRANT SELECT ON ventas TO analista;
```

### **✅ Conclusión**  

🎯 Ahora tienes un **Data Warehouse optimizado y seguro en Redshift**.  

🔹 **¿Qué sigue?**  
- Integrar con herramientas de BI como Power BI o AWS QuickSight  
- Automatizar ETL con AWS Glue  
- Configurar backups y snapshots  

### Resumen

### ¿Cómo preparar tu entorno para trabajar con bases de datos?

Antes de sumergirte en la manipulación de datos dentro de tu warehouse, es fundamental asegurarte de que tu entorno esté bien configurado. Si al abrir la pantalla de consulta no encuentras tu base de datos, intenta cerrar y volver a abrir el editor de consultas. Esto debería permitirte visualizar tu base de datos y comenzar a ejecutar las consultas necesarias.

### ¿Cómo hacer una consulta SELECT en tu base de datos?

Para realizar una simple consulta de datos en tu tabla de alumnos, sigue estos pasos concretos:

1. **Ubica tu tabla de alumnos** en el editor de consultas.
2. **Realiza clic derecho** sobre la tabla y selecciona la opción "select" para obtener una representación básica de la tabla.
3. Podrás observar el esquema completo de la tabla y opciones para programar la ejecución de queries, guardarlas o visualizar el historial de consultas.

Al ejecutar la consulta SELECT, si la tabla está vacía, lo notarás ya que no se mostrarán registros. Además, puedes exportar los resultados obtenidos en diferentes formatos como JSON o CSV y generar gráficos para visualizar los datos de manera efectiva.

```sql
SELECT * FROM Alumnos;
```

### ¿Cómo insertar un nuevo registro en la tabla?

Para insertar datos en la tabla, utilizamos las sentencias SQL `INSERT INTO`. Aquí tienes cómo proceder para añadir tu primer registro:

1. D**efine los valores a insertar**. Decidimos inicialmente un ID y luego nombres, apellidos y país.

```sql
INSERT INTO BaseDatos.Eschema.Alumnos (id, nombre, apellido, país) VALUES (1, 'José', 'García', 'Argentina');
```

2. Selecciona y ejecuta el código para ver la confirmación de la inserción exitosa. Podrás observar la cantidad de registros afectados, un ID de consulta, tiempos, y el código utilizado.

3. Comprueba los datos insertados ejecutando un SELECT para verificar que los registros sean correctos.

### ¿Cómo actualizar un registro existente?

Actualizar un registro es sencillo con la cláusula SQL `UPDATE`. Supongamos que necesitas cambiar el país de un estudiante:

1. **Decide el nuevo valor** que deseas establecer en el campo relevante.

2. Especifica el criterio de selección con: `WHERE`.

```sql
UPDATE BaseDatos.Eschema.Alumnos SET país = 'Uruguay' WHERE id = 3;
```

Al ejecutar este código, valida nuevamente con un `SELECT` para confirmar que la actualización haya sido correcta.

### ¿Cómo eliminar un registro de la tabla?

Eliminar un registro se realiza con una sentencia `DELETE FROM`. Este procedimiento es útil cuando necesitas limpiar o modificar tu base de datos:

1. **Identifica el registro** a eliminar mediante el ID o cualquier otra columna.

```sql
DELETE FROM BaseDatos.Eschema.Alumnos WHERE id = 3;
```

2. Ejecuta el código y verifica los cambios con un `SELECT`.

La gestión adecuada de tus consultas también implica limpiar y cerrar queries abiertas que no necesitas, así evitarás errores debido a consultas simultáneas.

¡Te animamos a que experimentes más con este proceso! Un desafío podría ser crear nuevas tablas, por ejemplo, una de profesores o asignaturas, e integrarlas en tu modelo de datos. Compartir tus experiencias y avances en los comentarios siempre es beneficioso.

## Creando mi DataWarehouse en Redshift – Parte 3

En esta tercera parte, nos enfocaremos en la **optimización avanzada**, la **integración con herramientas de BI** y la **monitorización del rendimiento** en **Amazon Redshift**. 🚀

### **1️⃣ Optimización Avanzada en Amazon Redshift**  

### **🔹 Uso de Materialized Views (Vistas Materializadas)**  
Las **vistas materializadas** mejoran la velocidad de consulta al almacenar resultados precomputados.  

🔹 **Crear una vista materializada**  
```sql
CREATE MATERIALIZED VIEW ventas_resumen AS
SELECT producto, SUM(cantidad) AS total_vendido, SUM(precio) AS ingreso_total
FROM ventas
GROUP BY producto;
```
🔹 **Refrescar la vista** (para actualizar los datos)  
```sql
REFRESH MATERIALIZED VIEW ventas_resumen;
```

### **🔹 Redshift Spectrum: Consultar Datos Externos**  
Puedes consultar **datos en S3 sin cargarlos en Redshift** mediante **Spectrum**.  

🔹 **Crear un esquema externo para S3**  
```sql
CREATE EXTERNAL SCHEMA ventas_s3
FROM DATA CATALOG DATABASE 'mi_catalogo'
IAM_ROLE 'arn:aws:iam::123456789012:role/RedshiftRole';
```
🔹 **Consultar datos directamente desde S3**  
```sql
SELECT * FROM ventas_s3.ventas WHERE fecha >= '2025-01-01';
```

### **2️⃣ Integración con Herramientas de BI**  

### **🔹 Conexión con Amazon QuickSight**  
QuickSight permite visualizaciones interactivas en tiempo real.  
1️⃣ Configurar **Amazon QuickSight** con Redshift como fuente de datos.  
2️⃣ Usar **queries optimizadas** y **vistas materializadas** para mejorar rendimiento.  

🔹 **Ejemplo de consulta optimizada para visualización**  
```sql
SELECT fecha, producto, SUM(cantidad) AS total_vendido
FROM ventas
WHERE fecha BETWEEN '2025-01-01' AND '2025-12-31'
GROUP BY fecha, producto;
```

### **🔹 Conexión con Power BI o Tableau**  
Redshift es compatible con herramientas externas como **Power BI y Tableau**.  
🔹 **Requisitos:**  
✅ Instalar el conector ODBC o JDBC para Redshift  
✅ Configurar la conexión con las credenciales de Redshift

### **3️⃣ Monitoreo del Rendimiento en Redshift**  

### **🔹 Uso de Amazon CloudWatch para métricas**  
**CloudWatch** permite monitorear uso de CPU, memoria y consultas lentas.  

🔹 **Métricas clave en CloudWatch:**  
✅ `CPUUtilization` → Uso de CPU  
✅ `DatabaseConnections` → Conexiones activas  
✅ `QueryDuration` → Duración de consultas 

### **🔹 Análisis de consultas con EXPLAIN y SVL_QUERY_REPORT**  
Para detectar **cuellos de botella**, usa `EXPLAIN` y `SVL_QUERY_REPORT`.  

🔹 **Ejemplo con EXPLAIN**  
```sql
EXPLAIN SELECT * FROM ventas WHERE producto = 'Laptop';
```
🔹 **Ver análisis de consultas lentas**  
```sql
SELECT query, total_exec_time/1000000 AS tiempo_segundos
FROM SVL_QUERY_REPORT
ORDER BY total_exec_time DESC
LIMIT 5;
```

### **✅ Conclusión**  

🎯 Ahora has llevado tu **Data Warehouse en Redshift** al siguiente nivel con:  
✅ **Optimización con vistas materializadas y Spectrum**  
✅ **Integración con herramientas de BI**  
✅ **Monitoreo avanzado para mejorar el rendimiento**  

🔹 **¿Qué sigue?**  
- Implementar **Redshift Workload Management (WLM)** para mejorar concurrencia  
- Explorar **Machine Learning en Redshift ML**  
- Automatizar la optimización de consultas con **Advisor de Redshift**  

### Resumen

### ¿Cuáles son las herramientas clave que se integran con Amazon RedShift?

A lo largo del uso de Amazon RedShift, aparecen diversas herramientas que potencian su funcionalidad y se integran de manera efectiva para diversas soluciones dentro del entorno de Amazon Web Services (AWS). A continuación, exploraremos algunas de estas herramientas esenciales y sus características clave.

### ¿Qué es Amazon CloudWatch?

Amazon CloudWatch es una herramienta esencial en el arsenal de AWS, diseñada para monitorear recursos y servicios. Su funcionalidad no se limita únicamente a RedShift; de hecho, abarca todos los servicios dentro de AWS, ofreciendo las siguientes capacidades:

- Creación de alarmas: Permite configurar alertas basadas en métricas específicas para predecir problemas antes de que ocurran.
- Monitoreo de métricas y eventos: Seguimiento en tiempo real de los recursos y servicios para obtener información precisa y detallada.
- Paneles operacionales: Ofrece un dashboard personalizable donde se visualizan todos los aspectos operativos de los servicios monitoreados.

Esta herramienta es indispensable para mantener la eficiencia y el rendimiento de las operaciones en cualquier servicio vinculado a AWS.

### ¿Cómo funciona Amazon S3 con RedShift?

Amazon S3 (Simple Storage Service) es otro de los servicios altamente integrados con RedShift. Su versatilidad permite almacenar litros de datos que se pueden vincular y manipular para diferentes propósitos:

- **Estructuración de datos**: Ideal para organizar y gestionar grandes volúmenes de datos.
- **Ingesiones y conexiones de datos**: Los usuarios pueden integrar datos directamente desde S3 a RedShift, facilitando la carga y descarga de datos para análisis masivo.

El conocimiento detallado de S3 ampliará tu capacidad para manejar datos de manera eficiente.

### ¿Qué nos ofrece Amazon QuickSight?

Si buscas herramientas analíticas dentro de AWS, Amazon QuickSight se destaca como una opción poderosa similar a Power BI, diseñada para:

- **Visualización de datos**: Crea informes, gráficos y varías representaciones visuales a partir de los datos.
- **Conexión directa con RedShift**: Los usuarios pueden conectar sus datos almacenados en RedShift y transformarlos en información visualizable.
- **Suscripción paga**: Es importante notar que, a diferencia de otros servicios, QuickSight requiere de una suscripción para su uso completo.

La capacidad de QuickSight para transformar datos en representaciones visuales ofrece un gran apoyo en la toma de decisiones.

### ¿Cuál es el rol de DMS en AWS?

Amazon Data Migration Service (DMS) es crucial para todos los procesos de migración de datos dentro de AWS:

- **Proceso de migración**: Facilita la transferencia de esquemas de bases de datos, muy útil para RedShift y otras bases de datos relacionales.
- **Compatibilidad con múltiples bases de datos**: Soporta una variedad de bases de datos, asegurando una transición fluida y sin complicaciones.

Este servicio es ideal para quienes necesitan migrar grandes cantidades de datos sin afectar la continuidad de los servicios.

### Guía práctica: ¿Cómo eliminar recursos de AWS de manera eficiente?

Cuando trabajas con AWS, es esencial saber cómo gestionar y eliminar recursos de manera efectiva para evitar costos innecesarios. Aquí te mostramos un procedimiento claro para eliminar recursos en Amazon RedShift:

1. **Desconectarte del clúster**: Asegúrate de cerrar la sesión en cualquier pantalla vinculada al clúster de RedShift.
2. **Eliminar el workgroup y namespace**:

- Dirígete a la sección `work group` en la configuración.
- Selecciona el clúster deseado.
- Haz clic en `actions` y luego en `delete`.

3. **Confirmación de la eliminación**:

- Escribe la palabra "delete" para habilitar el botón de eliminación.
- Desactiva la opción para crear una copia final si no es necesaria.

4. **Validar la eliminación**: Verifica que todos los apartados hayan quedado vacíos.

Dominar este tipo de procedimientos es crucial para gestionar eficazmente tus recursos en la nube sin incurrir en gastos adicionales.

**Lecturas recomendadas**

[Supervisión y observabilidad de Amazon: Amazon CloudWatch - AWS](https://aws.amazon.com/es/cloudwatch/)

[Amazon QuickSight](https://aws.amazon.com/es/pm/quicksight/?gclid=Cj0KCQiA-aK8BhCDARIsAL_-H9loNCu_hA84o-gSNqYP_-oeJvb6p-g_ZTl7lOEwuy8uc5z3m4SUElgaAjoFEALw_wcB&trk=4b63ce16-547d-47f3-8767-99c56998f891&sc_channel=ps&ef_id=Cj0KCQiA-aK8BhCDARIsAL_-H9loNCu_hA84o-gSNqYP_-oeJvb6p-g_ZTl7lOEwuy8uc5z3m4SUElgaAjoFEALw_wcB:G:s&s_kwcid=AL!4422!3!651510248532!e!!g!!amazon%20quicksight!19836376513!152718795728)

[Migración de bases de datos a la nube, AWS Database Migration Service (AWS DMS), AWS](https://aws.amazon.com/es/dms/)

## Arquitectura Medallón

La **Arquitectura Medallón** es un enfoque utilizado en **Data Lakes** para estructurar y organizar los datos en diferentes niveles de calidad. Su objetivo es mejorar la gestión, el procesamiento y la gobernanza de los datos dentro de plataformas como **Databricks, Apache Spark y Azure Synapse**.

### **🔹 Concepto Clave**  
Los datos se dividen en **tres niveles jerárquicos** representados como **medallas** (bronce, plata y oro), cada uno con mayor calidad y estructuración.

### **🏅 Niveles de la Arquitectura Medallón**  

1️⃣ **Capa Bronce (Raw Data - Datos en crudo) 🟤**  
   - Datos en su estado **original**, sin transformaciones ni limpieza.  
   - Pueden estar en diferentes formatos: JSON, CSV, Parquet, logs, etc.  
   - Origen: bases de datos, IoT, APIs, redes sociales, etc.  
   - Usos: almacenamiento a largo plazo, auditoría y reproducción de datos.  

2️⃣ **Capa Plata (Cleansed Data - Datos limpiados) ⚪**  
   - Se aplica **limpieza, normalización y validación** de datos.  
   - Se eliminan duplicados, se corrigen errores y se homogenizan formatos.  
   - Se optimiza el almacenamiento para consultas más eficientes.  
   - Usos: análisis exploratorio, BI, Machine Learning.  

3️⃣ **Capa Oro (Curated Data - Datos listos para negocio) 🟡**  
   - Datos altamente refinados, listos para consumo.  
   - Estructurados en modelos relacionales o no relacionales según la necesidad.  
   - Utilizados para reportes, dashboards y decisiones de negocio.  
   - Usos: inteligencia de negocios, analítica avanzada y reporting estratégico.

### **🔍 Beneficios de la Arquitectura Medallón**  
✅ **Estructura escalable** para gestionar grandes volúmenes de datos.  
✅ **Mejora la calidad y gobernanza de datos** en cada nivel.  
✅ **Optimiza costos y rendimiento** en Data Lakes.  
✅ **Facilita la integración con procesos de ETL/ELT y Data Warehouses.**

### **📌 Aplicación en la Nube**  
🔹 **Databricks Lakehouse**: Arquitectura basada en medallón con Apache Spark.  
🔹 **AWS (S3 + Athena + Redshift)**: Almacena datos en S3 con niveles de procesamiento.  
🔹 **Azure Synapse Analytics**: Implementa medallón con Data Lake Storage Gen2.  

🚀 **En resumen:** La **Arquitectura Medallón** permite estructurar datos de manera eficiente dentro de un **Data Lake**, asegurando su calidad y disponibilidad para analítica y toma de decisiones.

## Creando mi Datalake en S3 - Parte 1

Para crear un **Data Lake en AWS S3**, se deben seguir varios pasos clave que incluyen la organización del almacenamiento, la ingesta de datos, la gobernanza y el acceso. A continuación, te explico cómo hacerlo.

### **🚀 Pasos para Crear un Data Lake en S3**  

### **1️⃣ Configurar un Bucket en S3**  
✅ Ve a la consola de **AWS S3** y crea un **nuevo bucket**.  
✅ Define un nombre único y selecciona una **región** cercana a los consumidores de datos.  
✅ Habilita **Versioning** para rastrear cambios en los datos.  
✅ Configura **encriptación** para proteger los datos sensibles. 

### **2️⃣ Diseñar la Arquitectura del Data Lake**  
Para seguir un enfoque **estructurado y escalable**, usa la **Arquitectura Medallón** con **tres carpetas principales** dentro del bucket:  

📂 `/raw` → **Capa Bronce:** Datos en crudo, sin procesar.  
📂 `/clean` → **Capa Plata:** Datos validados y transformados.  
📂 `/curated` → **Capa Oro:** Datos listos para análisis y reportes.  

Ejemplo de organización en S3:  

```
s3://mi-datalake/raw/iot/  
s3://mi-datalake/raw/socialmedia/  
s3://mi-datalake/clean/transactions/  
s3://mi-datalake/curated/sales/  
```

### **3️⃣ Ingesta de Datos en S3**  
🔹 **Carga manual:** Desde la consola de AWS o AWS CLI.  
🔹 **Automática:** Con AWS Glue, AWS Lambda o AWS DataSync.  
🔹 **Streaming:** Con Kinesis Firehose para datos en tiempo real.

### **4️⃣ Procesamiento y Transformación**  
💡 Usa servicios como:  
✔ **AWS Glue** → Para ejecutar ETL (Extract, Transform, Load).  
✔ **Amazon Athena** → Para consultas SQL sin necesidad de servidores.  
✔ **AWS Lambda** → Para procesamiento en tiempo real.  
✔ **Amazon EMR (Hadoop/Spark)** → Para grandes volúmenes de datos.

### **5️⃣ Seguridad y Gobernanza**  
🔐 **IAM Policies** → Control de acceso por usuario/servicio.  
🔐 **AWS Lake Formation** → Gestión centralizada del Data Lake.  
🔐 **Logging con AWS CloudTrail** → Seguimiento de accesos y cambios.

### **6️⃣ Consultar los Datos del Data Lake**  
💾 Usa **Amazon Athena** para consultar datos con SQL sin necesidad de infraestructura:  

```sql
SELECT * FROM "mi_datalake_db"."ventas"
WHERE fecha > '2024-01-01';
```

📊 También puedes integrar con **AWS QuickSight** para visualización de datos.

### **✅ Beneficios de un Data Lake en AWS S3**  
✔ **Escalabilidad** → Maneja petabytes de datos sin problemas.  
✔ **Costo-Eficiente** → S3 tiene almacenamiento por niveles para optimizar costos.  
✔ **Flexibilidad** → Soporta datos estructurados, semiestructurados y no estructurados.  
✔ **Seguridad** → Encriptación, control de acceso y auditoría.

### **📌 Conclusión**  
Crear un **Data Lake en S3** te permite almacenar, procesar y analizar grandes volúmenes de datos de manera **eficiente y segura**. Integrando servicios como **AWS Glue, Athena y Lake Formation**, puedes convertirlo en una solución robusta para **Big Data y Analytics**. 🚀

### Resumen

### ¿Qué es Amazon S3 y cuál es su importancia?

Amazon S3, parte de Amazon Web Services (AWS), es un servicio clave para el almacenamiento de datos. Funciona como un "Data Lake" que permite trabajar con datos estructurados, semi-estructurados y no estructurados, siendo vital para las aplicaciones empresariales que requieren integrarse con diversas fuentes de datos. Facilita almacenar, gestionar y manipular grandes cantidades de datos de manera eficiente y segura.

### ¿Cómo se crea un bucket en Amazon S3?

Un bucket en S3 se puede describir como un directorio o carpeta, permitiendo definir y almacenar diversas estructuras de datos:

1. **Acceso al servicio S3**: Una vez en AWS, localiza y accede a S3 desde el panel principal.
2. **Creación del bucket**:
- Selecciona "Crear bucket" y completa el formulario.
- Define el propósito del bucket: general o específico (baja latencia, etc.).
- Elige un nombre único para el bucket en AWS.
- Puedes optar por utilizar un bucket existente como base, pero en este caso, lo crearemos desde cero.

3. **Configuraciones de acceso**:

- Determina si el bucket será público o privado.
- Gestiona quién puede acceder al bucket y a sus objetos.

4. Versionado y configuraciones adicionales:

- Puedes habilitar el versionado de archivos para mantener un historial de versiones.
- Configura etiquetas, encriptación de datos y otras opciones avanzadas.

### Finalización del proceso:

- Al completar el formulario, presiona "Create bucket". Ajusta cualquier inconveniente, como caracteres inválidos, antes de proceder.

### ¿Cómo gestionar y asegurar los datos en un bucket de S3?

La gestión y seguridad de los datos en un bucket de S3 son aspectos críticos:

- **Subida y gestión de archivos**: Sube archivos, crea carpetas y gestiona directorios desde el panel principal del bucket.

- **Meta Datos y propiedades**: Accede a la descripción de los metadatos y propiedades de los archivos, que incluyen opciones de encriptación y versiones.

- **Permisos y configuraciones de acceso**:

- Modifica configuraciones de acceso para establecer quién puede modificar o acceder al bucket.

- Usa políticas de privacidad en formato JSON para definir acceso detallado a los objetos del bucket.

### ¿Qué herramientas adicionales ofrece S3 para la gestión de datos?

Amazon S3 ofrece herramientas avanzadas para la gestión eficiente de datos:

- **Métricas del bucket**: Proporciona acceso a métricas relacionadas con accesos, consumos y almacenamiento. Aunque estas estarán vacías si no hay datos, ofrecen una manera de monitorear el uso del bucket.

- **Reglas de ciclo de vida**: Crea "Lifecycle Rules" para automatizar tareas como la eliminación de datos antiguos, actualizaciones automáticas, y modificaciones basadas en periodos de tiempo.

- **Puntos de acceso**: Establece puntos de acceso para gestionar redes y networking asociados al bucket.

A través de estas funcionalidades, S3 ofrece un poderoso entorno para el almacenamiento y gestión de datos en la nube, respaldado por herramientas que refuerzan la eficiencia y seguridad del manejo de información.

**Lecturas recomendadas**

[AWS | Almacenamiento de datos seguro en la nube (S3)](https://aws.amazon.com/es/s3/?nc=sn&loc=0)

## Creando mi Datalake en S3 – Parte 2

En esta segunda parte, configuraremos permisos, automatizaremos la ingesta de datos y realizaremos consultas con **Athena**.

### **1️⃣ Configurar Permisos y Seguridad en S3**  
Antes de que los servicios de AWS puedan acceder a los datos, debes configurar los permisos adecuados en **IAM** y **S3**.

### **✅ Configurar IAM Policies para Acceso a S3**  
Crea una política en IAM para otorgar permisos a un usuario o servicio específico:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::mi-datalake",
        "arn:aws:s3:::mi-datalake/*"
      ]
    }
  ]
}
```
📌 **Tip**: Usa **IAM Roles** si otros servicios como AWS Glue o Athena necesitan acceder a S3.

### **2️⃣ Automatizar la Ingesta de Datos**
### **🔄 Opciones de Ingesta Automática**  
- **AWS Lambda + S3** → Procesa archivos en tiempo real.  
- **Kinesis Firehose** → Para transmisión de datos en tiempo real.  
- **AWS DataSync** → Para sincronización con servidores locales.  

Ejemplo: Crear una **función Lambda** que mueve archivos de la capa **raw** a la **clean**:

```python
import boto3

s3 = boto3.client('s3')

def lambda_handler(event, context):
    source_bucket = "mi-datalake"
    destination_bucket = "mi-datalake"
    
    for record in event['Records']:
        key = record['s3']['object']['key']
        copy_source = {'Bucket': source_bucket, 'Key': key}
        
        if key.startswith("raw/"):
            new_key = key.replace("raw/", "clean/")
            s3.copy_object(Bucket=destination_bucket, CopySource=copy_source, Key=new_key)
            s3.delete_object(Bucket=source_bucket, Key=key)
    
    return "Proceso de transformación completado"
```

✅ **Este script mueve archivos de la capa `raw/` a `clean/` cuando se suben a S3.**

### **3️⃣ Consultar Datos con Athena**  
Athena permite **consultar S3 con SQL**, sin necesidad de infraestructura.

### **🛠️ Crear una Base de Datos en Athena**
Ejecuta en **Athena Query Editor**:

```sql
CREATE DATABASE mi_datalake_db;
```

### **🛠️ Crear una Tabla para Datos en S3**
Ejemplo para datos de ventas en **Parquet**:

```sql
CREATE EXTERNAL TABLE mi_datalake_db.ventas (
    id STRING,
    fecha DATE,
    producto STRING,
    cantidad INT,
    precio DOUBLE
)
STORED AS PARQUET
LOCATION 's3://mi-datalake/curated/ventas/';
```

### **4️⃣ Visualizar Datos con AWS QuickSight**
Para crear dashboards, conecta **QuickSight** con **Athena** y consulta los datos en S3.

✅ **Pasos:**  
1. En **QuickSight**, agrega una nueva fuente de datos.  
2. Selecciona **Athena** y la base de datos `mi_datalake_db`.  
3. Crea visualizaciones basadas en los datos **curados** de S3.

### **📌 Conclusión**  
En esta segunda parte:  
✔ **Configuramos permisos de IAM para S3** 🔒  
✔ **Automatizamos la ingesta con Lambda** 🔄  
✔ **Consultamos datos con Athena** 🛠️  
✔ **Visualizamos información con QuickSight** 📊  

🔜 **Parte 3:** Optimización y monitoreo del Data Lake. 

### Resumen

### ¿Cómo implementar la arquitectura Medageon en AWS S3?

Comenzar con la implementación de una arquitectura de datos, como Medageon, en AWS S3 puede parecer una tarea desafiante, pero con los pasos adecuados, es posible crear una estructura eficiente y funcional. En este proceso, configuraremos capas de bronce, plata y oro, esenciales para la gestión de datos. También exploraremos la carga de datos manual y remota, utilizando diferentes técnicas y herramientas ofrecidas por AWS.

### ¿Cómo crear capas en S3?

Nuestra primera tarea es crear una estructura dentro de un bucket en S3 que representará nuestras distintas capas de almacenamiento: bronce, plata y oro. Vamos a guiarte en la configuración de estas capas dentro del bucket.

### Crear carpetas en el bucket:

- **Bronce**: El primer paso es seleccionar la opción para crear una carpeta y nombrarla "bronce". Aunque AWS permite configuraciones adicionales como la encriptación, para este ejercicio no es necesario.
- **Plata**: Repite el proceso anterior, creando una segunda carpeta llamada "silver" o "plata".
- **Oro**: Finalmente, crea una tercera carpeta llamada "gold".

Con estos pasos sencillos, ya cuentas con tu arquitectura estructurada en S3.

### ¿Cómo cargar datos manualmente en S3?

Ahora que hemos configurado las capas, el siguiente paso es aprender a cargar datos. Comenzaremos con una carga manual, incorporando archivos de diferentes formatos para demostrar la versatilidad de S3 con distintos tipos de datos.

1. Acceder a la capa de bronce:

- Elige la opción "upload" para cargar datos.
- Puedes arrastrar y soltar los archivos o seleccionarlos directamente desde tu computadora.

2. Ejemplos de carga de datos:

- **Archivos estructurados**: Carga un archivo CSV. Por ejemplo, un archivo llamado "Disney.csv" se carga arrastrándolo a la carpeta "bronce".
- **Archivos semiestructurados**: Similar al proceso anterior, carga un archivo JSON, como "Batman.json".
- **Archivos no estructurados**: Por último, carga una imagen, por ejemplo, "Argentina.jpg".

Para cada uno de estos archivos, asegúrate de validar el destino y las configuraciones de permisos antes de presionar "upload".

### ¿Cómo realizar una carga remota con Cloud9?

La carga manual es útil, pero para optimizar la gestión de datos, una carga remota es más eficiente. Utilizaremos AWS Cloud9, un servicio que proporciona un IDE completo en la nube, para facilitar esta tarea.

1. Configurar Cloud9:
- Busca el servicio Cloud9 en AWS y créate un entorno de trabajo virtual (EC2) llamado, por ejemplo, "ClownEye_ilatsi44". Asegúrate de seleccionar la versión T2 Micro para entrar dentro del nivel de uso gratuito.
- Dentro de las configuraciones, puedes dejar todo por defecto o especificar preferencias como sistema operativo, autoapagado y etiquetas.

2. Interacción con S3 desde Cloud9:

- Una vez que el entorno está listo, abre Cloud9 IDE y crea un directorio llamado "dataset".

- Dentro de este directorio, carga un archivo de ejemplo, como "Spotify_tracks.csv".

- Usa el terminal para copiarlo al bucket S3 con el comando:

`aws s3 cp Spotify_tracks.csv s3://nombre_del_bucket/bronce`

Este proceso confirma que el archivo se ha cargado correctamente, validándose al regresar al bucket en S3.

Estos pasos subrayan cómo AWS S3 y servicios adicionales como Cloud9 pueden integrarse para realizar operaciones de manejo de datos de varios tipos, demostrando la versatilidad de esta plataforma en la gestión avanzada de datos. ¡Continúa explorando y aprovechando estas herramientas en tu arquitectura de datos!

**Nota:**: para cargar el archivo toca utilizar powershell con: 
`aws s3 cp .\spotify_tracks.xlsx s3://aws-bucket-platzi-0202/bronze/`

**Archivos de la clase**

[argentina.jpg](https://static.platzi.com/media/public/uploads/argentina_e475bfab-6182-4abd-8902-541b6afa81ce.jpg)

[batman.json](https://static.platzi.com/media/public/uploads/batman_f5c53912-b86c-4091-99d4-49b9a197f25a.json)

[disney-movies.xlsx](https://static.platzi.com/media/public/uploads/disney_movies_003944a1-0a00-416a-8f88-248bfb7314ed.xlsx)

[spotify-tracks.xlsx](https://static.platzi.com/media/public/uploads/spotify_tracks_7deb1c86-72fe-4039-84be-82b889dd6425.xlsx)

**Lecturas recomendadas**

[AWS Cloud9: IDE en la nube para escribir, ejecutar y depurar código](https://aws.amazon.com/es/pm/cloud9/?gclid=Cj0KCQiA-aK8BhCDARIsAL_-H9mJuXfMc0LKpXzCE5kUdeTz6eT5nANutDmqkWSRfjTeXv5n5zgSzLEaAk2fEALw_wcB&trk=ced24737-44d0-45e6-9dbc-3f332552f769&sc_channel=ps&ef_id=Cj0KCQiA-aK8BhCDARIsAL_-H9mJuXfMc0LKpXzCE5kUdeTz6eT5nANutDmqkWSRfjTeXv5n5zgSzLEaAk2fEALw_wcB:G:s&s_kwcid=AL!4422!3!651510248559!e!!g!!aws%20cloud9!19836376525!147106151196)

## Creando mi Datalake en S3 – Parte 3

Crear un **Data Lake en Amazon S3** implica varios pasos clave para asegurarte de que los datos estén organizados, accesibles y seguros. Aquí tienes una guía paso a paso para construirlo correctamente:

### **1️⃣ Crear el Bucket en S3**
Amazon S3 será el almacenamiento principal del Data Lake.

1. **Ir a la consola de AWS** → S3.
2. **Crear un nuevo bucket**:
   - Asigna un nombre único (ejemplo: `mi-datalake-bucket`).
   - Selecciona una región.
   - Configura el acceso: Desactiva "Bloquear acceso público" solo si es necesario.
   - Habilita la opción de **versionado** para evitar pérdidas de datos.

### **2️⃣ Definir la Estructura del Data Lake**
Organiza los datos en **capas** para mantener un flujo limpio:

```
s3://mi-datalake-bucket/
    ├── raw/          # Datos sin procesar
    ├── processed/    # Datos limpios y transformados
    ├── curated/      # Datos listos para análisis
```

- **Raw:** Datos en bruto sin modificaciones.
- **Processed:** Datos transformados y limpios.
- **Curated:** Datos listos para análisis o machine learning.

### **3️⃣ Configurar Permisos y Seguridad**
Controlar quién y cómo se accede a los datos es fundamental.

🔹 **IAM Roles y Políticas:**  
Crea una política en IAM para que solo ciertos servicios y usuarios accedan a cada capa.

🔹 **Bucket Policies:**  
Ejemplo de política para permitir acceso solo desde AWS Glue y Athena:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "glue.amazonaws.com"
      },
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::mi-datalake-bucket/*"
    }
  ]
}
```

🔹 **Cifrado de Datos:**  
- Activa **AWS KMS** para cifrar los datos en reposo.
- Usa **SSL/TLS** para la transmisión segura.

### **4️⃣ Ingestar Datos en el Data Lake**
Hay varias formas de cargar datos en el Data Lake:

✅ **Subida manual**  
Desde la consola o con AWS CLI:
```sh
aws s3 cp mi_archivo.csv s3://mi-datalake-bucket/raw/
```

✅ **Automatización con AWS Glue o Lambda**  
Ejemplo: Un **trigger Lambda** que mueva los archivos de `raw/` a `processed/`.

✅ **AWS DataSync o Kinesis**  
Para ingestar datos en tiempo real desde IoT o bases de datos.

### **5️⃣ Catalogar Datos con AWS Glue**
Para poder consultar los datos fácilmente:

1. **Crear un Data Catalog** en AWS Glue.
2. **Definir un Crawler** que indexe las carpetas `processed/` y `curated/`.
3. **Ejecutar el Crawler** para registrar los esquemas en **Glue Catalog**.

### **6️⃣ Consultar el Data Lake con Amazon Athena**
Athena permite hacer consultas SQL sin necesidad de una base de datos:

1. Ir a **Amazon Athena**.
2. Conectar con el **Glue Data Catalog**.
3. Ejecutar consultas en los datos almacenados:
   ```sql
   SELECT * FROM mi_datalake.processed_datos WHERE fecha = '2024-03-11';
   ```

### **7️⃣ Optimización y Monitoreo**
💡 **Particionamiento y Formatos Eficientes**  
- Usar **Parquet** o **ORC** en lugar de CSV para mejorar rendimiento.
- Particionar datos por fecha (`year/month/day`) para consultas más rápidas.

📊 **Monitoreo con AWS CloudWatch y CloudTrail**  
- CloudWatch para métricas de almacenamiento y acceso.
- CloudTrail para auditar quién accede a los datos.

🚀 ¡Listo! Con estos pasos, tendrás un **Data Lake en Amazon S3** bien estructurado, seguro y listo para análisis.

### Resumen

### ¿Qué herramientas se integran con AWS S3?

AWS S3 es una de las soluciones de almacenamiento más robustas de Amazon Web Services. Además de ser versátil y segura, cuenta con un ecosistema amplio de herramientas que se pueden integrar para mejorar y facilitar el trabajo con este servicio. Estas herramientas van desde la gestión de datos hasta su visualización, lo que permite optimizar diversos procesos empresariales y personales.

### ¿Cómo se utiliza AWS CLI?

Una de las herramientas más útiles y relacionadas con el manejo de AWS S3 es la AWS Command Line Interface (CLI). Esta interfaz de línea de comandos permite una gestión eficaz de los servicios de AWS desde la línea de comandos de tu computador. Ofrece simplicidad y flexibilidad, permitiéndote administrar diferentes servicios de AWS con comandos sencillos. Entre sus ventajas destaca su capacidad para:

- **Configurar múltiples servicios**: Puedes gestionar no solo S3, sino también otros servicios de AWS.
- **Automatización**: Al automatizar tareas, puedes ahorrar tiempo y reducir errores.
- **Portabilidad**: Trabaja de manera local sin necesidad de interfaces gráficas extensas.

### ¿Qué es Amazon Athena?

Amazon Athena es otra herramienta poderosa para los usuarios de AWS S3 que buscan realizar análisis de datos de manera más efectiva. Esta plataforma te permite conectar directamente a un bucket de S3 y ejecutar consultas SQL, lo que facilita la generación de reportes y el manejo de datos almacenados en el bucket. Las características principales de Amazon Athena incluyen:

- **Consulta directa**: Ejecuta SQL sobre datos en S3 sin necesidad de moverlos.
- **Administración de bases de datos**: Crea bases de datos, tablas y catálogos directamente desde Athena.
- **Configuración flexible**: La opción de cambiar el bucket al que te conectas para una gestión fluida de los datos.

### ¿Qué rol juega AWS Glue?

Si estás trabajando con grandes volúmenes de datos en AWS S3 y necesitas realizar transformaciones complejas, AWS Glue es la herramienta a considerar. Este servicio de ETL (Extracción, Transformación y Carga) facilita la preparación de datos para análisis, ofreciendo:

- **Transformación de datos**: Simplifica el proceso de transformación de datos para su análisis.
- **Ingesta de datos**: Recoge datos de varias fuentes y los procesa directamente en S3.
- **Cargado eficiente**: Optimiza el proceso de carga para integrarlo con otras herramientas analíticas.

### ¿Cómo se integran herramientas de visualización de datos?

La integración de AWS S3 no se limita solo a herramientas de AWS. Existen tecnologías de terceros que también se integran eficientemente, particularmente para la visualización de datos. Entre las más populares se encuentran:

- **Power BI**: Utilizada para crear reportes y dashboards dinámicos, con integración directa a los datos almacenados en S3.
- **Tableau (versión paga)**: Otra herramienta poderosa para la visualización de datos, permitiendo integraciones efectivas con servicios de AWS.

### ¿Cómo se eliminan servicios de AWS para evitar costos?
Eliminar servicios no utilizados es crucial para evitar costos innecesarios en AWS. S3, aunque ofrece una capa gratuita con almacenamiento de hasta cinco gigas, sería sensato practicar la eliminación de recursos si no se necesitan más. Aquí tienes el proceso para eliminar un bucket en S3:

1. **Vaciar el bucket**: No puedes eliminar un bucket que tiene objetos dentro, así que primero debes vaciarlo.
2. **Confirmación de vaciado**: Escribe el texto de eliminación permanente para confirmar el vaciado del bucket.
3. **Eliminar el bucket vacío**: Una vez vacío, selecciona el bucket y confirma la eliminación escribiendo su nombre completo.

Este enfoque no solo libera espacio y evita costos adicionales, sino que también fomenta un entorno de trabajo más organizado.

A modo de cierre, AWS S3 ofrece una rica variedad de herramientas integradas que potencian su uso. Sin embargo, es esencial mantener una buena gestión de los recursos para optimizar costos y eficiencia. Continúa explorando y seleccionando las herramientas que mejor se ajusten a tus necesidades específicas para maximizar tus capacidades en la nube de AWS.

**Lecturas recomendadas**

[Integración de datos sin servidor: AWS Glue, Amazon Web Services](https://aws.amazon.com/es/glue/)

[Interfaz de la línea de comandos - AWS CLI - AWS](https://aws.amazon.com/es/cli/)

[Consultas de datos al instante | Análisis de datos SQL | Amazon Athena](https://aws.amazon.com/es/athena/)