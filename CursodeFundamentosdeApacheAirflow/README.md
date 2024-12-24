# Curso de Fundamentos de Apache Airflow

## Introducción al curso

Aquí tienes un ejemplo de introducción para un curso, estructurado y claro, que puedes adaptar según la temática específica:

### **Introducción al Curso**

¡Bienvenidos al curso de [Nombre del Curso]! Este programa ha sido diseñado especialmente para brindarte las herramientas, los conocimientos y las habilidades necesarias para destacar en el área de [tema del curso]. A lo largo de este curso, exploraremos conceptos clave, resolveremos problemas prácticos y aplicaremos estrategias que te permitirán dominar los objetivos del aprendizaje.

#### **¿Qué puedes esperar?**
- **Contenido actualizado:** Nuestro curso cubre las últimas tendencias y mejores prácticas en [tema del curso].
- **Aprendizaje práctico:** Incluye ejemplos, casos de estudio y ejercicios que te ayudarán a aplicar lo aprendido en situaciones reales.
- **Soporte continuo:** Durante todo el curso, contarás con el apoyo de [instructores, tutores, comunidad de aprendizaje] para resolver tus dudas y guiarte en el proceso.

#### **Objetivos del Curso**
Al finalizar este curso, serás capaz de:
1. [Objetivo 1: Describe una habilidad o conocimiento específico].
2. [Objetivo 2: Describe otro resultado esperable].
3. [Objetivo 3: Añade un objetivo relevante y medible].

#### **Metodología**
El curso combina:
- Sesiones teóricas para entender los fundamentos de [tema].
- Talleres prácticos para fortalecer tu aprendizaje a través de la experiencia.
- Evaluaciones y proyectos para medir tu progreso y consolidar tu conocimiento.

#### **¿A quién está dirigido este curso?**
Este curso está diseñado para [perfil del estudiante, e.g., principiantes, profesionales, estudiantes avanzados], que deseen [beneficio principal del curso].

#### **Requisitos previos**
Antes de comenzar, asegúrate de contar con:
- [Requisito 1].
- [Requisito 2].
- Ganas de aprender y explorar nuevos desafíos.

### **¡Comencemos!**

Estamos emocionados de acompañarte en este viaje de aprendizaje. Prepárate para alcanzar nuevas metas y descubrir todo lo que [tema del curso] tiene para ofrecer.

## ¿Para qué sirve airflow?

Apache Airflow es una plataforma de código abierto diseñada para orquestar y programar flujos de trabajo (workflows) en la gestión de datos y tareas repetitivas de manera eficiente y escalable. A continuación, te explico sus usos principales:

### **¿Para qué sirve Apache Airflow?**

#### 1. **Automatización de flujos de trabajo**
   Airflow permite definir, programar y monitorear flujos de trabajo que constan de tareas interdependientes. Por ejemplo:
   - Extraer datos de una fuente (ETL).
   - Procesar esos datos.
   - Almacenar los datos procesados en un Data Warehouse o Data Lake.
   - Generar reportes o visualizaciones.

#### 2. **Gestión de dependencias**
   Los flujos de trabajo en Airflow están representados como gráficos acíclicos dirigidos (DAGs), lo que permite definir relaciones y dependencias entre tareas. Esto asegura que:
   - Las tareas se ejecuten en el orden correcto.
   - Las fallas puedan ser gestionadas de manera efectiva sin afectar todo el flujo.

#### 3. **Programación de tareas**
   Airflow permite programar la ejecución de tareas de manera flexible, desde una vez al día hasta horarios más complejos, como "el último día laborable de cada mes".

#### 4. **Monitorización y re-ejecución**
   Airflow ofrece una interfaz gráfica (UI) para monitorear y depurar los flujos:
   - Ver el estado actual de las tareas (en progreso, exitoso, fallido).
   - Volver a ejecutar tareas que fallaron sin necesidad de repetir todo el flujo.

#### 5. **Integración con múltiples herramientas**
   Airflow puede interactuar con diversas tecnologías y plataformas gracias a sus operadores predefinidos y la capacidad de personalizarlos:
   - Bases de datos (PostgreSQL, MySQL, etc.).
   - Herramientas en la nube (AWS, Google Cloud, Azure).
   - APIs personalizadas.
   - Sistemas de procesamiento como Spark o Hadoop.

#### 6. **Escalabilidad**
   Al ejecutarse en arquitecturas distribuidas, Airflow puede manejar grandes volúmenes de tareas en entornos de producción con múltiples nodos.

### **Casos de uso típicos**
1. **ETL (Extracción, Transformación y Carga)**  
   Orquestar procesos que extraen datos, los transforman y los cargan en sistemas de almacenamiento centralizado.
   
2. **Pipeline de Machine Learning**  
   Automatizar pasos como preprocesamiento, entrenamiento, evaluación y despliegue de modelos.

3. **Gestión de datos en Data Warehousing**  
   Planificar cargas regulares o incrementales de datos al Data Warehouse.

4. **Procesos empresariales repetitivos**  
   Automatizar reportes financieros, reconciliaciones de cuentas, entre otros.

### **Ventajas de Apache Airflow**
- Código definido en Python, lo que lo hace flexible y personalizable.
- Soporte para tareas dinámicas y complejas.
- Comunidad activa y ecosistema creciente.
- UI intuitiva para monitorear flujos.

En resumen, **Apache Airflow sirve para orquestar y gestionar flujos de trabajo complejos de manera programada, monitoreable y escalable**, siendo una herramienta clave en proyectos de analítica, Big Data y desarrollo de pipelines.

## ¿Por qué usar airflow?

Usar **Apache Airflow** es una decisión estratégica en proyectos de análisis de datos, automatización y orquestación de flujos de trabajo por las siguientes razones:

### **1. Escalabilidad**
   - Airflow está diseñado para entornos de producción en los que las cargas de trabajo pueden crecer rápidamente.
   - Puede ejecutarse en una arquitectura distribuida, gestionando miles de tareas simultáneamente.

### **2. Flexibilidad**
   - Los flujos de trabajo (DAGs) se definen en Python, lo que permite gran personalización y la integración con bibliotecas existentes.
   - Ofrece soporte para tareas dinámicas y dependencias complejas.

### **3. Orquestación avanzada**
   - Permite gestionar dependencias entre tareas, asegurando que cada paso del flujo se ejecute en el orden adecuado.
   - Soporta reintentos automáticos, pausas, o reanudaciones en caso de fallos.

### **4. Programación de tareas**
   - Ofrece una programación avanzada, desde tareas diarias simples hasta ejecuciones complejas como "el tercer martes de cada mes" o "el último día hábil del trimestre".

### **5. Monitoreo y visualización**
   - Incluye una interfaz gráfica intuitiva para:
     - Monitorear el estado de las tareas.
     - Reejecutar tareas fallidas.
     - Analizar dependencias y duración de los flujos.

### **6. Integración con múltiples tecnologías**
   - Airflow tiene operadores predefinidos para integrarse con herramientas populares:
     - Bases de datos (MySQL, PostgreSQL, etc.).
     - Servicios en la nube (AWS, Google Cloud, Azure).
     - Procesos de Big Data (Hadoop, Spark).
     - APIs personalizadas y scripts locales.

### **7. Mantenimiento de historial**
   - Conserva un registro detallado de las ejecuciones pasadas, facilitando auditorías y análisis de desempeño.

### **8. Comunidad activa y soporte**
   - Es una herramienta de código abierto con una comunidad amplia y recursos de soporte.
   - Frecuentes actualizaciones que mejoran su funcionalidad y estabilidad.

### **9. Evita la "codificación manual" de flujos**
   - Sin Airflow, es común tener scripts individuales que se ejecutan manualmente o mediante crons, lo que puede volverse inmanejable.
   - Airflow centraliza la definición y gestión de todos los flujos, reduciendo la complejidad y el riesgo de errores.

### **10. Casos de uso típicos**
   - **ETL y ELT:** Automatizar extracción, transformación y carga de datos.
   - **Pipelines de Machine Learning:** Orquestar procesos como preprocesamiento, entrenamiento y evaluación de modelos.
   - **Procesos de negocio repetitivos:** Generación de reportes financieros, sincronización de datos.
   - **Gestión de Big Data:** Ejecutar y monitorear procesos en clústeres de datos.

**En resumen:** Usar Airflow es ideal cuando necesitas una herramienta confiable, escalable y flexible para orquestar tareas y flujos complejos, reduciendo la carga operativa y mejorando la eficiencia de tu proyecto.

## DAG

Un **DAG** (*Directed Acyclic Graph*) es un componente central en **Apache Airflow** que representa un flujo de trabajo. Este flujo está compuesto por tareas individuales y sus dependencias, organizadas de manera que sigan una estructura de grafo dirigido y acíclico.

### **Características principales de un DAG**

1. **Dirigido**:
   - Cada tarea en un DAG tiene una dirección específica, indicando el flujo lógico de las dependencias.
   - Por ejemplo, si `Task A → Task B`, significa que `Task B` se ejecutará después de que `Task A` haya finalizado correctamente.

2. **Acíclico**:
   - No puede haber bucles o ciclos en el flujo de trabajo.
   - Esto asegura que las tareas no entren en un estado de ejecución infinita.

3. **Configuración programática**:
   - Los DAGs se definen en código Python, lo que brinda flexibilidad para agregar lógica personalizada en la definición de tareas o dependencias.

### **Estructura de un DAG**

Un DAG en Airflow se configura definiendo:

- **Nombre del DAG**: Identificador único del flujo.
- **Programación (Schedule)**: Frecuencia con la que debe ejecutarse el flujo (diario, semanal, cada hora, etc.).
- **Conjunto de tareas**: Tareas individuales que conforman el flujo.
- **Dependencias entre tareas**: Relaciones que determinan el orden de ejecución.

```python
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from datetime import datetime

# Crear el DAG
dag = DAG(
    'mi_primer_dag',
    description='Un ejemplo básico de DAG',
    schedule_interval='@daily',  # Se ejecutará diariamente
    start_date=datetime(2023, 1, 1),  # Fecha de inicio del DAG
    catchup=False  # Evita ejecutar tareas atrasadas
)

# Definir las tareas
inicio = DummyOperator(task_id='inicio', dag=dag)
proceso = DummyOperator(task_id='proceso', dag=dag)
fin = DummyOperator(task_id='fin', dag=dag)

# Definir dependencias
inicio >> proceso >> fin
```

### **Elementos clave en un DAG**

1. **Tareas (Tasks)**:
   - Componentes individuales del flujo.
   - Pueden ser operadores predefinidos, como `BashOperator`, `PythonOperator`, o tareas personalizadas.

2. **Dependencias**:
   - Se especifican usando `>>` (dependencia directa) o `<<` (dependencia inversa).
   - Ejemplo:
     ```python
     tarea1 >> [tarea2, tarea3]
     ```

3. **Programación (Schedule)**:
   - Define cuándo se debe ejecutar el DAG.
   - Puede ser con expresiones cron (`0 12 * * *`) o intervalos predefinidos como `@daily`, `@hourly`.

4. **Fecha de inicio y fin**:
   - El **`start_date`** marca cuándo comienza el DAG.
   - Opcionalmente, un **`end_date`** puede limitar su ejecución.

5. **Propiedades adicionales**:
   - **`Retries`**: Número de intentos en caso de fallo.
   - **`Timeout`**: Límite de tiempo para ejecutar las tareas.
   - **`Catchup`**: Permite ejecutar tareas atrasadas si el DAG se activa después de la fecha de inicio.

### **Ventajas de los DAGs**

1. **Visualización clara**:
   - Airflow proporciona una interfaz gráfica para observar la estructura del DAG y el estado de las tareas.

2. **Escalabilidad**:
   - Los DAGs permiten manejar flujos complejos con dependencias múltiples.

3. **Reutilización**:
   - Los DAGs definidos en código son fáciles de modificar, mantener y reutilizar.

### **Usos comunes de los DAGs**

1. **Pipelines de ETL/ELT**:
   - Extracción, transformación y carga de datos de sistemas fuente a un Data Warehouse o Data Lake.

2. **Procesos de Machine Learning**:
   - Automatización de entrenamientos, evaluaciones y despliegues de modelos.

3. **Reportes automatizados**:
   - Generación y envío de reportes periódicos.

4. **Integraciones de sistemas**:
   - Orquestar sincronización de datos entre APIs o bases de datos.

En resumen, un **DAG** es el núcleo de cualquier flujo de trabajo en Apache Airflow, proporcionando una estructura programable, visualizable y altamente escalable para ejecutar tareas dependientes.

## Tasks y Operators

En **Apache Airflow**, **tasks** y **operators** son conceptos clave para construir y ejecutar flujos de trabajo.

## **Tasks (Tareas)**

Una **task** es una unidad individual de trabajo dentro de un DAG (*Directed Acyclic Graph*). Cada tarea representa una operación específica que se ejecuta como parte del flujo de trabajo. Las tareas son instancias de operadores, y juntas conforman las actividades que ocurren en un DAG.

### **Características de una Task**
1. **Independencia**: Cada tarea es independiente y realiza una acción específica.
2. **Dependencias**: Las tareas pueden depender unas de otras para garantizar que se ejecuten en el orden correcto.
3. **Configuración de reintentos**:
   - Puedes configurar cuántas veces intentará ejecutarse una tarea si falla.
   - Ejemplo: `retries=3`
4. **Estado de ejecución**:
   - Los estados posibles incluyen: `success`, `failed`, `running`, `skipped`.

## **Operators (Operadores)**

Un **operator** es una plantilla predefinida en Airflow que define lo que hace una tarea. Los operadores proporcionan la lógica para ejecutar una acción específica, como ejecutar un script de Python, interactuar con una API o copiar datos entre bases de datos.

### **Tipos de Operadores**

1. **Operadores de acción**:
   - Ejecutan una acción concreta, como un script o comando.
   - Ejemplos:
     - `BashOperator`: Ejecuta comandos Bash.
     - `PythonOperator`: Ejecuta funciones de Python.

2. **Operadores de transferencia**:
   - Manejan transferencias de datos entre sistemas.
   - Ejemplos:
     - `S3ToGCSOperator`: Copia datos de S3 a Google Cloud Storage.
     - `MySqlToPostgresOperator`: Transfiere datos entre bases de datos.

3. **Operadores de sensores**:
   - Esperan a que ocurra un evento antes de continuar.
   - Ejemplos:
     - `S3KeySensor`: Espera a que un archivo específico esté disponible en S3.
     - `HttpSensor`: Verifica que una URL esté activa.

4. **Operadores personalizados**:
   - Puedes crear operadores personalizados mediante herencia de clases base como `BaseOperator`.

### **Ejemplo de Operators en uso**

#### **BashOperator**
Ejecuta comandos en el sistema operativo:
```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

dag = DAG(
    'bash_example',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

bash_task = BashOperator(
    task_id='print_date',
    bash_command='date',
    dag=dag,
)
```

#### **PythonOperator**
Ejecuta funciones en Python:
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def my_python_function():
    print("Hello from Python!")

dag = DAG(
    'python_example',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

python_task = PythonOperator(
    task_id='run_python',
    python_callable=my_python_function,
    dag=dag,
)
```

#### **Sensor**
Espera a que un archivo exista en un sistema S3:
```python
from airflow import DAG
from airflow.providers.amazon.aws.sensors.s3_key import S3KeySensor
from datetime import datetime

dag = DAG(
    'sensor_example',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

s3_sensor = S3KeySensor(
    task_id='wait_for_file',
    bucket_name='my-bucket',
    bucket_key='path/to/file.csv',
    aws_conn_id='my_aws_connection',
    dag=dag,
)
```

## **Relación entre Tasks y Operators**

- Una **task** es una instancia de un **operator**.
- Un **operator** define la lógica de ejecución (qué hace).
- Una **task** utiliza esa lógica y la aplica en un DAG específico (cómo y cuándo lo hace).

Por ejemplo:
```python
bash_task = BashOperator(
    task_id='show_date',
    bash_command='date',
    dag=dag,
)
```
En este caso:
- **`BashOperator`** define cómo ejecutar un comando Bash.
- **`bash_task`** es la tarea específica que ejecuta el comando `date`.

## **Configuración Avanzada de Tasks**

1. **Dependencias**:
   - Se configuran para controlar el orden de ejecución.
   ```python
   task1 >> task2  # task2 se ejecuta después de task1
   task3 << task1  # task1 se ejecuta antes de task3
   ```

2. **Propiedades comunes**:
   - **`retry_delay`**: Tiempo entre reintentos.
   - **`timeout`**: Límite de tiempo para completar la tarea.
   - **`execution_timeout`**: Tiempo máximo permitido para que la tarea se ejecute.

3. **Ejemplo con varias dependencias**:
   ```python
   start = DummyOperator(task_id='start', dag=dag)
   process = PythonOperator(task_id='process', python_callable=my_function, dag=dag)
   end = DummyOperator(task_id='end', dag=dag)

   start >> process >> end
   ```

### **Resumen**
- Las **tasks** son las actividades que ejecuta un DAG.
- Los **operators** son plantillas predefinidas que implementan la lógica de ejecución de las tareas.
- Juntos, permiten crear flujos de trabajo dinámicos, reutilizables y escalables en Airflow.

## Scheduler

El **Scheduler** en **Apache Airflow** es el componente central encargado de gestionar la ejecución de los DAGs (Directed Acyclic Graphs) y las tareas definidas en ellos. Es responsable de planificar, desencadenar y supervisar las tareas según las dependencias y los intervalos de tiempo especificados.

## **Funciones principales del Scheduler**

1. **Planificación de DAGs**:
   - Detecta automáticamente los DAGs disponibles y calcula los intervalos de ejecución para ellos.
   - Identifica las tareas que están listas para ejecutarse según sus dependencias y programación.

2. **Asignación de tareas a los workers**:
   - Determina qué tareas deben ejecutarse y las asigna a los **workers** para su ejecución.

3. **Supervisión del estado de las tareas**:
   - Supervisa continuamente el estado de las tareas: `queued`, `running`, `success`, `failed`, etc.
   - Reintenta tareas en caso de fallos si está configurado.

4. **Ejecuta tareas programadas o manuales**:
   - Procesa ejecuciones automáticas basadas en el parámetro `schedule_interval` del DAG.
   - Maneja ejecuciones manuales iniciadas por usuarios desde la interfaz de usuario o la línea de comandos.

## **Flujo de trabajo del Scheduler**

1. **Carga de DAGs**:
   - El Scheduler analiza los DAGs definidos en los archivos de Python dentro del directorio especificado (`dags_folder`).
   - Verifica si hay nuevas ejecuciones pendientes basadas en las definiciones de los DAGs.

2. **Determinación de las tareas ejecutables**:
   - Evalúa las dependencias entre tareas para determinar cuáles están listas para ejecutarse.
   - Considera las configuraciones como:
     - `start_date`: Fecha desde la cual debe comenzar a ejecutarse el DAG.
     - `schedule_interval`: Frecuencia de ejecución del DAG.
     - `catchup`: Si debe ejecutarse para intervalos pasados o solo para el más reciente.

3. **Cola de tareas**:
   - Las tareas listas se colocan en una cola para ser recogidas por los workers disponibles.

4. **Monitoreo continuo**:
   - El Scheduler sigue monitoreando los DAGs y tareas para desencadenar nuevas ejecuciones y manejar fallos.

## **Configuración del Scheduler**

El Scheduler puede configurarse desde el archivo `airflow.cfg` bajo la sección `[scheduler]`. Algunas opciones clave incluyen:

1. **`scheduler_heartbeat_sec`**:
   - Intervalo en segundos en que el Scheduler verifica el estado de los DAGs.

2. **`min_file_process_interval`**:
   - Tiempo mínimo entre análisis de los archivos del directorio de DAGs.

3. **`num_runs`**:
   - Número máximo de ejecuciones antes de reiniciar el proceso del Scheduler.

4. **`max_threads`**:
   - Número de subprocesos que puede utilizar el Scheduler para procesar tareas simultáneamente.

5. **`dag_dir_list_interval`**:
   - Intervalo para buscar cambios en el directorio de DAGs.

## **Inicio del Scheduler**

El Scheduler se ejecuta como un servicio continuo que procesa los DAGs y las tareas. Para iniciar el Scheduler, puedes usar el siguiente comando en la línea de comandos:

```bash
airflow scheduler
```

Este comando:
- Comienza a analizar los DAGs en el directorio especificado.
- Gestiona las tareas programadas en función de sus dependencias y disponibilidad de recursos.

## **Ejemplo práctico**

Supongamos que tienes un DAG que ejecuta una tarea diaria para procesar datos. El Scheduler:
1. Verifica si es necesario ejecutar el DAG en función de su `schedule_interval`.
2. Evalúa las dependencias entre tareas en el DAG.
3. Coloca las tareas listas en la cola.
4. Asigna las tareas a los workers disponibles para ejecutarlas.

### Código de ejemplo:
```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

dag = DAG(
    dag_id='daily_processing',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

task = BashOperator(
    task_id='process_data',
    bash_command='echo "Procesando datos..."',
    dag=dag,
)
```

En este caso:
- El Scheduler verificará diariamente si la tarea `process_data` debe ejecutarse.
- Si está lista, la colocará en la cola para que un worker la procese.

## **Interacción con otros componentes**

1. **Workers**:
   - El Scheduler delega la ejecución de tareas a los workers disponibles.

2. **Base de datos metadata**:
   - El Scheduler consulta y actualiza continuamente el estado de los DAGs y tareas en la base de datos.

3. **Webserver**:
   - Permite a los usuarios monitorear y controlar las ejecuciones desde la interfaz web, interactuando indirectamente con el Scheduler.

## **Buenas prácticas para el Scheduler**

1. **Optimizar la cantidad de DAGs**:
   - Evita tener demasiados DAGs pequeños o con tareas innecesarias que sobrecarguen al Scheduler.

2. **Usar dependencias correctamente**:
   - Configura las dependencias de las tareas para evitar bloqueos innecesarios.

3. **Configurar el número adecuado de workers**:
   - Asegúrate de que los workers disponibles puedan procesar las tareas programadas eficientemente.

4. **Monitorear el rendimiento del Scheduler**:
   - Usa herramientas de monitoreo para verificar si el Scheduler está procesando los DAGs a tiempo y eficientemente.

Con estas consideraciones, el Scheduler puede manejar flujos de trabajo complejos y escalar según las necesidades del proyecto.

## Instalación de Apache Airflow usando Docker

La instalación de **Apache Airflow** utilizando **Docker** es una de las formas más sencillas y flexibles de implementar Airflow, ya que utiliza contenedores para encapsular las dependencias necesarias. Aquí está una guía paso a paso para instalarlo con Docker:

## **Requisitos previos**

1. **Docker** y **Docker Compose** instalados en tu sistema:
   - Para instalar Docker y Docker Compose:
     - En sistemas basados en Linux:
       ```bash
       sudo apt update
       sudo apt install docker.io docker-compose
       ```
     - En Windows/MacOS, descarga e instala Docker Desktop desde [Docker](https://www.docker.com/).

2. Al menos **4 GB de RAM** disponible.

3. Familiaridad con la línea de comandos.

## **Pasos para la instalación**

### **1. Configura un directorio de proyecto**
Crea un directorio donde se almacenarán los archivos necesarios:
```bash
mkdir airflow-docker
cd airflow-docker
```

### **2. Crea el archivo `docker-compose.yml`**
En este archivo se define la configuración del clúster de Airflow. Dentro del directorio creado, crea un archivo llamado `docker-compose.yml`:
```bash
touch docker-compose.yml
```

Copia y pega el siguiente contenido en el archivo:

```yaml
version: '3.7'
x-airflow-common:
  &airflow-common
  image: apache/airflow:2.7.2
  environment:
    &airflow-env
    AIRFLOW__CORE__EXECUTOR: CeleryExecutor
    AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CELERY__BROKER_URL: redis://redis:6379/0
    AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres/airflow
    AIRFLOW__CORE__FERNET_KEY: ""
    AIRFLOW__WEBSERVER__SECRET_KEY: ""
  user: "${AIRFLOW_UID:-50000}:0"
  volumes:
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins
    - ./config/airflow.cfg:/opt/airflow/airflow.cfg
  restart: always

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres_db:/var/lib/postgresql/data
    restart: always

  redis:
    image: redis:6
    restart: always

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    depends_on:
      - postgres
      - redis

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - 8080:8080
    depends_on:
      - postgres
      - redis
      - airflow-scheduler

  airflow-worker:
    <<: *airflow-common
    command: celery worker
    depends_on:
      - redis
      - postgres

  airflow-init:
    <<: *airflow-common
    command: bash -c "airflow db init && airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com"
    depends_on:
      - postgres
      - redis

volumes:
  postgres_db:
  airflow_logs:
  airflow_plugins:
```

### **3. Establece permisos para Airflow**
Asegúrate de que el contenedor tenga permisos para acceder a los volúmenes:
```bash
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

### **4. Inicializa la base de datos**
Ejecuta el comando para inicializar el servicio:
```bash
docker-compose up airflow-init
```

Este comando:
- Inicializa la base de datos de Airflow.
- Crea un usuario administrador con las credenciales predeterminadas:
  - Usuario: `admin`
  - Contraseña: `admin`.

### **5. Inicia los servicios de Airflow**
Levanta los servicios utilizando el siguiente comando:
```bash
docker-compose up -d
```

Este comando:
- Inicia los contenedores de Airflow (webserver, scheduler, worker, redis y PostgreSQL).

### **6. Accede a la interfaz web de Airflow**
Abre tu navegador y ve a la URL: [http://localhost:8080](http://localhost:8080).

Inicia sesión con las credenciales:
- **Usuario**: `admin`
- **Contraseña**: `admin`.

### **7. Estructura del proyecto**
Una vez que todo esté configurado, tu proyecto tendrá la siguiente estructura:

```
airflow-docker/
│
├── dags/            # Carpeta para tus DAGs
├── logs/            # Carpeta donde se guardan los logs
├── plugins/         # Carpeta para plugins personalizados
├── config/          # Archivo de configuración de Airflow
├── docker-compose.yml
├── .env
```

### **8. Administración de los servicios**
- Para detener los servicios:
  ```bash
  docker-compose down
  ```
- Para reiniciar los servicios:
  ```bash
  docker-compose restart
  ```

### **Consejos adicionales**
- **Extender Airflow**:
  Puedes agregar más operadores o plugins personalizados colocando los archivos correspondientes en la carpeta `plugins/`.

- **Gestión de DAGs**:
  Coloca tus DAGs (archivos `.py`) en la carpeta `dags/`. Airflow detectará automáticamente los nuevos DAGs.

- **Escalabilidad**:
  Puedes escalar los workers aumentando el número de contenedores del servicio `airflow-worker`:
  ```bash
  docker-compose up --scale airflow-worker=3
  ```

Ahora tienes un entorno de Apache Airflow completamente funcional utilizando Docker, listo para ejecutar flujos de trabajo complejos.

#### Instalación de clase 

1. descarage el .yaml en `curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.10.3/docker-compose.yaml'` para linux y en powershell se utiliza `Invoke-WebRequest -Uri 'https://airflow.apache.org/docs/apache-airflow/2.10.3/docker-compose.yaml' -OutFile 'docker-compose.yaml'`

2. en la terminal levantamos el compose con `docker-compose up`

**Archivos de la clase**

[instalacion-airflow-usando-docker.pdf](https://static.platzi.com/media/public/uploads/instalacion-airflow-usando-docker_7b7a4efa-df4c-43df-847d-4c5b2d158a57.pdf)

**Lecturas recomendadas**

[Apache Airflow](https://airflow.apache.org/)

[Curso de Docker [Empieza Gratis] - Platzi](https://platzi.com/cursos/docker/)

[Running Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/start/docker.html#docker-compose-yaml)

## Posibles configuraciones

Aquí tienes varias configuraciones posibles para ejecutar Apache Airflow con Docker según tus necesidades. Cada configuración aborda diferentes escenarios y requisitos.

### **Configuración Básica**
Ideal para pruebas locales y proyectos pequeños.

#### Requisitos:
- Archivo `docker-compose.yaml` básico.
- Usuario local configurado.

```yaml
version: '3.7'
services:
  airflow-webserver:
    image: apache/airflow:2.10.3
    container_name: airflow_webserver
    ports:
      - "8080:8080"
    environment:
      - AIRFLOW__CORE__LOAD_EXAMPLES=False
      - AIRFLOW__WEBSERVER__WORKER_REFRESH_BATCH_SIZE=5
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
    depends_on:
      - airflow-init
  airflow-scheduler:
    image: apache/airflow:2.10.3
    container_name: airflow_scheduler
    depends_on:
      - airflow-webserver
  airflow-init:
    image: apache/airflow:2.10.3
    container_name: airflow_init
    entrypoint: ["airflow", "db", "init"]
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
```

### **Configuración con Base de Datos Externa**
Si necesitas usar una base de datos más robusta (como PostgreSQL) en lugar de SQLite.

#### Requisitos:
- Servicio de PostgreSQL.
- Credenciales configuradas.

```yaml
version: '3.7'
services:
  postgres:
    image: postgres:13
    container_name: airflow_postgres
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    ports:
      - "5432:5432"
  airflow-webserver:
    image: apache/airflow:2.10.3
    container_name: airflow_webserver
    environment:
      - AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
    ports:
      - "8080:8080"
    depends_on:
      - postgres
  airflow-init:
    image: apache/airflow:2.10.3
    container_name: airflow_init
    entrypoint: ["airflow", "db", "init"]
    environment:
      - AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow
```

### **Configuración con Celery**
Para escalabilidad en la ejecución de tareas distribuidas.

#### Requisitos:
- Redis como backend de mensajes.
- Varias instancias de worker.

```yaml
version: '3.7'
services:
  redis:
    image: redis:6
    container_name: airflow_redis
    ports:
      - "6379:6379"
  airflow-webserver:
    image: apache/airflow:2.10.3
    container_name: airflow_webserver
    environment:
      - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
      - AIRFLOW__CELERY__RESULT_BACKEND=db+sqlite:///airflow/airflow.db
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
    ports:
      - "8080:8080"
    depends_on:
      - redis
  airflow-worker:
    image: apache/airflow:2.10.3
    container_name: airflow_worker
    environment:
      - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
      - AIRFLOW__CELERY__RESULT_BACKEND=db+sqlite:///airflow/airflow.db
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
    depends_on:
      - redis
  airflow-init:
    image: apache/airflow:2.10.3
    container_name: airflow_init
    entrypoint: ["airflow", "db", "init"]
```

### **Configuración para Desarrollo**
Si quieres montar volúmenes locales para edición dinámica de DAGs y plugins.

#### Configuración de Volúmenes:
```yaml
volumes:
  dags:
    driver: local
  logs:
    driver: local
  plugins:
    driver: local
```

#### Docker Compose:
```yaml
services:
  airflow-webserver:
    image: apache/airflow:2.10.3
    ports:
      - "8080:8080"
    volumes:
      - ./dags:/opt/airflow/dags
      - ./plugins:/opt/airflow/plugins
```

### **Configuración Multiusuario**
Para entornos donde varias personas necesitan trabajar con un servidor web compartido.

#### Ambiente:
- Agrega soporte para autenticación (por ejemplo, con LDAP o OAuth).

```yaml
environment:
  - AIRFLOW__WEBSERVER__AUTH_BACKEND=airflow.providers.google.cloud.auth_backend.google_auth
```

Configura las credenciales de Google OAuth siguiendo la [guía oficial de Airflow](https://airflow.apache.org/docs/apache-airflow/stable/security/auth-backends.html).

### **Notas Generales**
1. **Comando de Inicialización**: Siempre ejecuta antes:
   ```bash
   docker-compose up airflow-init
   ```
2. **Archivo `.env`**: Utiliza un archivo `.env` para gestionar variables sensibles:
   ```env
   AIRFLOW_UID=50000
   ```

1. ver contenedores corriendo `docker ps` 
2. para ingresar al contenedo `docker exec -it <CONTEINER_ID> bash`

**Lecturas recomendadas**

[Configuration Reference — Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html)

## Variables y conexiones

En Apache Airflow, **variables** y **conexiones** son componentes clave que te permiten configurar y manejar el flujo de trabajo de manera más flexible y segura. Aquí te explico cada uno y cómo usarlos:

### 1. **Variables**
Las **variables** en Airflow son pares clave-valor que puedes usar para almacenar información reutilizable y configurable, como rutas de archivos, parámetros de conexión a APIs, y otros valores de configuración que se pueden necesitar en las tareas de los DAGs.

#### Uso de Variables:
- **Creación de Variables**: Puedes crear variables en la interfaz web de Airflow (en **Admin > Variables**) o a través de la CLI de Airflow.
  
  **CLI**:
  ```bash
  airflow variables set MY_VARIABLE "value"
  ```

  **Interfaz Web**:
  - Ve a **Admin > Variables**.
  - Haz clic en **+** para agregar una nueva variable.

- **Acceso a Variables**:
  Puedes acceder a las variables en tus DAGs de la siguiente manera:
  ```python
  from airflow.models import Variable

  my_value = Variable.get("MY_VARIABLE", default_var="default_value")
  ```

#### Usos Comunes:
- Almacenar configuraciones de acceso, como claves de API.
- Rutas de archivos de entrada/salida.
- Parámetros de ejecución de DAGs que pueden cambiar entre entornos (producción, desarrollo, etc.).

### 2. **Conexiones**
Las **conexiones** son configuraciones que se utilizan para conectar Airflow a diversas fuentes de datos y servicios externos, como bases de datos, APIs, servidores de almacenamiento en la nube, y otros servicios de terceros.

#### Uso de Conexiones:
- **Configuración de Conexiones**: Puedes configurarlas en la interfaz web de Airflow en **Admin > Connections** o usando la CLI.

  **CLI**:
  ```bash
  airflow connections add 'my_conn_id' \
      --conn-type 'postgres' \
      --conn-host 'localhost' \
      --conn-login 'username' \
      --conn-password 'password' \
      --conn-schema 'my_database'
  ```

  **Interfaz Web**:
  - Ve a **Admin > Connections**.
  - Haz clic en **+** para agregar una nueva conexión.

- **Acceso a Conexiones**:
  Para usar una conexión en tus DAGs o tareas, puedes hacer uso de `BaseHook` o especificar el ID de la conexión en los operadores:
  ```python
  from airflow.hooks.base import BaseHook

  conn_id = 'my_conn_id'
  conn = BaseHook.get_connection(conn_id)
  print(conn.host)  # Accede al host de la conexión
  ```

#### Ejemplos de Uso de Conexiones:
- **Base de datos**: Conectar a una base de datos PostgreSQL, MySQL, etc.
- **Servicios en la nube**: Conectar a AWS, Google Cloud, Azure, etc.
- **APIs externas**: Conectar a servicios como Twitter, Slack, etc.

### **Ventajas de Usar Variables y Conexiones**
- **Flexibilidad**: Permiten modificar la configuración sin necesidad de cambiar el código de tus DAGs.
- **Seguridad**: Puedes almacenar valores sensibles de forma segura en la interfaz web de Airflow y en el backend de Airflow, en lugar de tenerlos codificados en tu script.
- **Centralización**: Gestionar configuraciones desde la interfaz web facilita la administración y el mantenimiento.

### **Consideraciones de Seguridad**
- **Conexiones sensibles**: Evita almacenar credenciales directamente en el archivo `docker-compose.yml` o en tu código. Usa Airflow para gestionar conexiones de forma centralizada.
- **Variables cifradas**: Airflow permite cifrar las variables para proteger datos sensibles. Configura esto en el archivo `airflow.cfg` con la opción `encrypt_s3_variables`.

### **Ejemplo de Uso en un DAG**
```python
from airflow.models import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from airflow.hooks.base import BaseHook

dag = DAG(
    'my_dag',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False
)

start = DummyOperator(
    task_id='start',
    dag=dag
)

my_conn_id = 'my_database_connection'
conn = BaseHook.get_connection(my_conn_id)

print(f"Host: {conn.host}, Login: {conn.login}")

# Tareas adicionales aquí...
```

Usar **variables** y **conexiones** de esta manera te permite hacer que tus DAGs sean más flexibles, escalables y fáciles de mantener.

**Lecturas recomendadas**

[Configuration Reference — Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html)

[Managing Variables — Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/stable/howto/variable.html)

[airflow.operators — Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/index.html?highlight=operators#module-airflow.operators)

## Implementando un DAG

Para implementar un DAG (Directed Acyclic Graph) en Apache Airflow, necesitas crear un archivo de Python en la carpeta de `dags` de tu proyecto y definir la estructura del DAG, incluyendo las tareas que deben ejecutarse y sus dependencias. A continuación, te explico los pasos detallados para implementar un DAG en Airflow.

### 1. **Configuración del entorno**
Asegúrate de que Apache Airflow esté instalado y en funcionamiento en tu entorno. Si estás usando Docker o un entorno virtual, verifica que Airflow esté corriendo correctamente.

### 2. **Estructura básica de un archivo DAG**
Los archivos de definición de un DAG deben guardarse en la carpeta de `dags` de tu instalación de Airflow (por ejemplo, `airflow/dags/`). Cada archivo debe tener un nombre único y la extensión `.py`.

Aquí tienes un ejemplo de cómo implementar un DAG básico:

```python
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Definición de la función de Python que se ejecutará en la tarea
def print_hello_world():
    print("¡Hola, Airflow!")

# Argumentos por defecto para el DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definición del DAG
with DAG(
    'my_first_dag',
    default_args=default_args,
    description='Este es mi primer DAG en Airflow',
    schedule_interval=timedelta(days=1),  # Ejecución diaria
    start_date=datetime(2024, 1, 1),
    catchup=False,  # No ejecutar tareas pasadas
) as dag:

    # Definición de las tareas
    start_task = DummyOperator(task_id='start')
    
    python_task = PythonOperator(
        task_id='print_hello',
        python_callable=print_hello_world
    )
    
    end_task = DummyOperator(task_id='end')

    # Definición del flujo de tareas
    start_task >> python_task >> end_task
```

### 3. **Explicación de la estructura**
- **Importación de módulos**: Se importan `DAG` para definir el DAG, y operadores como `DummyOperator` y `PythonOperator` para definir las tareas.
- **Definición de la función de Python**: Una función llamada `print_hello_world` que imprime un mensaje, que se ejecutará como una tarea.
- **Argumentos por defecto**: Se definen `default_args` que aplican a todas las tareas del DAG (ejemplo: reintentos, correo electrónico en caso de fallo, etc.).
- **Definición del DAG**:
  - `schedule_interval`: Define la frecuencia con la que el DAG se ejecuta. En este ejemplo, es diario (`timedelta(days=1)`).
  - `start_date`: Fecha de inicio de la programación.
  - `catchup`: Se establece en `False` para evitar la ejecución de tareas pasadas.
- **Definición de tareas**: Se crean tareas usando operadores como `DummyOperator` y `PythonOperator`.
- **Dependencias de tareas**: Se definen utilizando el operador `>>` para establecer el orden de ejecución.

### 4. **Ejecución y monitoreo**
- Guarda el archivo en la carpeta de `dags` de tu instalación de Airflow.
- Accede a la interfaz web de Airflow (`http://localhost:8080` por defecto) para verificar que el DAG se haya cargado y está en la lista de DAGs.
- Activa el DAG y supervisa su ejecución desde la interfaz web para ver cómo se ejecutan las tareas y su estado.

### 5. **Ejemplo de un DAG más complejo**
Aquí tienes un ejemplo de un DAG con tareas adicionales y notificaciones por correo electrónico:

```python
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta

def process_data():
    print("Procesando los datos...")

def notify_success():
    print("Notificación de éxito enviada.")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['your-email@example.com'],
    'retries': 3,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    'complex_dag_example',
    default_args=default_args,
    description='Un DAG más avanzado con notificaciones y procesamiento de datos',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    start = DummyOperator(task_id='start')
    process = PythonOperator(task_id='process_data', python_callable=process_data)
    notify = PythonOperator(task_id='notify_success', python_callable=notify_success)
    end = DummyOperator(task_id='end')

    start >> process >> notify >> end
```

### 6. **Conclusión**
Implementar un DAG en Apache Airflow es un proceso sencillo una vez que comprendes la estructura y cómo se definen las tareas y sus dependencias. Con esta guía, deberías poder crear y ejecutar tus propios DAGs para automatizar flujos de trabajo y realizar análisis o procesamiento de datos.

## Bash Operator

El **BashOperator** en Apache Airflow permite ejecutar comandos de Bash en un flujo de trabajo. Es útil para realizar tareas como mover archivos, ejecutar scripts de shell, o interactuar con herramientas del sistema operativo desde un DAG.

### **Estructura Básica del BashOperator**
Aquí tienes un ejemplo básico de cómo usar el `BashOperator`:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# Definir el DAG
with DAG(
    dag_id="bash_operator_example",
    description="Ejemplo de uso del BashOperator",
    start_date=datetime(2024, 11, 28),  # Fecha en el pasado
    schedule_interval="@daily",         # Se ejecuta diariamente
    catchup=False,                      # No ejecuta tareas pendientes de fechas pasadas
) as dag:

    # Definir una tarea usando BashOperator
    t1 = BashOperator(
        task_id="print_date",
        bash_command="date",  # Comando de Bash que imprime la fecha actual
    )

    # Otra tarea para ejecutar un script
    t2 = BashOperator(
        task_id="run_script",
        bash_command="echo 'Ejecutando mi script'; ./mi_script.sh",  # Comando Bash con script
    )

    # Definir dependencias entre tareas
    t1 >> t2
```

### **Parámetros principales**
El `BashOperator` tiene varios parámetros que puedes configurar:

| **Parámetro**       | **Descripción**                                                                                     |
|---------------------|-----------------------------------------------------------------------------------------------------|
| `task_id`           | Identificador único de la tarea.                                                                    |
| `bash_command`      | Comando de Bash que se ejecutará. Puede incluir múltiples líneas.                                    |
| `env`               | Diccionario para configurar variables de entorno que se usarán durante la ejecución del comando.    |
| `cwd`               | Cambia el directorio de trabajo donde se ejecutará el comando.                                      |
| `execution_timeout` | Tiempo máximo permitido para que el comando termine antes de ser interrumpido.                      |

### **Ejemplo con variables de entorno**
Puedes pasar variables de entorno al comando de Bash:

```python
t3 = BashOperator(
    task_id="custom_environment",
    bash_command="echo 'El usuario actual es: $USER'",
    env={"USER": "airflow_user"}  # Establece la variable USER
)
```

### **Logs de la ejecución**
Airflow registra los logs de cada ejecución, por lo que puedes verificar los resultados de tu comando en el log del operador. Para ver los logs:
1. Ve al **Airflow UI**.
2. Selecciona el DAG y luego la tarea.
3. Haz clic en **View Log**.

### **Errores comunes**
1. **Permisos insuficientes**: Asegúrate de que los scripts o comandos tengan los permisos necesarios.
2. **Rutas incorrectas**: Si usas rutas relativas, verifica que el directorio de trabajo sea el correcto.
3. **Dependencias del sistema**: Si tu comando requiere herramientas externas, verifica que estén instaladas en el entorno donde se ejecuta Airflow.

### **Avanzado: Plantillas Jinja**
El `bash_command` soporta **plantillas Jinja**, lo que permite usar variables dinámicas como fechas de ejecución:

```python
t4 = BashOperator(
    task_id="templated_command",
    bash_command="echo 'Fecha de ejecución: {{ ds }}'",  # `ds` es la fecha de ejecución
)
```

Este ejemplo imprime la fecha de ejecución del DAG (por ejemplo, `2024-11-28`).

¡Con estos ejemplos deberías poder usar el `BashOperator` con confianza! 😊

## Definiendo dependencias entre tareas

En Apache Airflow, las dependencias entre tareas se definen usando operadores que establecen relaciones de ejecución. Estas dependencias determinan el orden en el que las tareas deben ejecutarse en el DAG.

### Métodos para definir dependencias

1. **Usando el operador `>>` (hacia adelante):**
   Este operador indica que una tarea debe ejecutarse antes de otra.

   ```python
   task1 >> task2  # task1 se ejecuta antes de task2
   ```

2. **Usando el operador `<<` (hacia atrás):**
   Este operador indica que una tarea debe ejecutarse después de otra.

   ```python
   task1 << task2  # task2 se ejecuta antes de task1
   ```

3. **Definiendo dependencias múltiples:**
   Puedes definir dependencias entre varias tareas a la vez:

   ```python
   task1 >> [task2, task3]  # task1 se ejecuta antes de task2 y task3
   [task2, task3] >> task4  # task2 y task3 deben completarse antes de ejecutar task4
   ```

4. **Usando el método `.set_downstream()` y `.set_upstream()`:**
   Estos métodos establecen relaciones explícitas entre tareas.

   ```python
   task1.set_downstream(task2)  # Igual a task1 >> task2
   task2.set_upstream(task1)    # Igual a task1 >> task2
   ```

### Ejemplo práctico
```python
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from datetime import datetime

# Definir el DAG
with DAG(
    dag_id="dependencias_dag",
    description="Definiendo dependencias entre tareas",
    start_date=datetime(2024, 11, 28),
    schedule_interval="@once",
) as dag:
    # Tareas
    inicio = EmptyOperator(task_id="inicio")
    procesar_datos = EmptyOperator(task_id="procesar_datos")
    generar_reporte = EmptyOperator(task_id="generar_reporte")
    fin = EmptyOperator(task_id="fin")

    # Definir dependencias
    inicio >> procesar_datos >> generar_reporte >> fin
```

### Resultado
En el ejemplo anterior:
1. La tarea `inicio` debe completarse antes de `procesar_datos`.
2. `procesar_datos` debe completarse antes de `generar_reporte`.
3. Finalmente, `generar_reporte` debe completarse antes de `fin`.

### Visualización
Cuando el DAG se carga correctamente, las dependencias se pueden observar en el interfaz de Airflow como un flujo claro entre las tareas. Esto asegura un orden lógico y ejecutable en el proceso.

## Custom Operator

En Apache Airflow, un **Custom Operator** permite extender las funcionalidades de los operadores estándar definiendo uno propio. Esto es útil cuando necesitas realizar tareas específicas que no están cubiertas por los operadores existentes.

### Pasos para crear un Custom Operator

1. **Importar las clases necesarias:**
   - `BaseOperator`: es la clase base para todos los operadores de Airflow.
   - `apply_defaults`: facilita el manejo de parámetros para el operador.

2. **Definir tu operador personalizado:**
   Heredas de `BaseOperator` y defines la lógica principal en el método `execute()`.

3. **Registrar parámetros:**
   Puedes pasar parámetros personalizados al operador y utilizarlos en la ejecución.

4. **Usar el operador en un DAG:**
   Una vez definido, el operador personalizado se utiliza como cualquier otro operador en un DAG.

### Ejemplo de un Custom Operator
Este operador escribe un mensaje personalizado en un archivo de texto.

#### Archivo del operador personalizado (`custom_operator.py`):
```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

class CustomWriteOperator(BaseOperator):
    @apply_defaults
    def __init__(self, file_path: str, message: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.message = message

    def execute(self, context):
        self.log.info("Escribiendo mensaje en el archivo...")
        with open(self.file_path, "w") as file:
            file.write(self.message)
        self.log.info(f"Mensaje escrito: {self.message}")
```

#### Archivo del DAG (`custom_operator_dag.py`):
```python
from airflow import DAG
from custom_operator import CustomWriteOperator  # Importar el operador personalizado
from datetime import datetime

# Definir el DAG
with DAG(
    dag_id="custom_operator_dag",
    description="Ejemplo de Custom Operator",
    start_date=datetime(2024, 11, 28),
    schedule_interval="@once",
) as dag:
    # Instancia del operador personalizado
    escribir_mensaje = CustomWriteOperator(
        task_id="escribir_mensaje",
        file_path="/tmp/mensaje.txt",
        message="¡Hola desde el operador personalizado!",
    )
```

### Explicación del código

1. **Clase `CustomWriteOperator`:**
   - `__init__`: inicializa los parámetros personalizados (`file_path` y `message`).
   - `execute`: contiene la lógica principal que se ejecuta cuando el DAG corre.

2. **Archivo del DAG:**
   - El DAG utiliza el operador personalizado `CustomWriteOperator` para escribir un mensaje en un archivo.

3. **Ejecución:**
   - Cuando el DAG se ejecuta, el operador crea un archivo en `/tmp/mensaje.txt` y escribe el mensaje proporcionado.

### Pruebas del operador
- Asegúrate de que el archivo `custom_operator.py` esté en la carpeta `dags` o en una ruta incluida en el `PYTHONPATH`.
- Verifica el registro de logs en la interfaz de Airflow para confirmar la ejecución del operador.

### Aplicaciones de operadores personalizados
- Automatización de tareas específicas como consultas API personalizadas.
- Procesos únicos de transformación de datos.
- Integraciones con herramientas o sistemas no soportados nativamente por Airflow.

Esto te permite adaptar Airflow a las necesidades exactas de tus proyectos.

## Orquestando un DAG I

Un DAG (Directed Acyclic Graph) en el contexto de la orquestación de tareas (como Apache Airflow) es una estructura que define la secuencia y las dependencias entre las tareas que se ejecutan como parte de un flujo de trabajo. Si estás interesado en aprender o trabajar con un DAG, te puedo ayudar con los siguientes pasos:

### 1. **Entender los Componentes de un DAG**
   - **Nodos**: Representan las tareas individuales.
   - **Aristas (Edges)**: Representan las dependencias entre las tareas.
   - **Atributos del DAG**: Incluyen el identificador, la programación (schedule), y los parámetros globales.

### 2. **Instalar Herramientas Necesarias**
   - Si estás usando Apache Airflow, instala el paquete con:
     ```bash
     pip install apache-airflow
     ```
   - Configura una base de datos para Airflow y el servidor web.

### 3. **Crear un DAG Básico**
   Un ejemplo básico de código para crear un DAG en Python usando Airflow:

   ```python
   from airflow import DAG
   from airflow.operators.python import PythonOperator
   from datetime import datetime, timedelta

   # Define una función de ejemplo
   def print_hello():
       print("Hola, este es un DAG de prueba.")

   # Configuración del DAG
   default_args = {
       'owner': 'Mario Alexander Vargas Celis',
       'depends_on_past': False,
       'email_on_failure': False,
       'email_on_retry': False,
       'retries': 1,
       'retry_delay': timedelta(minutes=5),
   }

   with DAG(
       'dag_prueba',
       default_args=default_args,
       description='Un DAG simple para imprimir un mensaje',
       schedule_interval=timedelta(days=1),
       start_date=datetime(2024, 1, 1),
       catchup=False,
   ) as dag:
       
       # Tarea
       tarea_hello = PythonOperator(
           task_id='tarea_hello',
           python_callable=print_hello,
       )
   ```

### 4. **Definir Dependencias entre Tareas**
   Las dependencias en Airflow se definen con operadores como `>>` (para indicar que una tarea debe ejecutarse antes de otra):
   ```python
   tarea_hello >> otra_tarea
   ```

### 5. **Ejecutar el DAG**
   1. Inicializa la base de datos:
      ```bash
      airflow db init
      ```
   2. Inicia el servidor:
      ```bash
      airflow webserver --port 8080
      ```
   3. Corre el planificador:
      ```bash
      airflow scheduler
      ```

   Luego, puedes visitar la interfaz en `http://localhost:8080` y monitorear la ejecución del DAG.

**Lecturas recomendadas**

[DAG Runs — Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/stable/dag-run.html?highlight=cron)

[Crontab.guru - The cron schedule expression editor](https://crontab.guru/)

## Orquestando un DAG II

En la segunda etapa de "Orquestando un DAG", profundizamos en conceptos avanzados y optimizaciones para manejar tareas más complejas. Aquí exploraremos técnicas clave para escalar, depurar, y mejorar la eficiencia en el diseño y la ejecución de DAGs.

### **1. Definiendo Dependencias Complejas**
A medida que tu flujo de trabajo crece, es posible que necesites manejar múltiples dependencias entre tareas:

- **Dependencias Lineales**:
  ```python
  tarea_1 >> tarea_2 >> tarea_3
  ```
- **Dependencias Ramificadas**:
  ```python
  [tarea_1, tarea_2] >> tarea_3
  tarea_3 >> [tarea_4, tarea_5]
  ```
  
- **Configuración Dinámica de Dependencias**:
  Si las tareas dependen de un número variable de entradas:
  ```python
  for i in range(5):
      previous_task >> PythonOperator(
          task_id=f'tarea_{i}',
          python_callable=funcion_dinamica,
      )
  ```

### **2. Uso de Sensores**
Los sensores son operadores especiales que esperan un evento o condición antes de continuar. Por ejemplo, esperar a que un archivo se cree:

```python
from airflow.sensors.filesystem import FileSensor

esperar_archivo = FileSensor(
    task_id='esperar_archivo',
    filepath='/ruta/al/archivo',
    poke_interval=30,  # Verifica cada 30 segundos
    timeout=600,       # Expira después de 10 minutos
)
```

### **3. Paralelismo y Pools**
Para flujos de trabajo grandes, el paralelismo optimiza el uso de recursos:

- **Configurar `concurrency` del DAG**:
  Limita el número máximo de tareas simultáneas en un DAG.
  ```python
  with DAG(
      'dag_con_paralelismo',
      concurrency=10,  # Máximo de 10 tareas a la vez
      ...
  )
  ```

- **Usar Pools**:
  Agrupa tareas para compartir recursos específicos:
  ```bash
  airflow pools set pool_name 5 "Descripción del pool"
  ```

  Luego, asigna el pool en las tareas:
  ```python
  tarea_optimizada = PythonOperator(
      task_id='tarea_optimizada',
      python_callable=mi_funcion,
      pool='pool_name',
  )
  ```
### **4. Manejo de Errores y Retries**
Es importante configurar estrategias de manejo de errores para mantener la robustez del DAG:

```python
default_args = {
    'retries': 3,  # Reintenta 3 veces
    'retry_delay': timedelta(minutes=5),  # Espera 5 minutos entre reintentos
    'on_failure_callback': mi_funcion_de_notificacion,
}
```

Además, puedes especificar una tarea en particular que debe ejecutarse en caso de fallos:
```python
tarea_fallida >> tarea_notificar_fallo
```

### **5. Integración con APIs y Scripts Externos**
Es común ejecutar scripts o interactuar con APIs externas desde un DAG. Por ejemplo, usando `BashOperator` o `HttpSensor`:

- **Ejecutar un Script Bash**:
  ```python
  from airflow.operators.bash import BashOperator

  tarea_bash = BashOperator(
      task_id='ejecutar_script',
      bash_command='python3 /ruta/a/mi_script.py',
  )
  ```

- **Esperar una Respuesta de API**:
  ```python
  from airflow.sensors.http import HttpSensor

  esperar_api = HttpSensor(
      task_id='esperar_api',
      http_conn_id='mi_api',
      endpoint='/status',
      response_check=lambda response: response.status_code == 200,
  )
  ```

### **6. Depuración Avanzada**
Para depurar errores en tareas o DAGs complejos:
- **Ver Logs Detallados**:
  Usa la interfaz de Airflow o la CLI:
  ```bash
  airflow tasks logs dag_id task_id execution_date
  ```
  
- **Ejecutar Tareas en Modo Local**:
  ```bash
  airflow tasks test dag_id task_id execution_date
  ```

### **7. Prácticas de Diseño Escalable**
- Divide DAGs grandes en DAGs más pequeños, vinculados mediante **ExternalTaskSensor**.
- Usa **temporalidad dinámica** con el parámetro `execution_date` para manejar tareas dependientes del tiempo.
- Emplea variables o conexiones definidas en Airflow para parametrizar tareas.

**Lecturas recomendadas**

[Crontab.guru - The cron schedule expression editor](https://crontab.guru/)

## Monitoring

### **Monitoring en la Orquestación de DAGs**

El monitoreo es una parte crucial para garantizar que tus flujos de trabajo (DAGs) se ejecuten de manera eficiente, manejando fallos y obteniendo visibilidad en tiempo real de su estado. En el contexto de herramientas como Apache Airflow, aquí tienes las mejores prácticas y herramientas para el monitoreo efectivo:

### **1. Interfaz Web**
La interfaz web de Airflow es la herramienta principal para el monitoreo visual de DAGs:

- **Vista de DAGs**:
  - Observa el estado general de todos los DAGs.
  - Muestra colores para representar el estado de las tareas:
    - Verde: Éxito
    - Rojo: Fallo
    - Amarillo: En ejecución
    - Gris: Sin ejecutar

- **Vista de Gantt**:
  - Proporciona un análisis temporal de las tareas ejecutadas.
  - Ayuda a identificar cuellos de botella.

- **Vista de Logs**:
  - Para cada tarea, puedes acceder a los registros de ejecución.
  - Ideal para depurar errores o evaluar tiempos de ejecución.

### **2. Alertas y Notificaciones**
Configura alertas automáticas para informar sobre fallos o eventos clave:

- **Notificaciones por Correo Electrónico**:
  Configura `email_on_failure` o `email_on_retry` en las tareas:
  ```python
  default_args = {
      'email': ['mario.vargas@example.com'],
      'email_on_failure': True,
      'email_on_retry': False,
  }
  ```

- **Callbacks Personalizados**:
  Usa `on_failure_callback` o `on_success_callback` para realizar acciones específicas, como enviar un mensaje a Slack o registrar errores en un sistema externo:
  ```python
  def notificar_error(context):
      print(f"Tarea fallida: {context['task_instance'].task_id}")

  tarea = PythonOperator(
      task_id='mi_tarea',
      python_callable=mi_funcion,
      on_failure_callback=notificar_error,
  )
  ```

### **3. Métricas y Logs Centralizados**
Integra Airflow con sistemas externos para recolectar y visualizar métricas:

- **Prometheus y Grafana**:
  - Configura el **exportador Prometheus** para Airflow.
  - Visualiza métricas como:
    - Número de tareas completadas.
    - Tiempos promedio de ejecución.
    - Tareas fallidas por DAG.

- **Elasticsearch**:
  - Centraliza los logs de ejecución para búsquedas y análisis más eficientes.

### **4. Manejo de Retries y Fallos**
Supervisa y ajusta las políticas de reintentos en tareas problemáticas:

- **Configurar Retries**:
  ```python
  tarea = PythonOperator(
      task_id='mi_tarea',
      python_callable=mi_funcion,
      retries=3,
      retry_delay=timedelta(minutes=5),
  )
  ```

- **Resúmenes de Errores**:
  La interfaz web permite acceder a listas de tareas fallidas para análisis detallado.

### **5. Auditorías y Seguimiento Histórico**
Monitorea cómo ha evolucionado el rendimiento de tus DAGs a lo largo del tiempo:

- **Historial de Ejecuciones**:
  Usa la vista "Tree View" o "Graph View" para ver el historial y patrones de fallos o ejecuciones exitosas.

- **Exportar Logs**:
  Guarda los registros para auditorías externas:
  ```bash
  airflow tasks logs dag_id task_id execution_date > log.txt
  ```

### **6. Optimización Basada en Monitoreo**
Identifica cuellos de botella y optimiza el rendimiento:
- Observa tareas que consumen mucho tiempo y evalúa su paralelización.
- Usa sensores de manera eficiente, evitando bloqueos prolongados.
- Configura límites de concurrencia y priorización de tareas.

### **7. Integración con Herramientas Externas**
- **Slack**: Notifica fallos directamente a un canal de Slack.
- **PagerDuty**: Alerta en caso de errores críticos en tiempo real.
- **AWS CloudWatch** (si se ejecuta en AWS): Monitorea recursos y ejecuta acciones automáticas en función del uso.