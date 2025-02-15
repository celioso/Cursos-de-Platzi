# Curso de Introducción a AWS Cómputo, Almacenamiento y Bases de Datos

## ¿Ya tomaste el curso de Fundamentos de Cloud Computing?

El profesor Enrique Alexis López Araujo, Data Architect en Platzi y Cloud Practicioner certificado, nos da la bienvenida al curso.

Antes de empezar, u**na recomendación: toma el [Curso de Introducción a AWS: Fundamentos de Cloud Computing](https://platzi.com/cursos/aws-fundamentos/ "Curso de Introducción a AWS: Fundamentos de Cloud Computing")**, que te servirá como base para lo que viene.

En este curso de Introducción a AWS: Cómputo, Almacenamiento y Bases de Datos veremos una introducción a los servicios de cómputo, almacenamiento y bases de datos en AWS.

### ¿Qué más vas a aprender?

Además de una **introducción al cómputo, almacenamiento y bases de datos en AWS**, veremos:

- - Máquinas virtuales, contenedores y computación sin servidor en AWS.
- **Almacenamiento en bloques, archivos y objetos**, y los servicios de AWS para cada tipo de almacenamiento.
- Laboratorios de **cómo usar el servicio de almacenamiento de objetos (S3)**.
- **Bases de datos de tipo relacional, de clave-valor y en memoria** en AWS.

**Archivos de la clase**

[curso-de-introduccion-a-aws-computo-almacenamiento-y-bases-de-datos.pdf](https://static.platzi.com/media/public/uploads/curso-de-introduccion-a-aws-computo-almacenamiento-y-bases-de-datos_591b2667-5cc8-478c-aea6-3f08a6d7a74e.pdf)

**Lecturas recomendadas**

[Curso de Fundamentos de AWS Cloud - Platzi](https://platzi.com/cursos/aws-cloud/)

## Cómputo en AWS

AWS (Amazon Web Services) ofrece varios servicios de cómputo en la nube que permiten ejecutar aplicaciones de manera escalable y flexible. Algunos de los servicios más importantes son:

### 🔹 **Servicios de Cómputo en AWS**
1. **Amazon EC2 (Elastic Compute Cloud)**  
   - Servidores virtuales en la nube.  
   - Permite elegir el tipo de instancia, sistema operativo y capacidad de cómputo.  
   - Escalabilidad vertical y horizontal.

2. **AWS Lambda**  
   - Ejecución de código sin necesidad de gestionar servidores.  
   - Modelo **Serverless**, ideal para eventos y microservicios.  
   - Se paga solo por el tiempo de ejecución.

3. **Amazon ECS (Elastic Container Service)**  
   - Administración de contenedores Docker.  
   - Compatible con AWS Fargate (sin necesidad de gestionar servidores).  

4. **Amazon EKS (Elastic Kubernetes Service)**  
   - Kubernetes gestionado en AWS.  
   - Facilita la administración y escalabilidad de contenedores.  

5. **AWS Fargate**  
   - Servicio sin servidores para ejecutar contenedores en ECS y EKS.  
   - No es necesario administrar infraestructura.  

6. **Amazon Lightsail**  
   - Alternativa más sencilla a EC2 con precios predecibles.  
   - Incluye servidores, almacenamiento y bases de datos en paquetes fijos.  

7. **AWS Batch**  
   - Ejecución eficiente de cargas de trabajo por lotes.  
   - Ideal para procesamiento masivo de datos y análisis.  

8. **Amazon Outposts**  
   - Infraestructura de AWS en centros de datos locales.  
   - Híbrido entre la nube y on-premise.  

### 🔹 **Casos de Uso**
✅ **Aplicaciones Web y Móviles** – EC2, Lambda, Lightsail  
✅ **Big Data y Machine Learning** – Batch, EC2 con instancias GPU  
✅ **Microservicios y Serverless** – Lambda, ECS, EKS  
✅ **Computación de Alto Rendimiento (HPC)** – EC2 con instancias especializadas  

Si estás aprendiendo sobre AWS, dime en qué te gustaría profundizar.

### Resumen

AWS describe su capacidad de cómputo como “**cómputo para cualquier carga de trabajo”. Cómputo se refiere a usar una computadora para procesamiento**, ya sea sumar dos números o alojar un sitio web. Entre los servicios de cómputo se encuentran:

### Instancias o máquinas virtuales

Una **máquina virtual** es un software que simula un sistema operativo, y que puede ejecutar programas dentro de dicho sistema como si fuera una computadora real. Los servicios de máquinas virtuales (o instancias) en AWS son:

- **Amazon EC2**: máquinas virtuales seguras y redimensionables.
- **Amazon EC2 Spot**: cargas de trabajo tolerante a fallas, por hasta el 90% del precio normal (nota: Amazon puede reclamar estas instancias en cualquier momento con solo dos minutos de anticipación).
- **Amazon EC2 AutoScaling**: agrega o elimina automáticamente la capacidad informática para satisfacer tus necesidades bajo demanda.
- **Amazon EC2 LightSail**: plataforma en la nube fácil de usar para crear una aplicación o un sitio web.

### Contenedores

Un **contenedor** es una unidad de *software* que empaca un software en específico junto con sus dependencias. Se diferencian de las máquinas virtuales en que estas virtualizan el hardware, mientras que los contenedores [virtualizan el sistema operativo](https://cloud.google.com/learn/what-are-containers "virtualizan el sistema operativo"). Los servicios de contenedores de AWS son:

- **Amazon Elastic Container Services (ECS)**: servicio para correr contenedores confiables y escalables.
- **Amazon Elastic Container Registry (ECR)**: servicio para almacenar, administrar e implementar imágenes de contenedores.
- **Amazon Elastic Kubernetes Service (EKS)**: servicio de Kubernetes administrado por AWS.

### Serverless

La computación **serverless** se refiere a que **la responsabilidad de administrar servidores o máquinas virtuales se le delega al proveedor de nube**, por lo que sólo debemos precuparnos por el código de nuestras aplicaciones. **Amazon Lambda** nos permite ejecutar piezas de código sin servidores.

### Servicios de borde (Edge)

El [Edge Computing](https://www.xataka.com/internet-of-things/edge-computing-que-es-y-por-que-hay-gente-que-piensa-que-es-el-futuro "Edge Computing") se refiere al **cómputo y procesamiento de datos en una ubicación cercana a la necesaria para el negocio**. Los servicios de borde o edge computing de AWS son:

- **Amazon Outposts**: permite ejecutar los servicios de AWS en nuestros propios servidores en lugar de Amazon.
- **Amazon Snow Family**: es una familia de dispositivos desde un disco duro portátil hasta un semi-remolque completo lleno de discos de almacenamiento. Estos dispositivos te permiten cargar archivos en ellos, para luego ser enviados a Amazon y cargados en sus servidores.
- **AWS Wavelength**: permite acceder a los servicios AWS desde dispositivos 5G sin pasar por Internet.
- **VMWare AWS**: permite migrar cargas de trabajo de VMWare a AWS.
- **AWS Local Zones**: permite ejecutar las aplicaciones más cerca de los usuarios finales, a una menor latencia.

### Conclusión

Exploramos una gran cantidad de servicios de computación en AWS. En las próximas clases veremos estos servicios más en detalle.

## Conoce qué es Amazon EC2

Amazon EC2 es un servicio de **computación en la nube** que proporciona **servidores virtuales** (llamados *instancias*) en AWS. Te permite ejecutar aplicaciones sin necesidad de invertir en infraestructura física, escalando según la demanda.

### **🔹 Características Principales**

✅ **Escalabilidad** – Puedes aumentar o reducir la cantidad de servidores según la carga.  
✅ **Elección de Hardware** – Puedes elegir tipo de procesador, memoria RAM, almacenamiento y red.  
✅ **Diferentes Tipos de Instancias** – Optimizadas para cómputo, memoria, almacenamiento o GPU.  
✅ **Pago por Uso** – Modelos de facturación flexibles según el tiempo y tipo de instancia.  
✅ **Seguridad** – Integración con AWS IAM y Virtual Private Cloud (VPC) para control de acceso.

### **🔹 Tipos de Instancias en EC2**

1. **Instancias de Propósito General** – Uso equilibrado de CPU, memoria y red. (Ej: *t3.micro, m5.large*).  
2. **Optimizadas para Cómputo** – Mayor capacidad de CPU, ideal para cálculos intensivos. (Ej: *c6g.large*).  
3. **Optimizadas para Memoria** – Mayor RAM, útil para bases de datos y análisis de datos. (Ej: *r5.large*).  
4. **Optimizadas para Almacenamiento** – Diseñadas para manejo de grandes volúmenes de datos. (Ej: *i3.large*).  
5. **Instancias GPU** – Para Machine Learning y gráficos intensivos. (Ej: *p3.2xlarge*).  

### **🔹 Modelos de Pago**

💰 **On-Demand** – Pago por hora/segundo sin compromisos.  
💰 **Reserved Instances** – Contrato a 1 o 3 años con descuentos.  
💰 **Spot Instances** – Hasta 90% más baratas, pero pueden ser interrumpidas.  
💰 **Savings Plans** – Planes de ahorro con descuentos a cambio de compromiso de uso.  

### **🔹 Casos de Uso**

✅ Aplicaciones web y backend escalables  
✅ Hosting de sitios y servidores  
✅ Big Data y análisis de datos  
✅ Machine Learning y AI con GPU  
✅ Simulaciones científicas  

### **Amazon EC2 (Elastic Compute Cloud)** 🚀  

Amazon EC2 es un servicio de **computación en la nube** que proporciona **servidores virtuales** (llamados *instancias*) en AWS. Te permite ejecutar aplicaciones sin necesidad de invertir en infraestructura física, escalando según la demanda.

### **🔹 Características Principales**

✅ **Escalabilidad** – Puedes aumentar o reducir la cantidad de servidores según la carga.  
✅ **Elección de Hardware** – Puedes elegir tipo de procesador, memoria RAM, almacenamiento y red.  
✅ **Diferentes Tipos de Instancias** – Optimizadas para cómputo, memoria, almacenamiento o GPU.  
✅ **Pago por Uso** – Modelos de facturación flexibles según el tiempo y tipo de instancia.  
✅ **Seguridad** – Integración con AWS IAM y Virtual Private Cloud (VPC) para control de acceso.

### **🔹 Tipos de Instancias en EC2**

1. **Instancias de Propósito General** – Uso equilibrado de CPU, memoria y red. (Ej: *t3.micro, m5.large*).  
2. **Optimizadas para Cómputo** – Mayor capacidad de CPU, ideal para cálculos intensivos. (Ej: *c6g.large*).  
3. **Optimizadas para Memoria** – Mayor RAM, útil para bases de datos y análisis de datos. (Ej: *r5.large*).  
4. **Optimizadas para Almacenamiento** – Diseñadas para manejo de grandes volúmenes de datos. (Ej: *i3.large*).  
5. **Instancias GPU** – Para Machine Learning y gráficos intensivos. (Ej: *p3.2xlarge*).

### **🔹 Modelos de Pago**
💰 **On-Demand** – Pago por hora/segundo sin compromisos.  
💰 **Reserved Instances** – Contrato a 1 o 3 años con descuentos.  
💰 **Spot Instances** – Hasta 90% más baratas, pero pueden ser interrumpidas.  
💰 **Savings Plans** – Planes de ahorro con descuentos a cambio de compromiso de uso.

### **🔹 Casos de Uso**
✅ Aplicaciones web y backend escalables  
✅ Hosting de sitios y servidores  
✅ Big Data y análisis de datos  
✅ Machine Learning y AI con GPU  
✅ Simulaciones científicas

### Resumen

[EC2](https://platzi.com/clases/1323-aws-cloud-practico/12577-que-es-ec2/ "EC2") **permite alquilar máquinas virtuales, llamadas instancias EC2**. Puedes elegir diferentes tipos de **EC2** con diferente CPU, RAM y almacenamiento. Hay instancias optimizadas para cómputo, memoria y almacenamiento, [entre otras](https://docs.aws.amazon.com/es_es/AWSEC2/latest/UserGuide/instance-types.html "entre otras").

En **EC2**, el sistema de pago más común es por hora o por segundo, dependiendo el tipo de instancia. Por ejemplo, para una instancia que cueste $0.1 la hora, puedes pagar, ya sea una instancia por 24 horas o 24 instancias por una hora. En ambos casos pagas lo mismo (24 * 0.10 = $2.4).

### Opciones y precios bajo demanda

Las instancias pueden redimiensionarse. Puedes empezar por una instancia de bajo costo, y si necesitas aumenta su capacidad, apagas la instancia y seleccionas un nuevo tipo de instancia. Cuando enciendas de nuevo la instancia, verás su capacidad aumentada. La siguiente tabla muestra **algunos tipos de instancias**.

| Nombre | Especificaciones | Precio |
|---|---|---|
| t3.nano | 2 vCPU’s, 0.5 GiB RAM | $0,0052/hora |
| t3.xlarge | 4 vCPU’s, 16 GiB RAM | $0,1664/hora |
| c6g.8xlarge | 32 vCPU’s, 64 GiB RAM | $1,088/hora |
| X1e.xlarge | 128 vCPU’s, 3904 GiB RAM, 2x 1920 GB SSD | $26,688/hora |

### Hosts dedicados

Los hosts dedicados en Amazon Web Services (AWS) son infraestructuras de servidores físicos que ofrecen un nivel exclusivo de recursos computacionales para las cargas de trabajo de los clientes. En lugar de compartir estos servidores con otros usuarios, los hosts dedicados permiten a los clientes tener un control más granular sobre la ubicación y asignación de sus instancias de Amazon EC2. Esto puede ser beneficioso para aplicaciones que requieren una mayor seguridad, cumplimiento normativo o rendimiento constante.

Los hosts dedicados también brindan la flexibilidad de llevar licencias de software existentes a la nube sin incurrir en costos adicionales. Al utilizar hosts dedicados, los principiantes en AWS pueden garantizar una mayor aislación de recursos y una mayor predictibilidad en el rendimiento de sus aplicaciones, al tiempo que aprovechan la escala y la elasticidad de la nube de AWS.

## Contenedores de software

Los **contenedores de software** son una forma de empaquetar aplicaciones junto con sus dependencias para que se ejecuten de manera **consistente en cualquier entorno**.  

✅ **Comparación con Máquinas Virtuales (VMs)**

| Característica  | Contenedores  | Máquinas Virtuales (VMs) |
|---------------|-------------|------------------|
| Arranque rápido | ✅ Milisegundos | ❌ Minutos |
| Uso de recursos | ✅ Ligero | ❌ Pesado |
| Portabilidad  | ✅ Alta | ❌ Depende del hipervisor |
| Escalabilidad | ✅ Fácil | ❌ Más difícil |

Los contenedores usan el **kernel del sistema operativo** y comparten recursos, lo que los hace más eficientes que las VMs.

### **🔹 Tecnologías de Contenedores**  
Las principales tecnologías de contenedores incluyen:  

1️⃣ **Docker** – Plataforma para construir, ejecutar y gestionar contenedores.  
2️⃣ **Kubernetes (K8s)** – Orquestador de contenedores que permite escalado y administración automatizada.  
3️⃣ **Podman** – Alternativa sin daemon a Docker.

### **🔹 Contenedores en AWS**
AWS ofrece varios servicios para manejar contenedores:  

### **1. Amazon ECS (Elastic Container Service)**

✅ Orquestación de contenedores sin necesidad de gestionar servidores.  
✅ Soporta Docker.  
✅ Se integra con **AWS Fargate** para ejecutar contenedores sin administrar infraestructura.  

### **2. AWS Fargate**  

✅ Ejecución de contenedores **sin servidores**.  
✅ AWS gestiona la infraestructura automáticamente.  
✅ Paga solo por los recursos usados.  

### **3. Amazon EKS (Elastic Kubernetes Service)**  

✅ Servicio Kubernetes totalmente gestionado en AWS.  
✅ Ideal para arquitecturas de **microservicios**.  
✅ Integración con **EC2, Fargate y AWS Lambda**.  

### **4. Amazon Lightsail Containers**  

✅ Servicio simple para desplegar contenedores.  
✅ Ideal para pequeñas aplicaciones y entornos de prueba.  

### **🔹 Ejemplo: Ejecutar un Contenedor en AWS ECS**

1️⃣ **Crear una imagen Docker** y subirla a Amazon Elastic Container Registry (ECR):  
   ```bash
   docker build -t mi-app .
   docker tag mi-app:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/mi-app
   docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/mi-app
   ```
2️⃣ **Crear un cluster en ECS**.  
3️⃣ **Definir una tarea ECS** y usar la imagen de ECR.  
4️⃣ **Ejecutar el servicio ECS** con balanceo de carga.

### **🔹 Casos de Uso de Contenedores**

✅ Aplicaciones escalables en la nube  
✅ Microservicios  
✅ CI/CD con Jenkins y GitHub Actions  
✅ Big Data y Machine Learning  
✅ Migración de aplicaciones monolíticas a la nube

### **🚀 Guía Práctica: Desplegar un Contenedor en AWS ECS con Fargate**  

En esta guía, vamos a:  
✅ Crear una imagen Docker y subirla a AWS Elastic Container Registry (ECR).  
✅ Configurar un **Cluster ECS** y ejecutar el contenedor con **AWS Fargate**.  
✅ Acceder a la aplicación en un navegador.

### **🔹 1. Configurar AWS CLI y Docker**
Antes de comenzar, asegúrate de que tienes instalados:  
📌 **AWS CLI** – Para interactuar con AWS desde la terminal.  
📌 **Docker** – Para construir y ejecutar contenedores localmente.  

### **🔹 Configurar Credenciales AWS**

Ejecuta en la terminal:  
```bash
aws configure
```
Ingresa:  
- **Access Key ID**  
- **Secret Access Key**  
- **Región** (Ej: `us-east-1`)  
- **Formato** (Deja en blanco o escribe `json`)

### **🔹 2. Crear y Subir una Imagen Docker a ECR**

### **Paso 1: Crear un Repositorio en AWS ECR**

Ejecuta:  
```bash
aws ecr create-repository --repository-name mi-app
```
Obtendrás un repositorio en **Amazon Elastic Container Registry (ECR)**.  

### **Paso 2: Construir la Imagen Docker**

Si no tienes una imagen, crea un archivo `Dockerfile` con:  
```dockerfile
FROM nginx:latest
COPY index.html /usr/share/nginx/html/index.html
EXPOSE 80
```
Luego, **construye la imagen**:  
```bash
docker build -t mi-app .
```

### **Paso 3: Autenticarse en ECR**

Ejecuta:  
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
```
Reemplaza `123456789012` con tu **ID de cuenta AWS**.  

### **Paso 4: Etiquetar y Subir la Imagen a ECR**

```bash
docker tag mi-app:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/mi-app
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/mi-app
```
¡Listo! Tu imagen ya está en AWS ECR. 🎉  

### **🔹 3. Crear un Cluster en Amazon ECS**

1️⃣ Ve a **AWS ECS** y haz clic en **Clústeres → Crear Clúster**.  
2️⃣ Elige **"AWS Fargate (sin servidor)"** y nómbralo **"mi-cluster"**.  
3️⃣ **Crea el clúster**.  

### **🔹 4. Definir una Tarea ECS**

1️⃣ En **ECS → Definiciones de Tarea**, haz clic en **Crear nueva definición de tarea**.  
2️⃣ Tipo de lanzamiento: **Fargate**.  
3️⃣ Contenedor:  
   - **Nombre:** `mi-contenedor`  
   - **Imagen:** `123456789012.dkr.ecr.us-east-1.amazonaws.com/mi-app:latest`  
   - **Puerto:** `80`  
4️⃣ Define la **Memoria y CPU** (Ej: 0.5 vCPU, 1GB RAM).  
5️⃣ Guarda y **crea la tarea**.

### **🔹 5. Ejecutar la Tarea en ECS**

1️⃣ Ve a **ECS → Servicios → Crear servicio**.  
2️⃣ **Tipo de Servicio:** `Fargate`.  
3️⃣ **Clúster:** `mi-cluster`.  
4️⃣ **Definición de tarea:** `mi-contenedor`.  
5️⃣ **Número de tareas:** `1`.  
6️⃣ **Configuración de red:**  
   - **VPC:** Elige una disponible.  
   - **Subredes:** Selecciona una pública.  
   - **Habilita Auto-Asignación de IP Pública**.  
7️⃣ **Crea el servicio**.  

### **🔹 6. Acceder a la Aplicación**

1️⃣ Ve a **ECS → Tareas**, selecciona la tarea en ejecución.  
2️⃣ Copia la **IP Pública** de la instancia.  
3️⃣ Abre un navegador y accede a:  
   ```
   http://IP_PUBLICA
   ```
🎉 ¡Tu aplicación en contenedores ya está corriendo en AWS ECS con Fargate! 🚀🔥

### **🔹 ¿Qué Sigue?**

🔹 Configurar balanceadores de carga con **AWS ALB**.  
🔹 Automatizar despliegues con **GitHub Actions + ECS**.  
🔹 Monitorear contenedores con **AWS CloudWatch**.

### Resumen

El propósito de un contenedor es **crear un paquete de tu programa y todas sus librerías y dependencias con las versiones específicas con las que has trabajado**, para producir una imagen que pueda ser ejecutada en cualquier máquina.

Un problema común del desarrollo de software es utilizar distintas versiones de diferentes librerías/lenguajes de programación/programas. **Docker nos permite crear contenedores** para resolver este problema.

### Amazon ECS

[Amazon ECS](https://aws.amazon.com/es/ecs/ "Amazon ECS") es un servicio de **contenedores**, donde puedes implementar tus imágenes en contenedores en AWS. Cuando corras tus contenedores en AWS, **no notarás diferencia entre tu máquina local y el entorno de AWS**.

**Lecturas recomendadas**

[Curso de Docker [Empieza Gratis] - Platzi](https://platzi.com/cursos/docker/)

[Curso de Kubernetes [Empieza Gratis] - Platzi](https://platzi.com/cursos/k8s/)

## AWS Lambda

**AWS Lambda** es un servicio de computación sin servidores que te permite ejecutar código sin necesidad de administrar servidores. Solo subes tu código y AWS se encarga de la ejecución, escalado y administración.

### **🔹 Características Clave**

✅ **Sin Servidores** – AWS gestiona la infraestructura automáticamente.  
✅ **Pago por Uso** – Solo pagas por el tiempo que tu función esté en ejecución.  
✅ **Ejecución en Respuesta a Eventos** – Se activa con eventos de otros servicios AWS.  
✅ **Escalado Automático** – Se ajusta a la demanda sin intervención manual.  
✅ **Compatible con Múltiples Lenguajes** – Soporta Python, Node.js, Java, Go, Ruby, y más.

### **🔹 Casos de Uso**  
✔️ **Procesamiento de Datos en Tiempo Real** (logs, métricas, IoT).  
✔️ **Automatización de Tareas** (backup, limpieza de datos).  
✔️ **Desarrollo de APIs sin Servidor** con API Gateway + Lambda.  
✔️ **Integración con Otros Servicios AWS** (S3, DynamoDB, SNS, SQS, etc.).

### **🚀 Creando una Función AWS Lambda en la Consola**  

### **1️⃣ Acceder a AWS Lambda**

1. Ve a la [consola de AWS Lambda](https://aws.amazon.com/lambda/).  
2. Haz clic en **Crear función**.  

### **2️⃣ Configurar la Función**

- **Nombre**: `mi-lambda`  
- **Tiempo de ejecución**: `Python 3.9` (puedes elegir otro lenguaje).  
- **Permisos**: Usa los permisos por defecto o crea un rol IAM personalizado.  
- Haz clic en **Crear función**.  

### **3️⃣ Escribir el Código**

AWS proporciona un editor en línea. Agrega el siguiente código en **Python**:  
```python
import json

def lambda_handler(event, context):
    return {
        'statusCode': 200,
        'body': json.dumps('¡Hola desde AWS Lambda!')
    }
```
Guarda los cambios y haz clic en **Implementar**.  

### **4️⃣ Probar la Función**

1. Haz clic en **Probar**.  
2. Crea un evento de prueba con datos de ejemplo.  
3. Ejecuta la función y verifica el resultado en la consola.

### **🔹 Activar Lambda con un Evento de S3**

Si quieres que Lambda se ejecute cuando se suba un archivo a S3:  
1. **Ve a Amazon S3** y selecciona un bucket.  
2. **Configuración → Eventos → Crear evento**.  
3. Selecciona **PUT (carga de archivos)** como evento.  
4. Especifica el nombre de tu función Lambda.  
5. Guarda y prueba subiendo un archivo a S3.

### **🔹 Desplegar Lambda con AWS CLI**

Si prefieres crear y subir la función desde la terminal:  
1️⃣ Guarda tu código en `lambda_function.py`.  
2️⃣ Empaqueta el archivo en un ZIP:  
   ```bash
   zip mi-lambda.zip lambda_function.py
   ```
3️⃣ Crea la función en AWS Lambda:  
   ```bash
   aws lambda create-function \
     --function-name mi-lambda \
     --runtime python3.9 \
     --role arn:aws:iam::123456789012:role/lambda-role \
     --handler lambda_function.lambda_handler \
     --zip-file fileb://mi-lambda.zip
   ```
   Reemplaza `123456789012` con tu ID de cuenta y `lambda-role` con un rol IAM válido.

### **🔹 Integración con API Gateway**

Para exponer tu función Lambda como una API REST:  
1. **Ve a API Gateway → Crear API**.  
2. Selecciona **API REST** y crea un nuevo endpoint.  
3. Configura una **Integración con AWS Lambda**.  
4. Implementa y obtén la URL pública de tu API.

### **🎯 ¿Qué Sigue?**

🔹 Conectar Lambda con **DynamoDB, SQS o SNS**.  
🔹 Usar **Lambda Layers** para reutilizar código y dependencias.  
🔹 **Optimizar costos** ajustando memoria y tiempo de ejecución.

### Resumen

[AWS Lambda](https://aws.amazon.com/es/lambda/faqs/ "AWS Lambda") es un servicio serverless que nos permite **ejecutar código en respuesta a eventos, sin preocuparnos por servidores o infraestructura**. Estos eventos pueden ser temporizadores, visitas a alguna sección de nuestra aplicación, solicitudes HTTP, entre [otros](https://docs.aws.amazon.com/es_es/lambda/latest/dg/lambda-services.html#intro-core-components-event-sources "otros").

Entre sus casos de uso encontramos el (pre)procesamiento de datos a escala, y la ejecución de backends web, móviles y de [IoT](https://aws.amazon.com/es/iot/ "IoT") interactivos. **Lambda** se puede combinar con otros servicios de AWS para crear experiencias en línea seguras, estables y escalables.

### ¿Cómo se factura Lambda?

Lambda se factura por milisegundos, y el precio depende del uso de RAM. Por ejemplo, 128MB RAM x 30 millones de eventos por mes resultan en un costo de $11.63 al mes.

## Almacenamiento de datos en AWS 

AWS ofrece múltiples servicios de almacenamiento diseñados para diferentes necesidades. A continuación, te explico las principales opciones y cuándo usarlas.  

### **🔹 1. Amazon S3 (Simple Storage Service) – Almacenamiento de Objetos**  
📌 **¿Qué es?**  
Amazon S3 es un almacenamiento escalable para **archivos y datos no estructurados**.  

✅ **Características**  
✔ Almacenamiento **ilimitado** y pago por uso.  
✔ Alta **disponibilidad y durabilidad** (99.999999999% de durabilidad).  
✔ Permite almacenamiento en diferentes clases:  
   - **S3 Standard** (acceso frecuente).  
   - **S3 Intelligent-Tiering** (automático según uso).  
   - **S3 Glacier** (archivado a largo plazo).  
✔ Seguridad con **cifrado y control de acceso IAM**.  
✔ Se integra con **Lambda, CloudFront y DynamoDB**.  

🛠 **Casos de Uso**  
✅ Almacenamiento de imágenes, videos, backups.  
✅ Hosting de sitios web estáticos.  
✅ Integración con Big Data y ML.  

👨‍💻 **Ejemplo: Subir un archivo con AWS CLI**  
```bash
aws s3 cp mi-archivo.txt s3://mi-bucket/
```

### **🔹 2. Amazon EBS (Elastic Block Store) – Almacenamiento de Bloques**  
📌 **¿Qué es?**  
EBS proporciona almacenamiento en **bloques** para instancias **EC2**.  

✅ **Características**  
✔ Diseñado para **bases de datos y aplicaciones** de alto rendimiento.  
✔ Se comporta como un **disco duro** (SSD/HDD).  
✔ Persistente incluso si la instancia EC2 se detiene.  
✔ Soporta **Snapshots** para respaldo y restauración.  

🛠 **Casos de Uso**  
✅ Almacenamiento para servidores web y bases de datos en EC2.  
✅ Aplicaciones que requieren **baja latencia y alta IOPS**.  

👨‍💻 **Ejemplo: Crear un volumen EBS**  
```bash
aws ec2 create-volume --size 10 --region us-east-1 --availability-zone us-east-1a --volume-type gp2
```

### **🔹 3. Amazon EFS (Elastic File System) – Almacenamiento de Archivos**  
📌 **¿Qué es?**  
EFS es un sistema de archivos **compartido y escalable** para instancias **EC2 y contenedores**.  

✅ **Características**  
✔ Sistema de archivos basado en **NFS**.  
✔ **Escalado automático** sin necesidad de gestionar capacidad.  
✔ Alta disponibilidad y rendimiento.  

🛠 **Casos de Uso**  
✅ Servidores web con contenido compartido.  
✅ Análisis de datos en tiempo real con múltiples instancias.  

👨‍💻 **Ejemplo: Montar un sistema de archivos EFS**  
```bash
sudo mount -t nfs4 fs-12345678.efs.us-east-1.amazonaws.com:/ efs
```

### **🔹 4. Amazon RDS (Relational Database Service) – Bases de Datos Relacionales**  
📌 **¿Qué es?**  
Amazon RDS permite ejecutar bases de datos **gestionadas** como **MySQL, PostgreSQL, SQL Server y MariaDB**.  

✅ **Características**  
✔ No necesitas administrar hardware ni backups.  
✔ Escalado automático y alta disponibilidad con **Multi-AZ**.  
✔ Soporte para **Read Replicas** para mejorar rendimiento.  

🛠 **Casos de Uso**  
✅ Aplicaciones que requieren **SQL** y consultas estructuradas.  
✅ Sitios web y plataformas transaccionales.  

👨‍💻 **Ejemplo: Crear una base de datos RDS MySQL**  
```bash
aws rds create-db-instance --db-instance-identifier mi-bd --engine mysql --master-username admin --master-user-password password --allocated-storage 20 --db-instance-class db.t2.micro
```

### **🔹 5. Amazon DynamoDB – Base de Datos NoSQL**  
📌 **¿Qué es?**  
DynamoDB es una base de datos **NoSQL** gestionada con **alta escalabilidad**.  

✅ **Características**  
✔ Soporta **millones de consultas por segundo**.  
✔ Modelo de datos **clave-valor** y **documentos JSON**.  
✔ Se integra con **Lambda, API Gateway y ML**.  

🛠 **Casos de Uso**  
✅ Aplicaciones serverless y de IoT.  
✅ Sistemas de recomendación y catálogos de productos.  

👨‍💻 **Ejemplo: Crear una tabla en DynamoDB**  
```bash
aws dynamodb create-table --table-name MiTabla --attribute-definitions AttributeName=ID,AttributeType=S --key-schema AttributeName=ID,KeyType=HASH --billing-mode PAY_PER_REQUEST
```

### **🔹 Comparación Rápida**
| Servicio  | Tipo de Almacenamiento | Uso Principal |
|-----------|----------------|--------------|
| **S3** | Objetos | Archivos, backups, datos no estructurados |
| **EBS** | Bloques | Discos virtuales para EC2 |
| **EFS** | Archivos | Sistemas de archivos compartidos |
| **RDS** | Relacional | Bases de datos SQL |
| **DynamoDB** | NoSQL | Aplicaciones escalables y rápidas |

### **🎯 ¿Qué Sigue?**
🔹 **Configurar backups y snapshots en AWS**.  
🔹 **Conectar almacenamiento con Machine Learning en AWS**.  
🔹 **Optimizar costos con S3 Intelligent-Tiering y EFS Infrequent Access**.

### Resumen

El almacenamiento de datos en la nube consiste en **subir tus datos a dicha red de servidores, donde se te proporcionan herramientas para que puedas acceder a ellos de diferentes maneras**.

### Tipos de almacenamiento y sus servicios

Podemos utilizar distintos tipos almacenamiento datos, y para estos hay servicios de AWS. Los tipos de almacenamiento son:

- **Basado en archivos**: el más conocido por todos. Archivos organizados por carpetas y subcarpetas (sistema de ficheros). En esta categoría encontramos a [Amazon Elastic File System (EFS)](https://aws.amazon.com/es/efs/ "Amazon Elastic File System (EFS)") y [Amazon FSx for Windows File Server](https://aws.amazon.com/es/fsx/windows/ "Amazon FSx for Windows File Server").
- **Bloque**: los archivos se almacenan en volúmenes por fragmentos de datos de igual tamaño, sin procesar. Este tipo de almacenamiento es utilizado como disco duro de nuestros servidores o máquinas virtuales. En esta categoría está [Amazon Elastic Block Store (EBS)](https://aws.amazon.com/es/ebs/ "Amazon Elastic Block Store (EBS)").
- **Objetos**: la información almacenada se almacena como objetos, de manera que cada objeto recibe un identificador único y se almacena en un modelo de memoria plana. Un ejemplo de esto es [Amazon Simple Storage Service (S3)](https://aws.amazon.com/es/s3/ "Amazon Simple Storage Service (S3)").

### Respaldo de datos

**Amazon Backup administra y automatiza de forma centralizada** las copias de seguridad en los servicios de AWS.

### Servicios de transferencia de datos

¿Qué pasa si necesitamos transferir datos de nuestros servidores hacia AWS (o viceversa)? AWS ofrece distintos servicios para la transferencia de datos.

- **AWS Storage Gateway**: un conjunto de servicios de almacenamiento en la [nube híbrida](https://platzi.com/clases/2200-introduccion-azure/38231-tipos-de-nube-publica-privada-e-hibrida/ "nube híbrida") que brinda acceso en las instalaciones al almacenamiento en la nube.
- **AWS DataSync**: acelera el traslado de datos desde y hacia AWS hasta diez veces más rápido de lo normal.
- **AWS Transfer Family**: escala de forma segura tus transferencias recurrentes de archivos de **Amazon S3** y **Amazon EFS** con los protocolos [FTP](https://www.arsys.es/soporte/hosting-web/ftp/que-es-ftp#:~:text=FTP%20es%20un%20protocolo%20que,directorios%2C%20borrar%20ficheros%2C%20etc. "FTP"), [SFTP](https://es.wikipedia.org/wiki/SSH_File_Transfer_Protocol "SFTP") y [FTPS](https://es.wikipedia.org/wiki/FTPS "FTPS").

**Conclusión**

Exploramos de manera breve los distintos servicios de almacenamiento de AWS, así como los tipos de almacenamiento que podemos utilizar.

## S3 y S3 Glacier

Amazon S3 y S3 Glacier son servicios de almacenamiento de objetos en AWS, pero tienen diferentes propósitos.  

- **Amazon S3** → Para datos de acceso frecuente o moderado.  
- **Amazon S3 Glacier** → Para archivado a largo plazo con menor costo.  

### **🔹 Amazon S3 (Simple Storage Service)**

📌 **¿Qué es?**  
Amazon S3 es un servicio de **almacenamiento de objetos** que permite guardar datos en la nube de forma escalable, segura y de alta disponibilidad.  

✅ **Características Clave**  
✔ **Almacenamiento duradero (99.999999999%)** para datos no estructurados.  
✔ **Pago por uso**, sin costos iniciales.  
✔ **Integración con Lambda, CloudFront, DynamoDB y más**.  
✔ **Versionado y control de acceso** con políticas IAM.  
✔ **Cifrado** en tránsito y en reposo.  

🛠 **Casos de Uso**  
✅ Hosting de sitios web estáticos.  
✅ Almacenamiento de imágenes, videos y backups.  
✅ Procesamiento de Big Data y Machine Learning.  

👨‍💻 **Ejemplo: Subir un archivo a S3 con AWS CLI**  
```bash
aws s3 cp mi-archivo.txt s3://mi-bucket/
```

### **🔹 Amazon S3 Glacier** (Archivado de Datos de Bajo Costo)

📌 **¿Qué es?**  
S3 Glacier es un servicio de almacenamiento **de bajo costo** para datos que no necesitan acceso inmediato.  

✅ **Características Clave**  
✔ **Hasta 80% más barato que S3 Standard**.  
✔ Diseñado para **archivado a largo plazo**.  
✔ **Recuperación de datos en diferentes velocidades** (Expedited, Standard, Bulk).  
✔ **Alta durabilidad (99.999999999%)**.  

🛠 **Casos de Uso**  
✅ Backups históricos y recuperación ante desastres.  
✅ Archivos de registros y auditorías.  
✅ Cumplimiento normativo y retención de datos.  

👨‍💻 **Ejemplo: Mover un archivo de S3 a Glacier**  
```bash
aws s3 mv s3://mi-bucket/mi-archivo.txt s3://mi-bucket/mi-archivo.txt --storage-class GLACIER
```

### **🔹 Comparación Rápida**  

| **Característica** | **Amazon S3** | **S3 Glacier** |
|--------------------|--------------|---------------|
| **Costo** | Más alto | Mucho más barato |
| **Velocidad de Acceso** | Inmediato | Desde minutos hasta horas |
| **Casos de Uso** | Archivos en uso frecuente | Archivado a largo plazo |
| **Durabilidad** | 99.999999999% | 99.999999999% |
| **Recuperación de Datos** | Instantánea | **Expedited (minutos), Standard (3-5h), Bulk (12-48h)** |

### **🎯 ¿Qué Elegir?**
- **Si necesitas acceso frecuente → S3 Standard.**  
- **Si los datos se usan poco → S3 Standard-IA o Intelligent-Tiering.**  
- **Si es almacenamiento de archivos a largo plazo → S3 Glacier.**

### Resumen

[Amazon S3](https://aws.amazon.com/es/s3/ "Amazon S3") es un servicio de almacenamiento de objetos, líder en la industria. Otorga una **garantía de no pérdida de datos del 99.999999999%** (11 9’s).

### Clases de almacenamiento en S3

Amazon nos ofrece [distintas clase de almacenamiento](https://aws.amazon.com/es/s3/storage-classes/?nc=sn&loc=3 "distintas clase de almacenamiento") S3 en función de nuestras necesidades de acceso y disponibilidad de los datos.

- **S3 Standard**: almacenamiento de objetos de alta durabilidad, disponibilidad y rendimiento para datos a los que se obtiene acceso con frecuencia.
- **S3 Standard-IA**: se utiliza con datos a los que se accede con menos frecuencia, pero que requieren un acceso rápido cuando es necesario.
- **S3 Zone-IA**: similar a Standard-IA, pero con un menor costo de almacenamiento ya que solo usa una zona de disponibilidad. Distinto de las demás clases de almacenamiento de S3, que almacenan datos en un mínimo de tres zonas de disponibilidad (AZ).
- **S3 Glacier**: ofrece el almacenamiento de menor costo para los datos de larga duración y acceso poco frecuente. Tiene un costo de $1 por TB al mes. Tiene tres opciones para la recuperación de datos (estándar, masiva y acelerada).
- **S3 Glacier Deep Archive**: la clase de almacenamiento más económica de Amazon S3. Admite la retención a largo plazo y la conservación digital de datos a los que se accede una o dos veces al año.
- **S3 Intelligent-Tiering**: un tipo de almacenamiento que intenta ahorrar costos moviendo archivos entre los distintos tipos de almacenamiento S3, basado en los patrones de uso de los archivos.

**Conclusión**

Tenemos variedad de opciones para escoger la clase de almacenamiento S3 en función de nuestras necesidades. Si necesitamos un almacenamiento altamente disponible y duradero, **S3 Standard** es la mejor opción, mientras que si necesitamos un almacenamiento a largo plazo y de acceso infrecuente, podemos usar **S3 Glacier**. Escoge la mejor opción según tu caso de uso.

**Lecturas recomendadas**

[Tipos de almacenamiento en la nube | Amazon S3](https://aws.amazon.com/es/s3/storage-classes/)