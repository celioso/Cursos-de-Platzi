# Curso de Introducción a la Nube

## Razones para usar la nube

El uso de la nube ha transformado la forma en que las empresas y desarrolladores administran infraestructura y aplicaciones. Aquí te dejo las principales razones para utilizar la nube:

### **1️⃣ Reducción de Costos 💰**  
✅ No necesitas invertir en servidores físicos ni en mantenimiento.  
✅ Pagas solo por lo que usas (modelo **pay-as-you-go**).  
✅ Reduces costos de electricidad, refrigeración y espacio físico.

### **2️⃣ Escalabilidad y Elasticidad 📈**  
✅ Puedes **aumentar o disminuir recursos** automáticamente según la demanda.  
✅ No necesitas predecir el tráfico con anticipación.  
✅ Ideal para aplicaciones con cargas de trabajo variables (e-commerce, streaming, etc.).

### **3️⃣ Alta Disponibilidad y Recuperación ante Desastres 🌎**  
✅ Los proveedores en la nube tienen centros de datos en diferentes regiones.  
✅ Puedes configurar replicación y copias de seguridad automatizadas.  
✅ Tiempos de inactividad reducidos en caso de fallos.

### **4️⃣ Seguridad 🔒**  
✅ Cifrado de datos en tránsito y en reposo.  
✅ Control de acceso basado en identidad (IAM).  
✅ Monitoreo y auditoría constante de los sistemas.

### **5️⃣ Agilidad e Innovación 🚀**  
✅ Puedes crear entornos de prueba en minutos.  
✅ Acceso a tecnologías avanzadas como **Machine Learning, IoT y Big Data**.  
✅ Implementación rápida de aplicaciones con servicios gestionados.

### **6️⃣ Accesibilidad y Trabajo Remoto 🌍**  
✅ Accede a tus aplicaciones y datos desde cualquier parte del mundo.  
✅ Facilita el trabajo colaborativo en tiempo real.  
✅ Compatible con múltiples dispositivos (PC, móvil, tablet).

### **7️⃣ Automatización y Gestión Inteligente 🤖**  
✅ Uso de herramientas como **AWS Lambda, CloudFormation y Terraform** para automatizar tareas.  
✅ Monitoreo en tiempo real con **CloudWatch** y **Azure Monitor**.  
✅ Integración con Inteligencia Artificial para optimización de recursos. 

### **📌 Conclusión**  
La nube permite ahorrar costos, escalar aplicaciones de manera eficiente y mejorar la seguridad. Es ideal para empresas de todos los tamaños, desde startups hasta grandes corporaciones.

## Servidores y almacenamiento

Los **servidores** y el **almacenamiento** son componentes esenciales en la nube, proporcionando potencia de cómputo y capacidad para almacenar datos de manera flexible y segura.

### **1️⃣ Servidores en la Nube 🖥️☁️**  

Los servidores en la nube permiten ejecutar aplicaciones sin necesidad de comprar y mantener hardware físico.  

### **Tipos de Servidores en la Nube**  

🔹 **Máquinas Virtuales (VMs)** – Servidores basados en software que emulan hardware físico.  
🔹 **Servidores Bare Metal** – Servidores físicos dedicados sin virtualización.  
🔹 **Servidores Sin Servidor (Serverless)** – Ejecutan funciones bajo demanda sin gestionar infraestructura.  

### **Ejemplos de Servicios de Servidores en la Nube**  

✅ **AWS EC2** – Servidores virtuales escalables.  
✅ **Google Compute Engine (GCE)** – Máquinas virtuales en Google Cloud.  
✅ **Azure Virtual Machines** – VMs en Microsoft Azure.  
✅ **AWS Lambda / Azure Functions** – Cómputo sin servidor (serverless).

### **2️⃣ Almacenamiento en la Nube 💾**  

El almacenamiento en la nube permite guardar y acceder a datos de forma remota, con alta disponibilidad y seguridad.  

### **Tipos de Almacenamiento en la Nube**  

📂 **Almacenamiento de Objetos** – Ideal para archivos, imágenes, videos, etc.  
📂 **Almacenamiento de Bloques** – Similar a un disco duro, usado en servidores y bases de datos.  
📂 **Almacenamiento de Archivos** – Compartición de archivos entre servidores o usuarios.  

### **Ejemplos de Servicios de Almacenamiento en la Nube**  

✅ **Amazon S3** – Almacenamiento de objetos escalable.  
✅ **Google Cloud Storage** – Alternativa en Google Cloud.  
✅ **Azure Blob Storage** – Almacenamiento de objetos en Microsoft Azure.  
✅ **Amazon EBS (Elastic Block Store)** – Almacenamiento de bloques para EC2.  
✅ **Amazon EFS (Elastic File System)** – Sistema de archivos elástico.  
✅ **AWS Glacier** – Almacenamiento de archivos a largo plazo.  

### **📌 Conclusión**  
Los servidores y el almacenamiento en la nube permiten mayor flexibilidad, escalabilidad y reducción de costos en comparación con la infraestructura tradicional.

1. Servidores: PC que forma parte de una red y provee servicios a los

- Sistema Operativo: Windows, Linux, MacOS
- Aplicaciones: Apache, Nginx, IIS, PlatziWallet
- Ubicación: Puede estar onpremises (Instalaciones de la empresa dueña) o nube (Espacio provisto por un proveedor puede ser AWS, Google, etc)
- Servicio: Provee servicios a cantidad grande de usuarios
2. Almacenamiento: Repositorio de almacenamiento de información para que este disponible cada que sea necesario

- Almacenamiento por objetos: Divide los datos en partes distribuidas en el hardware. Cada unidad se llama objeto. (archivos, imágenes, archivos estáticos. No es para instalación de aplicaciones)
- Almacenamiento por Archivos: Los datos son guardados como una pieza de información dentro de una carpeta. (Almacenamiento que debe ser compartido por múltiples servidores)
- Almacenamiento por Bloque: Divide la información en bloques. Tiene un identificador único. Permite que se coloquen los datos más pequeños donde sea más conveniente. (Disco duro del servidor virtual en la nube)

## Bases de datos

Las bases de datos en la nube permiten almacenar, gestionar y acceder a los datos de manera flexible, escalable y segura sin preocuparse por la infraestructura física.

### **1️⃣ Tipos de Bases de Datos en la Nube 📊**  

Existen dos categorías principales:  

🔹 **Bases de Datos Relacionales (SQL)** – Utilizan estructuras de tablas y soportan consultas en **SQL**. Son ideales para transacciones estructuradas y relaciones definidas.  
🔹 **Bases de Datos NoSQL** – Diseñadas para datos no estructurados o semiestructurados, como documentos JSON, datos en gráficos o almacenamiento en clave-valor.

### **2️⃣ Servicios de Bases de Datos en AWS ☁️**  

### **🔹 Bases de Datos Relacionales (SQL) en AWS**  

📌 **Amazon RDS (Relational Database Service)** – Servicio administrado que soporta:  
✅ **MySQL**  
✅ **PostgreSQL**  
✅ **MariaDB**  
✅ **SQL Server**  
✅ **Oracle**  
✅ **Amazon Aurora** – Base de datos optimizada y escalable compatible con MySQL y PostgreSQL.  

### **🔹 Bases de Datos NoSQL en AWS**  

📌 **Amazon DynamoDB** – Base de datos NoSQL de clave-valor y documentos, altamente escalable.  
📌 **Amazon DocumentDB** – Base de datos NoSQL compatible con MongoDB.  
📌 **Amazon ElastiCache** – Bases de datos en memoria (Redis y Memcached) para acelerar aplicaciones.  
📌 **Amazon Neptune** – Base de datos para grafos, ideal para redes sociales y análisis de relaciones.  
📌 **Amazon Keyspaces** – Base de datos compatible con Apache Cassandra.

### **3️⃣ Beneficios de Usar Bases de Datos en la Nube 🚀**  

✅ **Alta Disponibilidad** – Replicación automática y recuperación ante fallos.  
✅ **Escalabilidad** – Ajusta capacidad según demanda sin afectar el rendimiento.  
✅ **Menos Administración** – AWS gestiona copias de seguridad, parches y actualizaciones.  
✅ **Seguridad** – Cifrado de datos y control de acceso detallado.  
✅ **Modelo de Pago por Uso** – Pagas solo por el almacenamiento y procesamiento utilizado.

### **📌 Conclusión**  
Las bases de datos en la nube ofrecen flexibilidad, escalabilidad y seguridad sin necesidad de administrar hardware.

## Microservicios, funciones y contenedores

Las arquitecturas modernas en la nube permiten desarrollar aplicaciones escalables y eficientes mediante **microservicios, funciones serverless y contenedores**.

### **1️⃣ Microservicios 🏗️**  

Los **microservicios** son un enfoque de desarrollo en el que una aplicación se divide en pequeños servicios independientes que se comunican entre sí mediante APIs.  

### **🔹 Características de los Microservicios**  

✅ **Desacoplamiento** – Cada servicio es independiente y puede desplegarse por separado.  
✅ **Escalabilidad** – Se pueden escalar individualmente según la demanda.  
✅ **Desarrollo Ágil** – Facilita la implementación continua y el desarrollo paralelo.  
✅ **Resiliencia** – Fallos en un servicio no afectan a toda la aplicación.  

### **🔹 Servicios en AWS para Microservicios**  

📌 **Amazon ECS (Elastic Container Service)** – Orquestación de contenedores.  
📌 **Amazon EKS (Elastic Kubernetes Service)** – Kubernetes gestionado para microservicios.  
📌 **AWS API Gateway** – Gestión de APIs para la comunicación entre microservicios.  
📌 **AWS Service Mesh (App Mesh)** – Permite el enrutamiento y monitoreo de microservicios.

### **2️⃣ Funciones Serverless ⚡**  

El modelo **serverless** permite ejecutar código sin administrar servidores. Ideal para tareas event-driven y procesamiento en segundo plano.  

### **🔹 Características del Serverless Computing**  

✅ **Sin gestión de infraestructura** – AWS maneja la ejecución del código.  
✅ **Escalabilidad automática** – Se ajusta a la demanda sin intervención manual.  
✅ **Pago por uso** – Se cobra solo por el tiempo de ejecución del código.  

### **🔹 Servicios Serverless en AWS**  

📌 **AWS Lambda** – Ejecuta código en respuesta a eventos sin servidores.  
📌 **Amazon EventBridge** – Integración de eventos entre aplicaciones.  
📌 **AWS Step Functions** – Orquestación de funciones serverless.

### **3️⃣ Contenedores 🐳**  

Los **contenedores** empaquetan aplicaciones con sus dependencias para garantizar que se ejecuten de la misma manera en cualquier entorno.  

### **🔹 Beneficios de los Contenedores**  

✅ **Portabilidad** – Funcionan en cualquier sistema con un runtime compatible.  
✅ **Eficiencia** – Uso optimizado de recursos en comparación con las máquinas virtuales.  
✅ **Rápido despliegue** – Permiten el escalado y despliegue ágil.  

### **🔹 Servicios de Contenedores en AWS**  

📌 **Amazon ECS** – Servicio de orquestación de contenedores basado en Docker.  
📌 **Amazon EKS** – Kubernetes gestionado en AWS.  
📌 **AWS Fargate** – Ejecuta contenedores sin administrar servidores.  
📌 **Amazon Elastic Container Registry (ECR)** – Almacenamiento de imágenes de contenedores.

### **📌 Conclusión**  

✅ **Microservicios** → Ideal para arquitecturas escalables y modulares.  
✅ **Funciones Serverless** → Perfecto para ejecutar tareas bajo demanda sin administrar infraestructura.  
✅ **Contenedores** → Facilitan el despliegue y portabilidad de aplicaciones.