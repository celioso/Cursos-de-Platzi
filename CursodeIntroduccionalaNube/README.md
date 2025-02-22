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

## ¿Qué es on-premises?

**On-Premises** (también conocido como **"en las instalaciones"**) se refiere a un modelo de infraestructura donde una empresa posee, opera y mantiene sus propios **servidores, almacenamiento y redes** dentro de sus instalaciones físicas, en lugar de utilizar servicios en la nube.

### **🔹 Características del On-Premises**  

✅ **Control Total** – La empresa gestiona toda la infraestructura y seguridad.  
✅ **Personalización** – Se pueden ajustar servidores y redes según las necesidades específicas.  
✅ **Costos Iniciales Altos** – Se requiere inversión en hardware, software y mantenimiento.  
✅ **Escalabilidad Limitada** – Ampliar la capacidad requiere adquirir más hardware.  
✅ **Responsabilidad Completa** – La empresa es responsable de la seguridad, parches y actualizaciones.

### **🔹 Diferencias Entre On-Premises y la Nube ☁️**  

| **Característica**     | **On-Premises** | **Cloud (Nube)** |
|------------------------|----------------|------------------|
| **Gestión**            | Empresa propia  | Proveedor en la nube |
| **Costos**            | Altos costos iniciales + mantenimiento | Pago por uso, sin inversión inicial |
| **Escalabilidad**     | Limitada, depende del hardware adquirido | Escalabilidad automática |
| **Seguridad**         | Control total, pero mayor responsabilidad | Seguridad gestionada por el proveedor |
| **Actualizaciones**   | Se deben hacer manualmente | Se actualiza automáticamente |

### **🔹 ¿Cuándo Usar On-Premises?**  

🔹 Empresas con **regulaciones estrictas** de datos (bancos, gobiernos, salud).  
🔹 Negocios que requieren **control total sobre la seguridad y la infraestructura**.  
🔹 Organizaciones con **infraestructura ya amortizada** y sin necesidad de escalabilidad.

### **📌 Conclusión**  

Si bien **On-Premises** ofrece control total, la **nube** permite mayor flexibilidad, reducción de costos y escalabilidad. Hoy en día, muchas empresas optan por **arquitecturas híbridas** que combinan ambas soluciones.

**Lecturas recomendadas**

[https://cloud.google.com/learn/what-is-cloud-computing?hl=es-419](https://cloud.google.com/learn/what-is-cloud-computing?hl=es-419)

[https://azure.microsoft.com/es-es/resources/cloud-computing-dictionary/what-is-the-cloud](https://azure.microsoft.com/es-es/resources/cloud-computing-dictionary/what-is-the-cloud)

[AWS | Informática en la nube. Ventajas y Beneficios](https://aws.amazon.com/es/what-is-cloud-computing/)

## ¿Qué es Cloud Computing o nube?

**Cloud Computing**, o **computación en la nube**, es un modelo que permite el acceso a recursos informáticos como **servidores, almacenamiento, bases de datos, redes y software** a través de Internet, en lugar de mantenerlos en una infraestructura física propia (**on-premises**).  

Con la nube, las empresas pueden **escalar** recursos según la demanda y pagar solo por lo que utilizan, lo que reduce costos y mejora la eficiencia operativa.

### **🔹 Características del Cloud Computing**  

✅ **Pago por Uso** – Solo pagas por lo que consumes, sin grandes inversiones iniciales.  
✅ **Escalabilidad** – Puedes aumentar o reducir recursos de manera flexible.  
✅ **Accesibilidad** – Se accede desde cualquier parte del mundo con conexión a Internet.  
✅ **Seguridad** – Protección avanzada con cifrado, firewalls y copias de seguridad automáticas.  
✅ **Automatización** – Gestión de actualizaciones, parches y mantenimiento sin intervención manual.

### **🔹 Modelos de Servicio en la Nube**  

| **Modelo** | **Descripción** | **Ejemplo** |
|------------|----------------|-------------|
| **IaaS** (Infraestructura como Servicio) | Alquila servidores, almacenamiento y redes sin gestionar hardware físico. | AWS EC2, Google Compute Engine, Azure Virtual Machines |
| **PaaS** (Plataforma como Servicio) | Proporciona entornos listos para el desarrollo sin gestionar servidores. | AWS Elastic Beanstalk, Google App Engine, Heroku |
| **SaaS** (Software como Servicio) | Aplicaciones listas para usar a través de Internet. | Google Drive, Dropbox, Gmail |

### **🔹 Tipos de Nube**  

🌍 **Nube Pública** – Gestionada por terceros (AWS, Google Cloud, Azure).  
🏢 **Nube Privada** – Infraestructura exclusiva de una empresa.  
🔗 **Nube Híbrida** – Combinación de nube pública y privada.  
🌐 **Nube Multicloud** – Uso de varios proveedores de nube.

### **🔹 Beneficios del Cloud Computing**  

🔹 **Reducción de costos** – Evita comprar y mantener servidores físicos.  
🔹 **Flexibilidad y escalabilidad** – Adapta los recursos a la demanda en segundos.  
🔹 **Alta disponibilidad** – Centros de datos globales garantizan continuidad.  
🔹 **Mayor seguridad** – Protección contra ataques y copias de seguridad automáticas.  
🔹 **Colaboración global** – Equipos pueden trabajar en cualquier lugar.

### **📌 Conclusión**  

El **Cloud Computing** ha revolucionado la forma en que las empresas gestionan su infraestructura tecnológica, permitiendo mayor eficiencia, seguridad y reducción de costos.

## ¿Por qué usar Cloud Computing o nube?

El **Cloud Computing** ha transformado la forma en que las empresas y usuarios acceden a la tecnología, ofreciendo **flexibilidad, escalabilidad y reducción de costos**. En lugar de depender de servidores físicos locales (**on-premises**), la nube permite acceder a recursos informáticos a través de Internet.  

A continuación, te explico las **principales razones para usar la nube**:

### **🔹 1. Reducción de Costos 💰**  
✅ **Menos inversión inicial** – No necesitas comprar hardware costoso.  
✅ **Pago por uso** – Solo pagas por los recursos que consumes.  
✅ **Menos costos de mantenimiento** – El proveedor de la nube gestiona la infraestructura.  

📌 *Ejemplo:* Empresas pueden alojar sus sitios web en AWS o Google Cloud sin comprar servidores propios.

### **🔹 2. Escalabilidad y Flexibilidad 📈**  
✅ **Escala automáticamente** según la demanda (más o menos recursos cuando sea necesario).  
✅ **Adapta los servicios** sin necesidad de grandes cambios en la infraestructura.  
✅ **Globalización rápida** – Implementa servidores en cualquier región del mundo.  

📌 *Ejemplo:* Netflix usa AWS para escalar sus servidores y atender millones de usuarios simultáneamente.

### **🔹 3. Alta Disponibilidad y Continuidad del Negocio 🔄**  
✅ **Centros de datos distribuidos** en todo el mundo garantizan disponibilidad.  
✅ **Resistencia a fallos** – Si un servidor falla, otro toma el control.  
✅ **Recuperación ante desastres** con copias de seguridad automáticas.  

📌 *Ejemplo:* Un banco puede seguir operando sin interrupciones gracias a la nube.

### **🔹 4. Seguridad Avanzada 🔒**  
✅ **Protección contra ataques cibernéticos** con firewalls, cifrado y autenticación multifactor.  
✅ **Cumplimiento de normativas** como GDPR, ISO 27001 y HIPAA.  
✅ **Monitoreo 24/7** y alertas en caso de amenazas.  

📌 *Ejemplo:* AWS proporciona seguridad de datos con cifrado y acceso restringido.

### **🔹 5. Acceso desde Cualquier Lugar 🌍**  
✅ Solo necesitas conexión a Internet para trabajar desde cualquier dispositivo.  
✅ Facilita el **trabajo remoto y la colaboración** en tiempo real.  
✅ No hay necesidad de instalar software localmente.  

📌 *Ejemplo:* Empresas como Google y Microsoft permiten trabajar en la nube con Google Drive y OneDrive.

### **🔹 6. Automatización y Agilidad 🚀**  
✅ **Actualizaciones automáticas** sin interrupciones.  
✅ **Integración con Inteligencia Artificial y Machine Learning**.  
✅ **Menos carga de trabajo para los equipos de TI**.  

📌 *Ejemplo:* Un e-commerce puede automatizar el procesamiento de pagos y gestión de inventario en la nube.

### **📌 Conclusión**  

🔹 **Cloud Computing** es ideal para empresas de todos los tamaños por su **costo reducido, escalabilidad, seguridad y accesibilidad**.  
🔹 Empresas como Amazon, Google, Netflix y startups **dependen de la nube** para operar con eficiencia.  
🔹 La nube permite a las empresas **innovar más rápido y mejorar la experiencia del usuario**.

## ¿Por qué una arquitectura en Cloud Computing o nube es diferente?

La arquitectura en la **nube** es diferente de la arquitectura **tradicional on-premises** porque está diseñada para aprovechar las características únicas de la computación en la nube: **escalabilidad, automatización, pago por uso y resiliencia**.  

A continuación, te explico las diferencias clave:

### **🔹 1. Escalabilidad y Elasticidad 📈**  
✅ **Automática y bajo demanda** – La nube ajusta los recursos según el tráfico.  
✅ **Horizontal y vertical** – Se pueden añadir más servidores (escalado horizontal) o mejorar los existentes (escalado vertical).  

📌 *Ejemplo:* Un e-commerce escala automáticamente en Black Friday sin interrupciones.

### **🔹 2. Pago por Uso y Optimización de Costos 💰**  
✅ **No requiere inversión en hardware** – Solo pagas por lo que consumes.  
✅ **Optimización dinámica** – Se pueden apagar recursos no utilizados.  

📌 *Ejemplo:* Un startup solo paga por los servidores mientras su aplicación está activa.

### **🔹 3. Alta Disponibilidad y Tolerancia a Fallos 🔄**  
✅ **Centros de datos distribuidos en diferentes regiones** garantizan continuidad.  
✅ **Balanceadores de carga** redirigen el tráfico si un servidor falla.  

📌 *Ejemplo:* Netflix usa AWS para asegurar que su servicio nunca se caiga.

### **🔹 4. Infraestructura como Código (IaC) ⚙️**  
✅ Se puede **automatizar y gestionar** toda la infraestructura con código.  
✅ Permite **despliegues rápidos y repetibles**.  

📌 *Ejemplo:* Con AWS CloudFormation, se puede crear toda una infraestructura con un solo comando.

### **🔹 5. Seguridad y Gobernanza 🔒**  
✅ **Control de acceso granular** – Se definen permisos con IAM (Identity and Access Management).  
✅ **Cifrado de datos en tránsito y en reposo**.  

📌 *Ejemplo:* Un banco en la nube usa AWS KMS para cifrar datos sensibles.

### **🔹 6. Desacoplamiento y Microservicios 🔧**  
✅ Se dividen las aplicaciones en **microservicios independientes**.  
✅ Se utilizan **APIs y colas de mensajes** para comunicación entre servicios.  

📌 *Ejemplo:* Uber usa microservicios en la nube para gestionar pagos, mapas y usuarios de forma separada.

### **📌 Conclusión**  

🔹 **La nube permite una arquitectura más eficiente, escalable y resiliente.**  
🔹 **Reduce costos operativos y facilita la automatización.**  
🔹 **Empresas como Netflix, Amazon y Google han optimizado sus sistemas gracias a la nube.**

## Infraestructura global, regiones y zonas

AWS cuenta con una **infraestructura global distribuida** para ofrecer **alta disponibilidad, baja latencia y escalabilidad**. Está diseñada para soportar cargas de trabajo críticas a nivel mundial.

### **🔹 1. Infraestructura Global de AWS 🌐**  
AWS opera en múltiples ubicaciones alrededor del mundo, organizadas en:  

✅ **Regiones** 🏢  
✅ **Zonas de Disponibilidad (AZs)** 📡  
✅ **Puntos de Presencia (PoPs) para CDN** 📍

### **🔹 2. ¿Qué es una Región en AWS? 📍**  
Una **Región de AWS** es una ubicación geográfica donde AWS tiene **múltiples centros de datos**.  

📌 **Características:**  
✔️ Cada **región es independiente** de las demás en términos de seguridad y cumplimiento.  
✔️ Está compuesta por **varias Zonas de Disponibilidad** para mayor redundancia.  
✔️ **Ejemplo de regiones:** `us-east-1` (Virginia), `eu-west-1` (Irlanda), `sa-east-1` (São Paulo).  

**🔎 ¿Cómo elegir una región?**  
✅ **Latencia baja** (cercanía a los usuarios).  
✅ **Requisitos de cumplimiento** (normativas locales).  
✅ **Costo** (diferentes precios por región).

### **🔹 3. ¿Qué es una Zona de Disponibilidad (AZ)? 🏢**  
Cada **Región** contiene **varias Zonas de Disponibilidad (AZs)**, que son centros de datos físicamente separados pero interconectados con redes de alta velocidad.  

📌 **Ejemplo:** La región `us-east-1` tiene **6 AZs** (`us-east-1a`, `us-east-1b`, etc.).  

**✅ Beneficios de las AZs:**  
✔️ **Alta disponibilidad** – Si una AZ falla, otra sigue operando.  
✔️ **Baja latencia** – Comunicación rápida entre AZs.  
✔️ **Balanceo de carga** – Distribución eficiente de tráfico.

### **🔹 4. Puntos de Presencia (PoPs) y AWS Edge Locations 🌎**  
AWS tiene **más de 450 Puntos de Presencia** (PoPs) en todo el mundo para acelerar la entrega de contenido a los usuarios.  

📌 **AWS CloudFront** usa estas ubicaciones para mejorar la velocidad de acceso a sitios web, videos y datos.  

✅ **Beneficios:**  
✔️ **Baja latencia** con servidores más cercanos a los usuarios.  
✔️ **Seguridad mejorada** con protección contra ataques DDoS.  
✔️ **CDN eficiente** para distribuir contenido globalmente.  

### **📌 Conclusión**  
🔹 AWS tiene una **infraestructura global robusta**, con **Regiones, Zonas de Disponibilidad y Puntos de Presencia**.  
🔹 Permite a empresas construir aplicaciones **escalables, seguras y altamente disponibles** en todo el mundo.  
🔹 **¿Listo para desplegar tu aplicación en la nube? 🚀**

**Lecturas recomendadas**

[https://pages.awscloud.com/rs/112-TZM-766/images/Enter_the_Purpose-Built-Database-Era.pdf](https://pages.awscloud.com/rs/112-TZM-766/images/Enter_the_Purpose-Built-Database-Era.pdf)

## Nube privada, pública, híbrida, multinube

El **Cloud Computing** ofrece diferentes modelos de implementación según las necesidades de cada empresa. Aquí exploramos los cuatro principales tipos de nube:

### **☁️ 1. Nube Pública**  
📌 **Definición:** Es una infraestructura de nube gestionada por un proveedor externo como AWS, Google Cloud o Azure, y los recursos (servidores, almacenamiento, bases de datos) se comparten entre múltiples clientes.  

✅ **Ventajas:**  
✔️ **Costo reducido** – No se necesita infraestructura propia.  
✔️ **Escalabilidad** – Se ajusta según la demanda.  
✔️ **Accesibilidad global** – Disponible en cualquier parte del mundo.  
✔️ **Mantenimiento gestionado** – El proveedor se encarga de actualizaciones y seguridad.  

🚀 **Ejemplo:** Usar **Amazon EC2**, **Google Compute Engine** o **Microsoft Azure Virtual Machines** para ejecutar aplicaciones en la nube sin administrar servidores físicos.

### **🏢 2. Nube Privada**  
📌 **Definición:** Es una infraestructura de nube dedicada a una sola organización. Puede estar ubicada en un centro de datos propio o administrada por un tercero.  

✅ **Ventajas:**  
✔️ **Mayor seguridad y control** – Datos y sistemas exclusivos de la empresa.  
✔️ **Cumplimiento normativo** – Ideal para industrias reguladas (banca, salud).  
✔️ **Personalización** – Se adapta a necesidades específicas.  

⚠️ **Desafíos:**  
❌ Costos elevados de mantenimiento.  
❌ Escalabilidad limitada en comparación con la nube pública.  

🚀 **Ejemplo:** Un banco con su propio centro de datos usa **VMware Cloud** o **OpenStack** para administrar sus servidores de manera privada.

### **🔄 3. Nube Híbrida**  
📌 **Definición:** Combina **nube privada y pública**, permitiendo mover cargas de trabajo entre ambas según las necesidades.  

✅ **Ventajas:**  
✔️ **Flexibilidad** – Datos sensibles en nube privada y cargas pesadas en nube pública.  
✔️ **Optimización de costos** – Se paga por recursos solo cuando se necesitan.  
✔️ **Mayor continuidad del negocio** – Alternativa en caso de fallos en una infraestructura.  

🚀 **Ejemplo:** Una empresa usa **AWS para desarrollo** y mantiene **datos confidenciales en servidores privados**.

### **🌍 4. Multinube**  
📌 **Definición:** Usa múltiples proveedores de nube (AWS, Azure, Google Cloud) para distribuir aplicaciones y servicios.  

✅ **Ventajas:**  
✔️ **Evita dependencia de un solo proveedor** (vendor lock-in).  
✔️ **Alta disponibilidad** – Redundancia entre diferentes nubes.  
✔️ **Optimización de rendimiento** – Se elige el mejor proveedor según la carga de trabajo.  

🚀 **Ejemplo:** Una empresa usa **Google Cloud para análisis de datos**, **AWS para almacenamiento** y **Azure para inteligencia artificial**.

### **📌 Conclusión**  
Cada tipo de nube tiene ventajas y desafíos:  

- **Nube pública**: Ideal para startups y empresas que buscan escalabilidad y costos bajos.  
- **Nube privada**: Perfecta para organizaciones con altos requerimientos de seguridad y cumplimiento.  
- **Nube híbrida**: Combina lo mejor de ambas, brindando flexibilidad y eficiencia.  
- **Multinube**: Adecuada para grandes empresas que necesitan resiliencia y evitar dependencias.

## ¿Qué es Cloud Native?

**Cloud Native** es un enfoque para diseñar, construir y operar aplicaciones aprovechando al máximo las capacidades de la computación en la nube. Se basa en principios como la escalabilidad, la resiliencia, la automatización y la eficiencia en el uso de los recursos.

### 📌 **Principales Características de Cloud Native**
1. **Microservicios**: Aplicaciones divididas en pequeños servicios independientes, cada uno con su propia lógica y base de datos.
2. **Contenedores**: Uso de tecnologías como Docker y Kubernetes para ejecutar aplicaciones de manera portable y eficiente.
3. **Orquestación y Automatización**: Uso de herramientas como Kubernetes para gestionar la escalabilidad y disponibilidad.
4. **DevOps y CI/CD**: Integración y entrega continua para agilizar el desarrollo y despliegue de aplicaciones.
5. **Escalabilidad Dinámica**: Uso de recursos en la nube bajo demanda para optimizar costos y rendimiento.
6. **Resiliencia y Auto-recuperación**: Arquitecturas diseñadas para tolerar fallos sin afectar la disponibilidad.
7. **Infraestructura como Código (IaC)**: Definición de infraestructuras mediante código para su gestión automatizada.

### 🔹 **Ejemplos de Tecnologías Cloud Native**
- **Contenedores**: Docker, Podman
- **Orquestación**: Kubernetes, OpenShift
- **Monitoreo y Logging**: Prometheus, Grafana, ELK Stack
- **Plataformas Serverless**: AWS Lambda, Azure Functions, Google Cloud Functions

### 🎯 **Ventajas de Cloud Native**
✅ Mayor velocidad de desarrollo  
✅ Mejor aprovechamiento de recursos  
✅ Escalabilidad bajo demanda  
✅ Resiliencia y alta disponibilidad  

**CNCF (Cloud Native Computing Foundation)** es una fundación que impulsa la adopción de tecnologías **Cloud Native** mediante el desarrollo de software de código abierto, estándares y mejores prácticas. CNCF forma parte de la **Linux Foundation** y es responsable de proyectos clave como **Kubernetes, Prometheus y Envoy**.

### 📌 **¿Qué hace CNCF?**
✅ **Desarrolla y mantiene proyectos de código abierto**: Kubernetes, Helm, gRPC, OpenTelemetry, etc.  
✅ **Define estándares para arquitecturas Cloud Native**.  
✅ **Fomenta la colaboración entre empresas y desarrolladores**.  
✅ **Organiza conferencias y certificaciones** (como la CKA – Certified Kubernetes Administrator).

### 🔹 **Principales proyectos de CNCF**
- **Kubernetes** 🛠️ (Orquestación de contenedores)  
- **Prometheus** 📊 (Monitoreo y alertas)  
- **Helm** 🎭 (Gestión de aplicaciones en Kubernetes)  
- **Envoy** 🔀 (Proxy y Service Mesh)  
- **OpenTelemetry** 📡 (Observabilidad y trazabilidad de datos)

### 🎯 **¿Por qué es importante CNCF?**
Si trabajas con **arquitecturas en la nube, DevOps o microservicios**, CNCF es clave porque ofrece tecnologías estandarizadas, seguras y escalables.  

Si te interesa explorar CNCF en tu trabajo o proyectos, dime qué tecnologías usas y te ayudo a integrarlas. 🚀


**Lecturas recomendadas**

[Cloud Native Computing Foundation](https://www.cncf.io/)

[Graduated and Incubating Projects | Cloud Native Computing Foundation](https://www.cncf.io/projects/)

## Ejemplo de arquitectura Cloud Native

Aquí tienes un ejemplo de una **arquitectura Cloud Native** basada en microservicios con Kubernetes en la nube.

### **🌐 Ejemplo: Plataforma de E-commerce Cloud Native**
Imagina que estamos construyendo una tienda en línea con pagos, catálogo de productos y usuarios.  

### **🔹 Componentes de la arquitectura Cloud Native**
1. **Frontend (React/Vue.js/Angular)**
   - Aplicación web servida como contenedor en **NGINX**.  
   - Se comunica con el backend a través de una **API REST o GraphQL**.  

2. **Backend (Microservicios con Python/Node.js/Go)**
   - Desplegado como microservicios en contenedores **Docker**.  
   - Cada microservicio maneja una parte de la lógica del negocio:
     - **Usuarios** 👤 (registro, login, autenticación JWT)
     - **Productos** 🛍️ (gestión del catálogo)
     - **Pagos** 💳 (procesamiento de transacciones con Stripe/PayPal)
     - **Órdenes** 📦 (gestión de compras y envíos)

3. **Base de Datos (Managed DB en la nube)**
   - **PostgreSQL/MySQL** en **Amazon RDS, Google Cloud SQL o Azure SQL**.  
   - **Redis** para caché de datos.  

4. **Orquestación con Kubernetes (K8s)**
   - Todos los microservicios corren en **pods de Kubernetes** en un clúster en la nube (**AWS EKS, Google GKE o Azure AKS**).  
   - Uso de **Helm** para gestionar los despliegues.  

5. **Service Mesh (Envoy/Istio/Linkerd)**
   - Maneja comunicación entre microservicios con seguridad y balanceo de carga.  

6. **Mensajería Asíncrona (Kafka/RabbitMQ)**
   - Para eventos como confirmaciones de pedidos, notificaciones y actualizaciones de stock.  

7. **Monitoreo y Logging**
   - **Prometheus** + **Grafana** para métricas.  
   - **ELK Stack (Elasticsearch, Logstash, Kibana)** para logs centralizados.  

8. **CI/CD (Integración y Entrega Continua)**
   - Uso de **GitHub Actions / GitLab CI / Jenkins** para automatizar despliegues.  
   - **Terraform o Pulumi** para Infraestructura como Código (IaC).

### **📌 Beneficios de esta arquitectura**
✅ **Escalabilidad dinámica** 📈 → Kubernetes ajusta la infraestructura según la demanda.  
✅ **Resiliencia** 🔄 → Si un microservicio falla, el sistema sigue funcionando.  
✅ **Agilidad** 🚀 → Equipos pueden desarrollar e implementar servicios de forma independiente.  
✅ **Optimización de costos** 💰 → Uso eficiente de recursos en la nube.

### **🛠️ Tecnologías utilizadas en el stack Cloud Native**
- **Infraestructura:** Kubernetes, Terraform  
- **Backend:** Node.js, Python (FastAPI), Go  
- **Base de Datos:** PostgreSQL, Redis  
- **Mensajería:** Kafka, RabbitMQ  
- **Monitoreo:** Prometheus, Grafana  
- **DevOps:** Docker, GitHub Actions

## ¿Qué es Serverless?

### **🚀 ¿Qué es Serverless?**  
**Serverless** es un modelo de computación en la nube donde los desarrolladores pueden ejecutar código sin administrar servidores. Aunque los servidores siguen existiendo, la nube los **provisiona, escala y administra automáticamente**, permitiendo a los desarrolladores enfocarse solo en el código.

### **📌 Características Clave de Serverless**
✅ **Sin gestión de servidores** → No necesitas configurar ni mantener infraestructura.  
✅ **Escalabilidad automática** → La plataforma ajusta los recursos según la demanda.  
✅ **Pago por uso** → Solo pagas cuando se ejecuta el código, lo que reduce costos.  
✅ **Ejecución basada en eventos** → Funciona en respuesta a eventos como peticiones HTTP, cargas de archivos o mensajes en colas.

### **🔹 Ejemplos de Plataformas Serverless**
- **AWS Lambda**  
- **Google Cloud Functions**  
- **Azure Functions**  
- **Cloudflare Workers**  
- **OpenFaaS (Open Source Serverless en Kubernetes)**

### **🔧 Ejemplo de Aplicación Serverless**  
### **Caso: API REST con AWS Lambda y API Gateway**  
1. **El usuario hace una solicitud HTTP** a una API (por ejemplo, para obtener información de un producto).  
2. **AWS API Gateway recibe la petición** y la envía a una función Lambda.  
3. **AWS Lambda ejecuta el código** (por ejemplo, consulta una base de datos en DynamoDB).  
4. **Lambda devuelve la respuesta al usuario**.  

📌 **Tecnologías utilizadas:**  
- AWS Lambda (ejecución del código sin servidor)
- API Gateway (manejo de solicitudes HTTP)  
- DynamoDB (base de datos escalable)  
- S3 (almacenamiento de archivos)

### **🎯 ¿Cuándo Usar Serverless?**  
🔹 Microservicios y APIs sin estado  
🔹 Procesamiento de eventos (archivos, notificaciones, IoT)  
🔹 Automatización de tareas (por ejemplo, generación de reportes)  
🔹 Aplicaciones que necesitan alta escalabilidad y baja latencia

**Lecturas recomendadas**

[Learning Serverless [Book]](https://www.oreilly.com/library/view/learning-serverless/9781492057000/)

