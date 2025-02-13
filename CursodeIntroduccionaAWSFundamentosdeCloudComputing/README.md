# Curso de Introducción a AWS: Fundamentos de Cloud Computing

## ¿Cómo aprender AWS en Platzi?

**Resumen**

Amazon Web Services (AWS) es el proveedor de servicios en la nube más grande del mundo. Muchas aplicaciones como Netflix, Meta y Platzi operan su arquitectura web gracias a su plataforma.

En este curso vamos a enfocarnos en los fundamentos del Cloud Computing (computación en la nube) de AWS.

### ¿Qué más vas a aprender?

Además de los conceptos de Cloud Computing, vas a conocer:

- La historia de AWS
- Regiones y zonas de disponibilidad en AWS
- Cómo crear tu cuenta de AWS
- Roles, seguridad e identidad en AWS

Sigue aprendiendo con [cursos de AWS](https://platzi.com/ruta/aws/).

## Visión general de las TI tradicionales

Las **Tecnologías de la Información (TI) tradicionales** abarcan el conjunto de sistemas, infraestructuras y prácticas que las empresas han utilizado durante décadas para gestionar datos, aplicaciones y procesos empresariales. A pesar del avance hacia la computación en la nube y soluciones más modernas, las TI tradicionales siguen siendo fundamentales en muchas organizaciones.

### **1. Características Claves de las TI Tradicionales**
- **Infraestructura local (On-Premise):** Servidores, bases de datos y almacenamiento físico dentro de la empresa.  
- **Modelo centralizado:** Gestión y control dentro de la organización sin depender de proveedores externos.  
- **Mantenimiento y actualización manual:** Se requiere personal de TI para administrar servidores, redes y software.  
- **Inversión inicial alta:** Costos significativos en hardware, software y mantenimiento.  
- **Seguridad y control total:** Mayor autonomía sobre la protección de los datos.  
- **Escalabilidad limitada:** La ampliación de la infraestructura requiere adquisición de nuevo hardware.  

### **2. Componentes Principales**
1. **Hardware:**  
   - Servidores físicos  
   - Computadoras y estaciones de trabajo  
   - Dispositivos de almacenamiento (HDD, SSD, NAS, SAN)  
   - Redes locales (LAN, routers, switches, firewalls)  

2. **Software:**  
   - Sistemas operativos empresariales (Windows Server, Linux)  
   - Bases de datos locales (Oracle, SQL Server, MySQL)  
   - Aplicaciones de gestión empresarial (ERP, CRM)  

3. **Redes y Seguridad:**  
   - Redes privadas internas  
   - Firewalls, VPNs y sistemas de autenticación  
   - Protocolos de seguridad internos para proteger datos

### **3. Ventajas y Desventajas de las TI Tradicionales**
| **Ventajas** | **Desventajas** |
|-------------|----------------|
| Mayor control sobre los datos y la infraestructura | Costos elevados de mantenimiento y actualización |
| Seguridad personalizada y sin depender de terceros | Escalabilidad limitada y lenta |
| Integración con sistemas internos y procesos específicos | Menos flexibilidad en la adopción de nuevas tecnologías |
| No depende de conexión a internet para operar | Requiere personal de TI capacitado para la gestión |

### **4. Comparación con las TI Modernas (Cloud Computing)**
| **Característica** | **TI Tradicional** | **TI en la Nube** |
|------------------|----------------|----------------|
| **Infraestructura** | Local, en servidores físicos | Basada en centros de datos remotos |
| **Escalabilidad** | Limitada y costosa | Flexible y basada en demanda |
| **Costos** | Alta inversión inicial | Pago por uso |
| **Mantenimiento** | A cargo del equipo de TI interno | Gestionado por el proveedor de la nube |
| **Accesibilidad** | Limitada a la red interna | Disponible desde cualquier lugar con internet |

### **5. Tendencias y Evolución**

A pesar de que muchas empresas están migrando a la nube, las TI tradicionales siguen siendo relevantes en sectores con altos requerimientos de seguridad y control, como el financiero y el gubernamental. Sin embargo, muchas organizaciones han adoptado un **modelo híbrido**, combinando infraestructura local con servicios en la nube para aprovechar lo mejor de ambos mundos.  

### **Conclusión**

Las TI tradicionales han sido la base del crecimiento tecnológico de muchas organizaciones, ofreciendo estabilidad y control. Sin embargo, con la evolución hacia la computación en la nube, muchas empresas están adoptando modelos más flexibles para mejorar la eficiencia y reducir costos. La decisión entre TI tradicional y moderna dependerá de factores como seguridad, escalabilidad y presupuesto.

### Resumen
Entendamos primero como funciona la web en términos simples. Tenemos un cliente con una dirección IP que se conecta a una red para hacer una petición a un servidor con otra dirección IP. Este servidor devuelve una respuesta al cliente.

Si la web fuera un servicio postal, el cliente seríamos nosotros con la petición o paquete que queremos enviar, la red sería el servicio postal en sí y el servidor representaría al destinatario al que queremos enviar el paquete.

[como-funciona-sitio-web.jpg](images/como-funciona-sitio-web.jpg)

### ¿Cómo está compuesto un servidor?

Un servidor posee los siguientes componentes:

- **Cómputo/CPU**: Realiza las operaciones que necesitamos.
- **Memoria RAM**: Contiene la información a procesar mediante la CPU. Es como un cerebro
- **Almacenamiento**: Archiva datos, a modo de archivos de texto plano.
- **Bases de datos**: Información almacenada de manera estructurada
- **Redes**: Cables, routers y servidores conectados unos a otros. Servidores DNS

### Terminología de IT (redes)

En términos generales, un cliente envía un paquete a un router, el cual reenvía este paquete al switch, y este se encarga de distribuirlo.

- **Router**: dispositivo que reenvía paquetes de datos entre redes informáticas
- **Switch**: dispositivo que toma paquetes y los envía al servidor/cliente correcto en la red

### Diseño tradicional de infraestructura

Las grandes empresas de IT empezaron comprando servidores y montándolos en sus garajes. Se encontraron con problemas al tratar de expandir su infraestructura, como los costos de mover estos servidores, comprar nuevos, y más…

### Problemas del enfoque de IT tradicional

A continuación, conocerás algunas dificultades del enfoque de tecnología de la información habitual:

- **Renta**: los costos de rentar espacios para mantener servidores son altos
- **Mantenimiento**: el funcionamiento de los servidores es difícil de asegurar
- **Remplazar y agregar hardware**: pueden existir largos tiempos de espera para conseguir el hardware necesario
- **Escalamiento limitado**: podemos vernos limitados por el espacio donde almacenamos los servidores
- **Monitoreo 24/7**: debemos contratar gente para monitorear los servidores
- **Desastres naturales**: ¿cómo evitamos que se caigan nuestros servicios si ocurre un imprevisto?

## Qué es la computacion en la nube

La **computación en la nube (Cloud Computing)** es un modelo de prestación de servicios informáticos a través de Internet. Permite acceder a recursos como servidores, almacenamiento, bases de datos, redes y software sin necesidad de infraestructura física local.  

### **1. Características Claves de la Computación en la Nube**
✅ **Accesibilidad:** Los servicios y datos están disponibles desde cualquier lugar con conexión a Internet.  
✅ **Escalabilidad:** Se pueden aumentar o reducir los recursos según la demanda.  
✅ **Pago por uso:** Se paga solo por los recursos utilizados, reduciendo costos iniciales.  
✅ **Mantenimiento gestionado:** Los proveedores de nube se encargan de actualizaciones y seguridad.  
✅ **Alta disponibilidad:** Ofrece redundancia y respaldo de datos para evitar fallos del sistema.

### **2. Modelos de Servicio en la Nube**
💡 Existen tres modelos principales de servicio en la computación en la nube:  

1️⃣ **IaaS (Infraestructura como Servicio):**  
   - Proporciona recursos básicos como servidores virtuales, almacenamiento y redes.  
   - Ejemplos: AWS EC2, Google Compute Engine, Microsoft Azure Virtual Machines.  

2️⃣ **PaaS (Plataforma como Servicio):**  
   - Ofrece entornos de desarrollo con herramientas para crear, probar y desplegar aplicaciones.  
   - Ejemplos: Google App Engine, AWS Elastic Beanstalk, Microsoft Azure App Services.  

3️⃣ **SaaS (Software como Servicio):**  
   - Aplicaciones listas para usarse sin necesidad de instalación o mantenimiento.  
   - Ejemplos: Gmail, Google Drive, Dropbox, Microsoft 365.  

### **3. Tipos de Nube**
🌍 Dependiendo del nivel de acceso y gestión, existen varios tipos de nube:  

🔹 **Nube Pública:** Administrada por proveedores como AWS, Google Cloud o Microsoft Azure, accesible para cualquier usuario.  
🔹 **Nube Privada:** Infraestructura dedicada a una sola empresa, proporcionando mayor control y seguridad.  
🔹 **Nube Híbrida:** Combinación de nube pública y privada, optimizando costos y rendimiento.  
🔹 **Multinube:** Uso de múltiples proveedores de nube para mayor flexibilidad y redundancia.

### **4. Comparación con la Infraestructura Tradicional**
| **Característica** | **Computación en la Nube** | **TI Tradicional** |
|-------------------|-----------------|----------------|
| **Costo Inicial** | Bajo, pago por uso | Alto, compra de servidores |
| **Mantenimiento** | Gestionado por el proveedor | Requiere personal de TI |
| **Escalabilidad** | Flexible y rápida | Limitada y costosa |
| **Accesibilidad** | Desde cualquier lugar | Requiere acceso a la red interna |
| **Seguridad** | Protección avanzada con cifrado y backups | Control total, pero mayor riesgo de fallos internos

### **5. Beneficios de la Computación en la Nube**
🔹 Reducción de costos operativos  
🔹 Mayor flexibilidad y escalabilidad  
🔹 Respaldo y recuperación de datos eficiente  
🔹 Acceso remoto y colaboración en tiempo real  
🔹 Innovación más rápida con despliegue ágil de aplicaciones

### **6. Desafíos y Consideraciones**
⚠️ **Seguridad:** Aunque los proveedores ofrecen cifrado y medidas de protección, la empresa debe gestionar el acceso y la privacidad de los datos.  
⚠️ **Latencia:** La velocidad de acceso a los datos depende de la conexión a Internet.  
⚠️ **Dependencia del proveedor:** Cambiar de proveedor puede ser complejo (vendor lock-in).  

### **Conclusión**
La computación en la nube ha transformado la manera en que las empresas y usuarios gestionan datos y aplicaciones, ofreciendo soluciones escalables, accesibles y eficientes. Sin embargo, la adopción de la nube debe evaluarse considerando necesidades de seguridad, costos y rendimiento

### Resumen

La computación en la nube es la entrega bajo demanda de **recursos de IT** como computación, almacenamiento y otros servicios a través de internet. En pocas palabras, es como si alquiláramos la computadora de otra persona.

Esta tecnología permite acceso instantáneo a los recursos que necesites, así como la adquisición del tipo y tamaño exacto de estos recursos. Algunos servicios que seguramente has usado son Gmail (proveedor de email), Dropbox (proveedor de almacenamiento) y Netflix (proveedor de video bajo demanda).

### Modelos de computación en la nube

A continuación, conocerás las distintas plataformas en la nube que utilizamos cuando trabajamos en proyectos personales o en nuestra empresa.

### Nube pública

La nube pública se refiere a los recursos de proveedores que utilizamos a través de internet y algunos ejemplos son Google Cloud Platform (GCP), Azure y AWS.

Además, posee estas ventajas:

- Elimina los gastos de capital comercial (CapEx) y reduce el gasto operativo (OpEx).
- Reduce los precios en economías de escala
- Despliega aplicaciones a nivel global en cuestión de minutos

### Nube privada

La nube privada es un servicio empleado por una organización que no está abierto al público. Permite un control total de la infraestructura y es útil para aplicaciones con requerimientos específicos de seguridad o comerciales.

### Nube híbrida

La nube híbrida consiste en mantener nuestra infraestructura y extender sus capacidades mediante la nube pública. Posibilita el control sobre activos sensibles en tu infraestructura privada, aprovechando la flexibilidad y rentabilidad de la nube pública.

### Características de la computación en la nube

Ahora que conoces los distintos modelos de tecnología en la nube, es importante que hablar sobre sus propiedades de computación.

- Este modelo genera un autoservicio en demanda (con registros en la plataforma ya se pueden proveer recursos)
- Tiene un amplio acceso a la red
- Proporciona un espacio donde los clientes pueden compartir infraestructura y recursos de manera segura

### Problemas resueltos por la nube

Por último, es crucial que conozcas las cualidades que trae implementar un sistema de computación en la nube.

- La nube aporta flexibilidad (puedes cambiar los tipos de recursos cuando sea necesario)
- Brinda rentabilidad y un servicio medido pues pagas solo por lo que usas
- Trae escalabilidad al agregar capacidad para hardware o equipos que necesitan acomodar cargas grandes
- Ofrece elasticidad al dar capacidad de escalar automáticamente cuando sea necesario
- Tiene alta disponibilidad y tolerancia a fallos
- Proporciona agilidad (puedes desarrollar, probar y ejecutar rápidamente aplicaciones en la nube)

**Lecturas recomendadas**

[Conoce la computación en la nube con AWS](https://platzi.com/blog/conoce-la-computacion-en-la-nube-con-aws/)

[Por qué decidimos usar Amazon Aurora en Platzi](https://platzi.com/blog/migramos-a-amazon-rds-aurora/)

[Conoce las nuevas funcionalidades de Bases de Datos en AWS](https://platzi.com/blog/funcionalidades-de-bases-de-datos-en-aws/)

[Estas son las nuevas funcionalidades de Storage AWS](https://platzi.com/blog/nuevas-funcionalidades-de-storage-aws/)

## Los diferentes tipos de cómputo: IaaS vs. PaaS vs. SaaS

En la computación en la nube, existen tres modelos principales de servicio: **IaaS, PaaS y SaaS**. Cada uno ofrece diferentes niveles de control, flexibilidad y responsabilidad para los usuarios. A continuación, exploramos sus características, ventajas y casos de uso.  

### **1. IaaS (Infraestructura como Servicio) 🏗️**  
**🔹 Definición:**  
IaaS proporciona acceso a infraestructura de TI virtualizada, como servidores, almacenamiento, redes y sistemas operativos. Es la opción más flexible, permitiendo a los usuarios configurar y administrar sus propios entornos.  

**🔹 Características:**  
✅ Recursos escalables bajo demanda  
✅ Pago por uso (sin necesidad de comprar hardware)  
✅ Mayor control sobre la infraestructura  
✅ Soporte para sistemas operativos y software personalizados  

**🔹 Ejemplos de IaaS:**  
🔹 Amazon EC2 (AWS)  
🔹 Google Compute Engine (GCP)  
🔹 Microsoft Azure Virtual Machines  

**🔹 Casos de Uso:**  
🔹 Creación de entornos de desarrollo y prueba  
🔹 Hosting de aplicaciones y sitios web  
🔹 Backup y recuperación de datos  
🔹 Implementación de máquinas virtuales  

### **2. PaaS (Plataforma como Servicio) 🚀**  
**🔹 Definición:**  
PaaS proporciona una plataforma para desarrollar, ejecutar y gestionar aplicaciones sin preocuparse por la infraestructura subyacente. Incluye herramientas de desarrollo, bases de datos y entornos de ejecución.  

**🔹 Características:**  
✅ Entorno preconfigurado para desarrollo de software  
✅ Reducción del tiempo de implementación  
✅ Escalabilidad automática  
✅ Soporte para múltiples lenguajes de programación  

**🔹 Ejemplos de PaaS:**  
🔹 Google App Engine  
🔹 AWS Elastic Beanstalk  
🔹 Microsoft Azure App Services  

**🔹 Casos de Uso:**  
🔹 Desarrollo de aplicaciones web y móviles  
🔹 Automatización de procesos de despliegue  
🔹 Integración continua y entrega continua (CI/CD)

### **3. SaaS (Software como Servicio) 📦**  
**🔹 Definición:**  
SaaS ofrece software listo para usar a través de Internet, sin necesidad de instalación o mantenimiento. El proveedor se encarga de la gestión de la aplicación, servidores y seguridad.  

**🔹 Características:**  
✅ No requiere instalación ni mantenimiento  
✅ Accesible desde cualquier dispositivo con Internet  
✅ Costos predecibles mediante suscripciones  
✅ Actualizaciones y soporte gestionados por el proveedor  

**🔹 Ejemplos de SaaS:**  
🔹 Google Drive  
🔹 Microsoft 365  
🔹 Dropbox  
🔹 Salesforce  

**🔹 Casos de Uso:**  
🔹 Aplicaciones de colaboración y productividad  
🔹 CRM (gestión de clientes)  
🔹 Almacenamiento en la nube  
🔹 Servicios de correo electrónico  

### **4. Comparación Entre IaaS, PaaS y SaaS**  

| **Característica** | **IaaS** | **PaaS** | **SaaS** |
|------------------|----------|----------|----------|
| **Nivel de control** | Alto (Infraestructura) | Medio (Plataforma) | Bajo (Software) |
| **Mantenimiento** | Usuario gestiona | Parcialmente gestionado | Totalmente gestionado |
| **Escalabilidad** | Alta | Alta | Limitada al proveedor |
| **Usuarios Objetivo** | Administradores de sistemas, DevOps | Desarrolladores | Usuarios finales |
| **Ejemplo de Uso** | Hosting de servidores | Desarrollo de aplicaciones | Uso de herramientas como Gmail, Zoom |


### **5. ¿Cuál Elegir?**
✅ **IaaS** si necesitas infraestructura escalable y control total.  
✅ **PaaS** si deseas enfocarte en desarrollo sin gestionar servidores.  
✅ **SaaS** si solo necesitas usar software sin preocuparte por la administración.

### Resumen
Ahora que conoces más sobre la tecnología en la nube, es importante introducir sus **distintos tipos de servicio** en la industria para identificar sus diferencias.

Estos modelos varían de acuerdo al tipo de servicio informático que pueda ofrecer, como servidores, almacenamiento, software o bases de datos.

### Infrastructure as a Service (IAAS)

La infraestructura como servicio (IAAS) proporciona componentes básicos de IT en la nube, es decir, **redes, computación, almacenamiento, etc**. A su vez, provee el máximo nivel de flexibilidad para adaptarlo a tus necesidades.

**Ejemplos:**

- Azure Virtual Machines
- Linode
- Digital ocean
- S2 AWS

### Platform as a Service (PAAS)

Los modelos que ofrecen una plataforma como servicio (PAAS) eliminan la necesidad de que administremos la infraestructura y proveen una plataforma para gestionar aplicaciones.

**Ejemplos:**

- Heroku
- Google App Engine
- AWS Elastic Beanstalk

### Software as a Service (SAAS)

El Software como servicio (SAAS) brinda un producto de software terminado que es ejecutado y administrado por el proveedor del servicio.

**Ejemplos:**

- Amazon Rekognition
- Dropbox
- Zoom
- Gmail

### On -premises

On-premises se refiere a una forma tradicional de cómputo en la cual nos encargamos de gestionar nuestra propia infraestructura.

### Responsabilidades según el tipo de cómputo

En la siguiente tabla se muestra qué componentes de IT están administrados según el tipo de cómputo en la nube. “Sí” indica que el componente está administrado por el proveedor de nube, “No” indica que nosotros somos responsables del componente.

Componente | On-premises | IAAS | PAAS  | SAAS
|---|---|---|---|
Aplicaciones | No | No | No | Sí
Data | No | No | No | Sí
Runtime | No | No | Sí | Sí
Middleware | No | No | Sí | Sí
O/S | No | No | Sí | Sí
Virtualización | No | Sí | Sí | Sí
Servidores | No | Sí | Sí | Sí
Almacenamiento | No | Sí | Sí | Sí
Redes | No | Sí | Sí | Sí

**Lecturas recomendadas**

[IaaS vs. PaaS | Platzi](https://platzi.com/blog/iaas-vs-paas/)

## Una pequeña historia de AWS

Amazon Web Services (AWS) es la plataforma de computación en la nube más grande y popular del mundo. Su historia es un ejemplo de innovación y crecimiento exponencial en el mundo tecnológico.  

## **📌 Los Inicios de AWS (2000 - 2006)**  
A principios de los años 2000, Amazon, la empresa de comercio electrónico fundada por **Jeff Bezos**, se dio cuenta de que su infraestructura tecnológica podía ser optimizada. Su equipo de ingenieros desarrolló herramientas internas para administrar servidores, almacenamiento y bases de datos de manera más eficiente.  

En 2003, Amazon identificó una oportunidad: las empresas y desarrolladores enfrentaban problemas similares con la gestión de infraestructura tecnológica. ¿Por qué no ofrecer estos servicios como una plataforma de computación en la nube?  

En 2006, **Amazon lanzó oficialmente AWS**, con sus primeros tres servicios:  
✅ **Amazon S3** (Simple Storage Service) → Almacenamiento en la nube  
✅ **Amazon EC2** (Elastic Compute Cloud) → Servidores virtuales escalables  
✅ **Amazon SQS** (Simple Queue Service) → Servicio de mensajería  

### **🚀 Crecimiento y Expansión (2007 - 2015)**  
AWS creció rápidamente al ofrecer servicios escalables y de bajo costo para startups y empresas grandes. En esta etapa, compañías como **Netflix, Airbnb y Dropbox** adoptaron AWS para reducir costos y escalar globalmente.  

Se lanzaron más servicios innovadores:  
🔹 **RDS (Relational Database Service)** en 2009  
🔹 **AWS Lambda** (computación sin servidores) en 2014  
🔹 **Amazon Aurora** (base de datos en la nube) en 2015  

### **🌍 Dominio del Mercado (2016 - Presente)**  
Hoy, AWS es el líder del mercado de la nube, con **más del 30% de participación global**. Ofrece más de **200 servicios** en inteligencia artificial, IoT, big data, seguridad y más.  

🔹 Empresas como **NASA, Netflix, Facebook y BMW** confían en AWS.  
🔹 AWS sigue innovando con servicios como **Amazon SageMaker** (Machine Learning) y **AWS Outposts** (nube híbrida).  
🔹 Ha expandido su presencia global con **más de 30 regiones y 100 zonas de disponibilidad**.  

### **🔥 Impacto de AWS en la Tecnología**  
✅ Ha revolucionado la industria de la computación en la nube.  
✅ Ha permitido la escalabilidad de startups y grandes empresas.  
✅ Ha reducido costos de infraestructura con su modelo de pago por uso.  


**Benjamin Black** y **Chris Pinkham** son los principales desarrolladores de Amazon Web Services y crearon esta compañía a partir de la necesidad de impulsar nuevas tecnológicas en momentos de mayor tráfico y demanda.

La historia de AWS está repleta de hitos, pues es una de las plataformas más utilizadas en distintas startups y compañías que están transformando su industria. ¡No te preocupes! Aquí te resumimos fácil su línea del tiempo.

### Línea del tiempo de AWS

Hace veinte años nació esta promesa tecnológica y en la actualidad ¡tiene clientes en más de 245 países y territorios!

- 2002 → Se lanza internamente la plataforma
- 2003 → Comienza a comercializarse la idea de AWS
- 2004 → Llega al público el servicio SQS
- 2006 → Vuelve a lanzarse al público SQS, S3 y EC2
- 2007 → Abren operaciones en Europa
- 2009 → Lanzan el servicio RDS (Relational Database)
- 2010 → Sale al mercado el servicio Route 53
- 2012 → Lanzan DynamoDB (una base de datos no relacional)

### AWS en números

Quizás sean un gran seguidor y fiel cliente de esta compañía, pero… ¿Conocías estos datos?

- En 2019, AWS logró $35.02 mil millones de dólares en ingresos anuales
- AWS representó el 47% del mercado en 2019
- Esta plataforma posee más de un millón de usuarios activos

**Lecturas recomendadas**

[¿Qué es Amazon Web Services?](https://platzi.com/blog/aws-curso/)

## Una visión global: regiones y zonas de disponibilidad


Las plataformas de computación en la nube, como **Amazon Web Services (AWS), Microsoft Azure y Google Cloud Platform (GCP)**, estructuran su infraestructura en **regiones y zonas de disponibilidad (Availability Zones - AZs)** para proporcionar alta disponibilidad, tolerancia a fallos y baja latencia en sus servicios.  

### **1. ¿Qué es una Región?**  
Una **región** es un área geográfica global donde un proveedor de nube tiene **varios centros de datos** interconectados. Cada región opera de manera independiente y tiene su propio conjunto de servicios en la nube.  

### **Ejemplos de Regiones en AWS, Azure y GCP**  
- **AWS:** us-east-1 (Norte de Virginia), eu-west-1 (Irlanda), sa-east-1 (São Paulo).  
- **Azure:** East US, West Europe, Southeast Asia.  
- **GCP:** us-central1 (Iowa), europe-west1 (Bélgica), asia-northeast1 (Tokio).  

### **Características Clave de las Regiones:**  
✅ **Ubicación Geográfica Estratégica:** Facilita la cercanía a los usuarios finales.  
✅ **Redundancia y Seguridad:** Asegura disponibilidad en caso de fallos.  
✅ **Cumplimiento Normativo:** Algunas regiones cumplen regulaciones específicas como GDPR en Europa o LGPD en Brasil.

### **2. ¿Qué es una Zona de Disponibilidad (AZ)?**  
Una **Zona de Disponibilidad (AZ)** es un **conjunto de uno o más centros de datos físicamente separados dentro de una misma región**.  

Cada AZ tiene:  
- **Conectividad de baja latencia** con otras zonas dentro de la misma región.  
- **Fuentes de energía y refrigeración independientes**, reduciendo riesgos de fallos.  
- **Alta disponibilidad y recuperación ante desastres**, permitiendo distribuir cargas de trabajo.  

### **Ejemplo en AWS:**  
La región **us-east-1 (Norte de Virginia)** tiene **6 zonas de disponibilidad** (us-east-1a, us-east-1b, us-east-1c, etc.).

### **3. Diferencia Clave entre Región y Zona de Disponibilidad**  

| **Característica**  | **Región** | **Zona de Disponibilidad (AZ)** |
|--------------------|------------|-------------------------------|
| **Alcance**       | Área geográfica global | Conjunto de centros de datos dentro de una región |
| **Tolerancia a Fallos** | Independiente de otras regiones | Alta disponibilidad dentro de la región |
| **Interconexión**  | Comunicación entre regiones con latencias más altas | Comunicación de baja latencia entre AZs |
| **Ejemplo AWS**    | us-east-1 (Norte de Virginia) | us-east-1a, us-east-1b, us-east-1c |

### **4. Beneficios de Usar Múltiples Regiones y AZs**  
✅ **Alta Disponibilidad:** Si una zona falla, los servicios siguen operando en otra.  
✅ **Baja Latencia:** Se elige la región más cercana al usuario.  
✅ **Recuperación ante Desastres:** Se pueden replicar datos en múltiples regiones.  
✅ **Cumplimiento Normativo:** Algunas regiones cumplen requisitos específicos de privacidad y seguridad.

### **5. Ejemplo de Implementación en AWS**  
📌 **Escenario:** Una empresa de streaming quiere alta disponibilidad y rapidez para usuarios en EE.UU. y Europa.  
🔹 **Solución:**  
1. **Distribuir servidores en dos regiones:** us-east-1 (Norte de Virginia) y eu-west-1 (Irlanda).  
2. **Usar múltiples AZs en cada región:** Balancear la carga entre ellas.  
3. **Implementar bases de datos replicadas en diferentes AZs:** Para garantizar continuidad en caso de fallos.

### **Conclusión**  
Las **regiones y zonas de disponibilidad** son la base de la infraestructura en la nube moderna. Elegir la **ubicación correcta** es clave para garantizar **rendimiento, disponibilidad y cumplimiento normativo** en las aplicaciones. 🌍☁️

### Resumen

La infraestructura de AWS está compuesta por **regiones**, **zonas de disponibilidad**, **data centers** y puntos de presencia. Además, se distribuye en diferentes regiones alrededor del mundo. Algunas de ellas son Ohio, Oregon, Norte de California, e incluso lugares exclusivos del gobierno de EE. UU. como GovCloud Este.

Si quieres conocer una lista completa con más sitios, puedes visitar esta página de AWS.

### Cómo escoger una región de AWS

Podemos escoger la región de nuestra aplicación basada en distintos aspectos que mencionaremos a continuación.

**Por ejemplo:**

- El cumplimiento de los requisitos legales y de gobernanza de datos, pues los datos nunca abandonan una región sin su permiso explícito

- La proximidad con los clientes porque lanzan en una región cercana en donde estén para reducir latencia. Puedes revisar esta característica desde tu ubicación a cada región en cloudping.info.

- Los servicios disponibles dentro de una región debido a que muchos no funcionan en todas partes. Algunos servicios globales o regionales son…

- Globales
 - IAM
 - Route 53
 - Cloudfront
 - WAF
- Regionales
 - EC2
 - Beanstalk
 - Lambda
 - Rekognition
 
Los precios varían de región a región y son transparentes en la página de precios del servicio

### Zonas de disponibilidad

Una zona de disponibilidad es un grupo de data centers donde cada uno está lleno de servidores. Estos *data centers* poseen energía, redes y conectividad redundante, están separados entre sí, conectados con un gran ancho de banda y redes de latencia ultra baja.

### Modelo de responsabilidad compartida

Ahora es crucial determinar las responsabilidades de AWS y del cliente dentro del servicio tecnológico que ofrece la compañía.

**AWS se hace responsable de:**

- Hardware y la infraestructura global
- Regiones
- Zonas de disponibilidad
- Ubicaciones de AWS Edge / puntos de presencia
- Software
- Cómputo
- Almacenamiento
- Bases de datos
- Redes

### El cliente se responsabiliza de:

- Actualizaciones de S.O.
- Protección de los datos que se almacenan
- Manejo de aplicaciones
- Accesos
- Administración de usuarios y grupos

**Lecturas recomendadas**

[Infraestructura global](https://aws.amazon.com/es/about-aws/global-infrastructure/?p=ngi&amp;loc=0)

[cloudping.info](https://www.cloudping.info/)

[Ping Amazon web services](https://pingaws.com/)

[Servicios regionales de AWS](https://aws.amazon.com/es/about-aws/global-infrastructure/regional-product-services/)

[Reconocimiento de objetos en JavaScript con Serverless Framework y AWS Rekognition](https://platzi.com/blog/serverless-framework-y-aws-rekognition/)

[AWS Ping Test (Latency) | Cloud Ping Test](https://cloudpingtest.com/aws)