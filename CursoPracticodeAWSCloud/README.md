# Curso Práctico de AWS Cloud

## ¿Qué es el cómputo en la nube?

El **cómputo en la nube** (o *cloud computing*) es un modelo de entrega de servicios tecnológicos que permite acceder a recursos de cómputo —como servidores, almacenamiento, bases de datos, redes, software y más— a través de internet y bajo demanda, sin necesidad de tener infraestructura física propia.

### 🔍 ¿Qué significa esto en palabras simples?

Imagina que en lugar de tener tus propios servidores o computadoras potentes en una oficina, puedes "alquilar" capacidad computacional desde centros de datos gigantes operados por empresas como **Amazon Web Services (AWS)**, **Microsoft Azure**, **Google Cloud**, entre otras.

### ✅ Ventajas del cómputo en la nube:

- **Escalabilidad:** Puedes aumentar o disminuir recursos según lo necesites.
- **Pago por uso:** Solo pagas por lo que usas, sin necesidad de grandes inversiones iniciales.
- **Accesibilidad:** Puedes acceder a tus servicios y datos desde cualquier lugar con internet.
- **Seguridad:** Proveedores en la nube invierten mucho en proteger tus datos e infraestructura.
- **Rapidez:** Lanzar nuevos servicios es más rápido que instalar y configurar hardware físico.

### 🧠 Tipos de servicios en la nube:

1. **IaaS (Infraestructura como Servicio)** – Ej: Amazon EC2.
2. **PaaS (Plataforma como Servicio)** – Ej: AWS Lambda, Heroku.
3. **SaaS (Software como Servicio)** – Ej: Gmail, Dropbox, Google Docs.

## ¿Cómo puedo empezar a usar AWS?

¡Excelente decisión! 🙌 Empezar a usar **AWS** es más fácil de lo que parece. Aquí te dejo una guía paso a paso para comenzar bien:

### 🚀 1. **Crea una cuenta gratuita en AWS**
Ve a 👉 [https://aws.amazon.com/es/free](https://aws.amazon.com/es/free)

- Solo necesitas una tarjeta de crédito o débito (no te cobrarán si te mantienes en el **Free Tier**).
- Asegúrate de **activar la autenticación multifactor (MFA)** por seguridad.
### 🧰 2. **Accede a la consola de administración (AWS Console)**
👉 [https://console.aws.amazon.com](https://console.aws.amazon.com)

Desde ahí puedes crear y administrar todos los servicios (EC2, S3, Lambda, etc.).

### 🛠️ 3. **Elige una herramienta para trabajar**
Tienes tres formas principales de interactuar con AWS:

| Opción           | ¿Para qué sirve?                          |
|------------------|-------------------------------------------|
| **Consola Web**  | GUI visual para empezar fácilmente.       |
| **AWS CLI**      | Línea de comandos para usuarios avanzados.|
| **SDK (como boto3, aws-sdk)** | Automatización desde Python, Node.js, etc. |

### 💡 4. **Haz tu primer proyecto práctico**
Te propongo algunos proyectos simples para empezar:

| Proyecto                              | Servicios involucrados               |
|---------------------------------------|--------------------------------------|
| 🚀 “Hola Mundo” con Lambda            | Lambda + API Gateway                 |
| ☁️ Almacenar archivos                 | S3                                   |
| 📨 Sistema de colas                   | SQS                                  |
| 🧠 Clasificador de imágenes básico    | Lambda + Rekognition + S3            |
| 🧾 CRUD de usuarios sin servidores     | Lambda + DynamoDB + API Gateway      |

### Resumen

Para crear tu cuenta de AWS debes entrar a [aws.amazon.com](https://aws.amazon.com/ "aws.amazon.com"), recuerda que para crear tu contraseña puedes usar [passwordsgenerator.net](https://passwordsgenerator.net/ "passwordsgenerator.net").

AWS dispone de dos programas que permiten a los clientes trasladar sus enseñanzas y esfuerzos de investigación a la nube y de este modo innovar más rápidamente y con un menor costo. Para aplicar a las becas de AWS entra a [aws.amazon.com/es/grants](https://aws.amazon.com/es/grants/ "aws.amazon.com/es/grants").

Cuando tengas tu cuenta registrada, entra a console.aws.amazon.com y tendrás acceso a la consola de amazon con todos los servicios disponibles.

**Lecturas recomendadas**

[Strong Random Password Generator](https://passwordsgenerator.net/)

## Introducción a la oferta de servicios de AWS y sus aplicaciones

¡Perfecto! Vamos con una introducción clara a la **oferta de servicios de AWS (Amazon Web Services)** y sus aplicaciones en el mundo real 🌍💻🚀

### ☁️ ¿Qué es AWS?
**AWS** es la **plataforma de servicios en la nube más usada del mundo**, que ofrece más de 200 servicios bajo demanda como almacenamiento, computación, bases de datos, inteligencia artificial, redes, y más.

Su principal ventaja: **pagar solo por lo que usas**, sin necesidad de comprar infraestructura física.

### 🔍 Categorías de Servicios AWS y sus Aplicaciones

| Categoría                     | Servicio Destacado       | Aplicación Común                                                                 |
|------------------------------|--------------------------|----------------------------------------------------------------------------------|
| 🧠 **Cómputo**               | AWS Lambda, EC2          | Ejecutar aplicaciones o funciones sin servidor (serverless), o con servidores dedicados |
| 💾 **Almacenamiento**        | S3, EBS, Glacier         | Guardar archivos, backups, sitios estáticos, copias de seguridad                |
| 🗃️ **Bases de Datos**         | RDS, DynamoDB            | Apps web, móviles, bases de datos SQL o NoSQL para gestión de datos             |
| 🔐 **Seguridad y acceso**    | IAM, Cognito             | Control de acceso a servicios y usuarios, autenticación                         |
| 🕸️ **Redes y CDN**           | VPC, CloudFront          | Entrega de contenido global rápido y seguro                                     |
| 📩 **Mensajería y colas**    | SQS, SNS                 | Comunicación entre componentes de aplicaciones asincrónicamente                 |
| 📊 **Monitoreo y análisis**  | CloudWatch, X-Ray        | Logs, métricas y trazas para diagnóstico y rendimiento                          |
| 🤖 **Inteligencia Artificial**| Rekognition, Comprehend  | Reconocimiento facial, análisis de texto, traducción automática                 |
| 🧪 **Desarrollo y DevOps**   | CodeDeploy, CloudFormation | Automatización del despliegue e infraestructura como código                     |

### 🌐 Aplicaciones Comunes en la Vida Real

- 👨‍💻 **Startups y apps móviles:** Backend 100% serverless con Lambda + API Gateway + DynamoDB.
- 🛒 **E-commerce:** Catálogo y órdenes con S3, RDS, CloudFront, Lambda.
- 🧠 **ML y análisis de datos:** Entrenamiento de modelos con SageMaker o análisis con Athena y QuickSight.
- 🏢 **Empresas tradicionales:** Migración de sistemas locales a la nube con EC2, RDS y VPC.

### 🧩 Ventajas de usar AWS

- 🔄 Escalabilidad automática.
- 🔒 Seguridad de clase mundial.
- 🌍 Infraestructura global.
- 💸 Costos optimizados (Free Tier incluido).

### Resumen

#### ¿Qué es AWS y qué servicios ofrece?

AWS, Amazon Web Services, es hoy en día un referente esencial en el ámbito de la computación en la nube. Ofrece una amplia gama de servicios que pueden resolver casi cualquier necesidad tecnológica que se presente. Desde máquinas virtuales hasta servicios de inteligencia artificial, AWS tiene algo para todos, y con la cuenta que ya registraste, puedes explorar estas posibilidades.

#### ¿Cómo gestionar el cómputo y almacenamiento en AWS?

AWS posee varias opciones dentro de su sección de cómputo. Puedes elegir entre máquinas virtuales, infraestructura o servicios serverless como Lambda, dependiendo de tu necesidad específica. En cuanto al almacenamiento, AWS te permite guardar archivos para su servicio en un sitio web o aplicación móvil, o simplemente para conservarlos de manera indefinida, como por ejemplo para propósitos fiscales.

#### ¿Qué opciones ofrece AWS para bases de datos?

AWS tiene una sección notablemente robusta cuando se trata de bases de datos. Proporciona opciones tradicionales como PostgreSQL y MySQL, además de otras recientes que pueden personalizarse según tus necesidades.

#### ¿Cuál es la propuesta de AWS en migración de datos?

La migración de servicios es otro punto fuerte de AWS. Te ofrece la capacidad de trasladar la información desde un data center existente a la infraestructura de Amazon. Esto es ideal para empresas que desean aprovechar las ventajas de la nube sin interrumpir sus operaciones actuales.

#### ¿Cómo mejorar el desarrollo y la gestión a través de AWS?
#### ¿Qué herramientas de desarrollo existen en AWS?

AWS aporta diversas herramientas para desarrolladores como Xcode que ayuda a analizar las aplicaciones en funcionamiento y detectar posibles mejoras. También está Code Commit que facilita la gestión del código.

#### ¿Qué ofrece AWS para la administración de infraestructura?

En términos de administración, AWS dispone de herramientas fundamentales como CloudWatch, que permite monitorear los recursos y servidores para identificar demandas de red o CPU. CloudTrail es vital para realizar auditorías y entender quién hace qué dentro de la infraestructura, útil para mantener la seguridad y el control.

#### ¿Qué otros servicios especializados ofrece AWS?
#### ¿Cómo te ayuda AWS en la transmisión de medios?

AWS cuenta con el servicio Elastic Transcoder, ideal para quienes necesitan realizar streaming de eventos, como un partido de fútbol, generando múltiples versiones de video para diferentes dispositivos.

#### ¿Qué soluciones de machine learning ofrece AWS?

AWS ha integrado soluciones de machine learning que son de gran ayuda. Recognition, por ejemplo, permite enviar imágenes para reconocimiento automático, ideal para trabajar con API o páginas web.

#### ¿Qué capacidades de análisis avanzado encuentras en AWS?

AWS no solo proporciona estadísticas básicas de visitantes del sitio. Sus servicios de analytics permiten profundizar en detalles técnicos como el uso de RAM y CPU por usuario.

#### ¿Cómo gestiona AWS la seguridad y accesibilidad?

La sección de seguridad de AWS incluye IAM (Identity and Access Management) que permite definir accesos granulares a diferentes recursos específicos. AWS también cuenta con Guard Duty para la detección y alerta de accesos no autorizados o ataques.

#### ¿Cuáles son las áreas emergentes y de futuro que abarca AWS?

AWS sigue expandiendo su oferta para abarcar áreas como la realidad virtual, la integración de aplicaciones y el internet de las cosas. También integra soluciones empresariales como Amazon Connect para mejorar la productividad y las conexiones de negocio.

Estos puntos resaltados son solo una muestra del vasto ecosistema que ofrece AWS. Explora estos servicios para descubrir cómo pueden ayudarte a potenciar tu próximo proyecto tecnológico.

## Ejemplo de arquitectura con Elastic Beanstalk

¡Claro! Aquí tienes un ejemplo sencillo pero muy útil de **arquitectura con AWS Elastic Beanstalk** y cómo se integra con otros servicios de AWS. Elastic Beanstalk es ideal si quieres desplegar aplicaciones sin preocuparte por la infraestructura subyacente.

## 📐 Ejemplo de Arquitectura con Elastic Beanstalk

### Caso de uso: Aplicación Web con Backend en Node.js + Base de Datos MySQL

### 🧱 Componentes

```
┌──────────────────────────┐
│        Usuarios          │
└────────────┬─────────────┘
             │
             ▼
   ┌─────────────────────┐
   │    Amazon Route 53  │  ← (DNS opcional, apunta a tu app)
   └────────┬────────────┘
            ▼
   ┌─────────────────────┐
   │  Elastic Load Balancer│
   └────────┬────────────┘
            ▼
   ┌─────────────────────┐
   │ AWS Elastic Beanstalk│  ← (App en Node.js o cualquier stack)
   │    (EC2 autoscaling) │
   └────────┬────────────┘
            ▼
   ┌─────────────────────┐
   │ Amazon RDS (MySQL)  │  ← (Base de datos relacional)
   └─────────────────────┘
```

### 🚀 Flujo de trabajo

1. **Desarrollas tu app** localmente en Node.js (o Python, Java, .NET, etc.).
2. **Empaquetas** el código (por ejemplo, `.zip`) o haces `git aws.push`.
3. **Elastic Beanstalk**:
   - Lanza las instancias EC2.
   - Configura el balanceador de carga.
   - Administra el auto escalado.
   - Proporciona monitoreo con CloudWatch.
4. Tu app se conecta a **RDS** para la base de datos.
5. Todo esto con mínima configuración de infraestructura.

### 🛠️ Beneficios

- ✅ Despliegue simplificado (solo subes tu código).
- ✅ Integración automática con EC2, ELB, RDS.
- ✅ Escalabilidad y balanceo de carga gestionados.
- ✅ Puedes personalizar con `.ebextensions`.

### 🧑‍💻 Comando básico para desplegar

```bash
eb init -p node.js my-app
eb create my-env
eb deploy
```

**Ejemplo** 

¡Perfecto! Aquí tienes un ejemplo completo de una arquitectura básica con AWS Elastic Beanstalk, usando una aplicación Node.js y conectándola a una base de datos RDS MySQL. Incluiré:

- Código base de la app.
- Archivos de configuración `.ebextensions`.
- Recomendaciones de despliegue.

### 📁 Estructura de archivos

```
myapp/
├── .ebextensions/
│   └── rds.config
├── app.js
├── package.json
└── Procfile
```

### 📄 `app.js`

```js
const express = require('express');
const mysql = require('mysql');
const app = express();

const connection = mysql.createConnection({
  host: process.env.RDS_HOSTNAME,
  user: process.env.RDS_USERNAME,
  password: process.env.RDS_PASSWORD,
  database: process.env.RDS_DB_NAME,
});

app.get('/', (req, res) => {
  connection.query('SELECT NOW() AS now', (err, results) => {
    if (err) {
      res.status(500).send('Database error');
    } else {
      res.send(`Connected to DB. Time is: ${results[0].now}`);
    }
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`App listening on port ${PORT}`);
});
```

### 📄 `package.json`

```json
{
  "name": "eb-node-app",
  "version": "1.0.0",
  "main": "app.js",
  "scripts": {
    "start": "node app.js"
  },
  "dependencies": {
    "express": "^4.17.1",
    "mysql": "^2.18.1"
  }
}
```

### 📄 `Procfile`

```
web: node app.js
```

### 📄 `.ebextensions/rds.config`

```yaml
option_settings:
  aws:elasticbeanstalk:application:environment:
    RDS_HOSTNAME: your-db-host.rds.amazonaws.com
    RDS_PORT: '3306'
    RDS_DB_NAME: yourdbname
    RDS_USERNAME: youruser
    RDS_PASSWORD: yourpassword
```

🔐 **Importante**: Para producción, usa secretos de AWS SSM Parameter Store o Secrets Manager, y no expongas tus credenciales en texto plano.

### 🚀 ¿Cómo desplegar?

1. **Inicializa Elastic Beanstalk:**
   ```bash
   eb init
   ```
2. **Crea un entorno (por ejemplo Node.js):**
   ```bash
   eb create myapp-env
   ```
3. **Despliega:**
   ```bash
   eb deploy
   ```

### Resumen

Hay muchas formas de desarrollar un sistema en AWS, una de ellas es la arquitectura Elastic Beanstalk.

Esta arquitectura tiene como ventaja la alta disponibilidad y la eficiencia para atender una gran cantidad de usuarios.

**Archivos de la clase**

[aws.pdf](https://static.platzi.com/media/public/uploads/aws_a42899ad-91a0-4213-9bce-d79e5f557dbc.pdf)

## ¿Qué es Amazon EC2?

Amazon EC2 (Elastic Compute Cloud) es un **servicio de cómputo escalable en la nube de AWS** que te permite lanzar y administrar **instancias de servidores virtuales** (máquinas virtuales) en la nube.

### ✅ ¿Para qué sirve EC2?

- **Alojar aplicaciones web** (como sitios en WordPress, Django, Node.js, etc.).
- **Correr bases de datos o servidores backend**.
- **Simulaciones, cálculos científicos o tareas batch**.
- **Desarrollar, probar y desplegar software**.
- **Correr contenedores o incluso clústeres de Kubernetes (EKS)**.

### 🧠 Características clave:

| Característica         | Descripción                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Escalabilidad**      | Puedes iniciar desde 1 hasta cientos o miles de instancias rápidamente.     |
| **Flexibilidad**       | Puedes elegir SO (Linux, Windows), tamaño, tipo de CPU, memoria, etc.       |
| **Elastic IP**         | Asigna una IP pública fija a tu instancia si lo necesitas.                  |
| **Precios por demanda**| Pagas por segundo/minuto mientras la instancia está activa.                 |
| **Integración**        | Se integra con otros servicios como S3, RDS, CloudWatch, IAM, etc.          |

### 💡 Ejemplo de uso:

Supón que desarrollaste una API en Node.js y necesitas un servidor donde alojarla. Puedes:

1. Crear una instancia EC2.
2. Instalar Node.js y subir tu código.
3. Exponer tu API al mundo (con un DNS o IP).
4. Usar un balanceador de carga si escalas a múltiples instancias.

### Resumen

EC2 (**Amazon Elastic Compute Cloud**) es un servicio de ****AWS (Amazon Web Services) que permite alquilar máquinas virtuales, llamadas instancias EC2. Puedes elegir diferentes tipos de EC2 con diferente CPU, RAM y almacenamiento. Hay instancias optimizadas para cómputo, memoria y almacenamiento, entre otras.

En **EC2**, el sistema de pago más común es por hora o por segundo, dependiendo el tipo de instancia. Por ejemplo, para una instancia que cueste $0.1 la hora, puedes pagar, ya sea una instancia por 24 horas o 24 instancias por una hora. En ambos casos pagas lo mismo (24 * 0.10 = $2.4).

#### Características de Amazon EC2

Amazon EC2 lo puedes utilizar para ejecutar tus propias aplicaciones, tanto de calidad como desarrollo o incluso producción. Estas son algunas características para tener en cuenta:

#### Instancias

Máquinas virtuales con diversas opciones de Sistema Operativo, CPU, RAM y disco duro, entre otros.

#### Seguridad

Generación de llaves únicas para conectarte a tu máquina Linux o Windows de forma segura. Es posible generar diferentes llaves o claves para diversas máquinas.

#### Espacio

Diversas opciones de espacio en disco duro y es virtualmente infinito. Puedes anexar recursos en cualquier momento, si lo necesitas.

#### Redundancia

Es posible tener diversas copias de la misma máquina en diversas regiones geográficas.

#### Firewall

Puedes controlar desde dónde te puedes conectar a la máquina y a través de qué puertos. Además, es posible hacer modificaciones en términos de servicios y es muy fácil crear las reglas del firewall.

#### Direcciones IP estáticas

Puedes optar por comprar una IP pública estática para que siempre puedas poner la última versión o la última máquina en esa IP.

#### Respaldos

Puedes respaldar toda la máquina (ambiente, sistema operativo) cada vez que lo necesites. Además, los respaldos los puedes mover a diferentes regiones y tener múltiples copias disponibles.

#### Escalable

Es posible incrementar o disminuir los recursos de la máquina (más vCPUs, más RAM, entre otros) cuando lo necesites.

#### Migración de snapshot

Puedes copiar un snapshot a otras regiones, en caso de que algo suceda en la que estás trabajando o para agilizar la conexión desde otros lugares.

#### Opciones y precios bajo demanda
Las instancias pueden redimiensionarse. Puedes empezar por una instancia de bajo costo, y si necesitas aumenta su capacidad, apagas la instancia y seleccionas un nuevo tipo de instancia. Cuando enciendas de nuevo la instancia, verás su capacidad aumentada. La siguiente tabla muestra algunos tipos de instancias.

| Nombre | Especificaciones | Precio |
|---|---|---|
| t3.nano | 2 vCPU’s, 0.5 GiB RAM | $0,0052/hora |
|t3.xlarge | 4 vCPU’s, 16 GiB RAM | $0,1664/hora |
|c6g.8xlarge | 32 vCPU’s, 64 GiB RAM | $1,088/hora |
|X1e.xlarge | 128 vCPU’s, 3904 GiB RAM, 2x 1920 GB SSD | $26,688/hora |

## Creando una instancia de EC2

¡Perfecto! Vamos a crear una instancia EC2 paso a paso desde la consola de AWS. Esto te permitirá tener un **servidor virtual** corriendo en minutos. Ideal para alojar una web, una API o hacer pruebas.

### 🛠️ **Pasos para crear una instancia EC2 desde la consola de AWS**

#### 1. **Ingresar a la consola de EC2**
- Ve a [https://console.aws.amazon.com/ec2](https://console.aws.amazon.com/ec2)
- Asegúrate de estar en la **región correcta** (por ejemplo: N. Virginia o São Paulo).

#### 2. **Lanzar una nueva instancia**
Haz clic en **“Launch Instance” (Lanzar instancia)**.

#### 3. **Configuración básica**
- **Nombre de la instancia**: `mi-servidor`
- **AMI**: Elige una imagen del sistema operativo, por ejemplo:  
  - ✅ Amazon Linux 2  
  - 🐧 Ubuntu 20.04  
  - 🪟 Windows Server (si lo necesitas)
- **Tipo de instancia**:  
  - Elige `t2.micro` (gratis en el Free Tier)
- **Par de claves**:  
  - Crea una nueva si no tienes una (`mi-clave.pem`)  
  - Guárdala, ya que la necesitarás para conectarte vía SSH

#### 4. **Configuración de red**
- Selecciona o crea una VPC (la predeterminada está bien).
- En “Firewall” activa:
  - ✅ SSH (puerto 22) → para conectarte
  - ✅ HTTP (puerto 80) → si vas a correr una web

#### 5. **Almacenamiento**
- Usa el valor predeterminado (8 GB General Purpose SSD)

#### 6. **Revisar y lanzar**
Haz clic en **“Launch Instance”** y luego en **“View Instances”**.

### ✅ Ya tienes una instancia EC2 corriendo

Puedes conectarte por SSH así:

```bash
ssh -i mi-clave.pem ec2-user@<IP_PUBLICA>
```

(O `ubuntu@...` si usaste Ubuntu)

### Resumen

#### ¿Cómo crear una instancia de S2 en AWS?

Dominar los servicios en la nube de Amazon Web Services (AWS) es crucial para cualquier desarrollador. En esta lección, te guiaré a través de cada paso necesario para crear una instancia de S2 en AWS, un proceso esencial para desplegar aplicaciones. Al finalizar, tendrás una instancia configurada para que puedas acceder a ella y comenzar a implementar tus proyectos.

#### ¿Cómo iniciar sesión en la consola de AWS?

Primero, debes estar logueado en la consola de AWS. Si no lo has hecho aún, visita console.aws.amazon.com y utiliza tus credenciales para acceder. Una vez dentro, asegúrate de tener tu cuenta verificada y activa, incluyendo la verificación de correo electrónico e introducción de información financiera, como una tarjeta de crédito o débito. Es importante seleccionar el plan gratuito, el cual no generará cargos mientras uses los servicios dentro de los límites establecidos.

#### ¿Cómo navegar por la consola hacia S2?

- En la consola de AWS, busca "All Services" y asegúrate de que esté expandido.
- Localiza y selecciona S2, que probablemente encontrará minimizado.

#### ¿Cómo seleccionar las configuraciones de la instancia?

Al crear una instancia, AWS te permite personalizarla de varias formas. Aquí te comento algunas configuraciones esenciales:

- **Sistema Operativo**: Elige imágenes gratuitas para evitar cargos. Amazon ofrece sistemas como Amazon Linux, Ubuntu, y más.
- **Tipo de Instancia**: Opta por la T2 Micro, que es gratuita y tiene 1 CPU y 1 GB de RAM.
- **Almacenamiento**: Mantén el límite gratuito de 8 GB de almacenamiento.

#### ¿Cómo configurar la seguridad de tu instancia?

La seguridad de tu servidor es crucial:

- **Grupo de seguridad**: Crea un nuevo grupo y permite únicamente conexiones SSH en el puerto 22. Considera especificar IPs desde las cuales deseas permitir acceso para mayor seguridad. Si tienes una IP estática, puedes limitar el acceso a esas IPs específicas.

#### ¿Qué es una Elastic IP?

Entender el concepto de Elastic IP es fundamental si deseas que tu instancia tenga una IP fija. Esto es útil cuando necesitas que tu dirección IP permanezca constante, independientemente de la máquina o instancias que cambies dentro de AWS. La Elastic IP te permite asignar una IP fija a tu instancia o cambiarla manteniendo el mismo dominio DNS.

#### ¿Cómo garantizar el acceso a tu instancia?

- **Llave de Acceso**: Crea una nueva llave durante el proceso de configuración de la instancia. Descárgala, guárdala en un lugar seguro, y considera almacenarla en un USB o enviarla por correo electrónico para fácil acceso y seguridad.

Este archivo (.pem) es crucial para conexiones SSH posteriores, por lo que su seguridad debe ser prioritaria.

#### ¿Cómo iniciar tu instancia y monitorear su estado?

Una vez completada la configuración, inicia tu instancia. La consola muestra el estado de tu instancia, desde la inicialización hasta que esté lista para usarse. Aquí podrás:

- Verificar el estado.
- Copiar la IP pública para realizar conexiones futuras.

#### ¿Qué sigue después de la creación de la instancia?

Una vez activa, podrás realizar tareas como actualizar el sistema operativo, instalar herramientas de programación, o añadir tu editor de texto favorito. Estos pasos son críticos para asegurar que tu entorno de desarrollo esté configurado según tus necesidades.

Explorar y configurar instancias en AWS te abre un mundo de posibilidades para implementar y escalar aplicaciones. Con tiempo y práctica, manipularás con confianza entornos de nube complejos. ¡Continúa aprendiendo y experimentando con estas tecnologías para potenciar tus habilidades en la nube!

**Lecturas recomendadas**

[MobaXterm free Xserver and tabbed SSH client for Windows](https://mobaxterm.mobatek.net/)

## Conectándonos a nuestra instancia desde Windows

¡Perfecto! Si estás usando **Windows**, hay varias formas de conectarte a tu instancia EC2. Aquí te muestro **la forma más común y sencilla usando PuTTY**, un cliente SSH muy popular en Windows.

### 🔑 **1. Convertir tu archivo `.pem` a `.ppk` con PuTTYgen**

Amazon te da una clave privada `.pem`, pero **PuTTY necesita `.ppk`**.

1. Descarga [PuTTYgen](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) si no lo tienes.
2. Abre PuTTYgen.
3. Haz clic en **“Load”** y selecciona tu archivo `.pem` (asegúrate de cambiar el tipo de archivo a *All Files*).
4. Haz clic en **“Save private key”** (puedes ignorar el mensaje de contraseña si no deseas poner una).
5. Guarda el archivo `.ppk`.

### 💻 **2. Conectarte con PuTTY a la instancia EC2**

1. Abre **PuTTY**.
2. En **Host Name**, escribe:
   ```bash
   ec2-user@<IP-PUBLICA>
   ```
   o `ubuntu@<IP-PUBLICA>` si usas Ubuntu.
3. En el menú lateral, ve a:
   - `Connection` → `SSH` → `Auth`
   - Haz clic en **Browse** y carga el archivo `.ppk` que guardaste.
4. Opcional: guarda la sesión en `Session > Saved Sessions`.
5. Haz clic en **Open**.

o `ssh -i "xxxxx.pem" ec2-user@ec2-3-231-217-109.compute-1.amazonaws.com`

### 🛡️ ¿Problemas comunes?
- **Permiso denegado (publickey)**: Asegúrate de usar el usuario correcto (`ec2-user` para Amazon Linux, `ubuntu` para Ubuntu).
- **No se puede conectar**: Revisa el *Security Group* y asegúrate de que el puerto **22 esté abierto** para tu IP.

### Resumen

Sistemas UNIX como Mac y Linux traen por defecto una terminal predeterminada que funciona perfecto para conectarte por SSH (secure shell) a nuestras instancias en AWS. Sin embargo, en Windows no hay alternativas nativas que funcionen tan bien, así que puedes usar [MobaXterm](https://mobaxterm.mobatek.net/ "MobaXterm"), un software que te ayudara a conectarnos a nuestras instancias, ademas de muchos otros beneficios.

Recuerda que eres tienes las siguientes responsabilidades:

- **Updates**: Con las instancias, nosotros somos totalmente responsables de la actualización de OS
- **Respaldos**: Nuestra instancia no se respaldará sola, tendremos que hacerlo nosotros.
- **Restauración snapshot**: Podemos hacer respaldos antes de hacer grandes cambios para poder hacer rollback del Sistema en caso necesario.

**Lecturas recomendadas**

[MobaXterm free Xserver and tabbed SSH client for Windows](https://mobaxterm.mobatek.net/)

## Conectándonos a nuestra instancia desde Linux

¡Perfecto! Conectarse a una instancia EC2 desde **Linux** es aún más sencillo que desde Windows, ya que puedes usar directamente el comando `ssh` desde la terminal.

### 🧾 **Requisitos previos**
- Tener tu archivo de clave privada (`.pem`) descargado desde AWS.
- Tener permisos adecuados en el archivo `.pem`.
- Conocer la **IP pública** o **DNS público** de tu instancia EC2.
- Saber qué sistema operativo estás usando en la instancia:
  - `ec2-user` → Amazon Linux / CentOS
  - `ubuntu` → Ubuntu
  - `admin` → Debian
  - `root` → Instancias personalizadas

### 🛠️ **Pasos para conectarte desde Linux**

#### 1. Abre tu terminal

#### 2. Asegura los permisos del archivo `.pem`
```bash
chmod 400 ruta/a/tu/archivo.pem
```

Esto evita que otros usuarios puedan leer tu clave privada, algo que `ssh` requiere para permitir la conexión.

#### 3. Conéctate con el comando SSH

```bash
ssh -i ruta/a/tu/archivo.pem ec2-user@<ip-o-dns-público>
```

Ejemplo:
```bash
ssh -i ~/Downloads/mi-clave.pem ec2-user@54.123.45.67
```

#### ⚠️ Nota:
- Si estás usando **Ubuntu** en la instancia, cambia `ec2-user` por `ubuntu`.

### 💡 ¿Problemas comunes?

- **Permission denied (publickey)**: Asegúrate de estar usando el nombre de usuario correcto y que los permisos del archivo `.pem` estén en 400.
- **Connection timed out**: Verifica que el puerto **22 esté habilitado** en el *Security Group* de la instancia.

### Resumen

#### ¿Cómo conectarte a una instancia S2 a través de Dynux?

Cuando se trata de conectar con instancias en la nube, el proceso puede parecer un reto. Sin embargo, con los pasos correctos, podrás lograr una conexión efectiva y segura. En esta guía, aprenderás cómo conectarte a una instancia S2 utilizando Dynux, utilizando una distribución Parrot enfocada a la seguridad. Así que, ¡vamos a ello!

#### ¿Qué es una instancia S2 y por qué usar Dynux?

Primero, hablemos un poco sobre las herramientas que se usarán. Las instancias S2 son máquinas virtuales que se ejecutan en la infraestructura de Amazon Web Services (AWS). Utilizar Dynux, una distribución del sistema operativo basada en Linux enfocada en seguridad, te permite conectar a estas instancias de forma segura.

#### ¿Cómo acceder al dashboard de S2?

Antes que nada, debes acceder al dashboard de S2 para ver todas las instancias disponibles. Aquí puedes encontrar detalles críticos necesarios para tu conexión, como:

- **Dirección IP de la instancia**: Necesitarás copiar esta IP al portapapeles, ya que será fundamental para establecer la conexión.
- **Configuraciones adicionales**: Como cambiar el tamaño de la instancia usando opciones de "resize".

#### ¿Cómo preparar tu terminal para la conexión?

Ahora que tienes la IP de tu instancia, es momento de abrir la terminal. Dependiendo de tu sistema, puedes abrirla de diferentes maneras, como usando un clic directo o buscándola en aplicaciones.

#### Paso a paso para asegurar la conexión

1. **Ubicar la clave de acceso**: Asegúrate de saber dónde se encuentra tu archivo penfile o la clave que descargaste. En este caso, la clave se encuentra en la carpeta de descargas.

2. **Comando de conexión**: Usa el comando secure shell para establecer la conexión. A continuación, el formato a utilizar:

`ssh -i <nombre_de_la_llave> s2-user@<dirección_IP>`

 - Reemplaza <nombre_de_la_llave> con el nombre exacto de tu key.
 - Sustituye <dirección_IP> por la IP que copiaste desde el dashboard.
 
3. **Confirmar la adición de la IP**: Al conectar, se te pedirá que aceptes la nueva IP. Es crucial decir "sí" para almacenarla en el sistema.

4. **Acceso al sistema operativo de Amazon**: Una vez que estás conectado, serás el usuario s2-user. Puedes ejecutar comandos como root para mayor control.

#### ¿Dónde aprender más?

Con cada conexión, te acercas a mejorar tus habilidades en manejo de instancias y seguridad en la nube. La práctica continua es clave para profundizar en este tema.

En la próxima parte, exploraremos cómo realizar conexiones similares desde un dispositivo Mac, expandiendo así tus capacidades y asegurando que puedas operar en diferentes plataformas con eficacia. Continua explorando y educando, que cada paso es un acercamiento a la maestría.

¡Genial que preguntes por **Emacs**! Es uno de los editores de texto más poderosos y versátiles del mundo. Aquí te va una introducción clara:

### 🧠 ¿Qué es Emacs?

**Emacs** es un editor de texto **extensible, personalizable y multiplataforma**, ampliamente usado por programadores, científicos y entusiastas del software libre. Fue creado por **Richard Stallman** y forma parte del proyecto GNU.

- ✅ Puede editar cualquier tipo de texto: código, documentos, notas, correos, etc.
- ✅ Tiene su propio lenguaje de programación: **Emacs Lisp**.
- ✅ Es famoso por poder convertirse prácticamente en cualquier herramienta: agenda, terminal, navegador web, IDE, lector RSS, etc.

#### 🛠️ ¿Cómo instalar Emacs?

### En Linux (Debian/Ubuntu):
```bash
sudo apt install emacs
```

### En MacOS (con Homebrew):
```bash
brew install emacs
```

### En Windows:
- Puedes instalarlo desde [https://www.gnu.org/software/emacs/](https://www.gnu.org/software/emacs/)

#### 🚀 ¿Cómo se usa?

Al principio, puede parecer raro, pero aquí va lo básico para **no perderte**:

#### 👆 Comandos básicos:

| Acción                     | Comando en Emacs        | Qué significa |
|---------------------------|-------------------------|----------------|
| Salir                     | `C-x C-c`               | Ctrl+x, luego Ctrl+c |
| Guardar archivo           | `C-x C-s`               | Ctrl+x, luego Ctrl+s |
| Abrir archivo             | `C-x C-f`               | Ctrl+x, luego Ctrl+f |
| Copiar                    | `M-w`                   | Alt+w |
| Cortar                    | `C-w`                   | Ctrl+w |
| Pegar                     | `C-y`                   | Ctrl+y |
| Deshacer                  | `C-/` o `C-x u`         | Ctrl+/ o Ctrl+x, luego u |

> `C-` es **Ctrl**, `M-` es **Alt** (o Esc).

#### 🧩 Cosas locas que puedes hacer en Emacs

- 📚 **Organizar tu vida** con Org-mode.
- 🧠 Escribir y ejecutar código Python, C, LaTeX, Markdown, etc.
- 🛠️ Configurarlo como **IDE** con paquetes como `lsp-mode` o `eglot`.
- 🌐 Navegar por internet, ver correos o leer feeds RSS.
- 🎵 Incluso puedes escuchar música en Emacs. En serio.

#### 🔌 ¿Quieres mejorar tu Emacs?

Usa el sistema de paquetes de Emacs con `M-x package-install`.

Recomendados:
- `use-package` (gestión de configuración)
- `magit` (Git en Emacs)
- `projectile` (gestión de proyectos)
- `org` (agenda, notas, y más)
- `lsp-mode` (programación moderna)


## Conectándonos a nuestra instancia desde OSX

¡Perfecto! Si estás usando **macOS (OSX)** y quieres conectarte a tu **instancia EC2 en AWS**, te explico cómo hacerlo paso a paso. Solo necesitas tener instalada la terminal, que ya viene por defecto en macOS.


### 🧭 Requisitos previos

1. 🔑 Tienes tu archivo `.pem` (clave privada) descargado desde AWS al crear la instancia.
2. 🆔 Conoces la **IP pública** o **DNS público** de tu instancia.
3. 💻 Estás en un equipo macOS con **Terminal**.

### ✅ Pasos para conectarte a EC2 desde macOS

### 1. **Abre la Terminal**

Puedes buscar "Terminal" en Spotlight (`Cmd + Espacio` y escribe "Terminal").
### 2. **Dale permisos al archivo `.pem`**

Navega a la carpeta donde tengas tu archivo `.pem` y ejecuta:

```bash
chmod 400 nombre-clave.pem
```

Esto asegura que el archivo tenga los permisos correctos (muy importante).

### 3. **Conéctate vía SSH**

Usa el siguiente comando, reemplazando `nombre-clave.pem` por tu archivo y la IP/DNS de tu instancia:

```bash
ssh -i nombre-clave.pem ec2-user@<IP-o-DNS>
```

- Para instancias **Amazon Linux / RHEL**: `ec2-user@`
- Para **Ubuntu**: `ubuntu@`
- Para **Debian**: `admin@` o `debian@`

🔍 **Ejemplo**:

```bash
ssh -i mi-clave.pem ec2-user@ec2-18-222-123-456.compute-1.amazonaws.com
```

### 4. 🚀 ¡Listo!

Si todo está bien, estarás dentro de tu instancia EC2 desde macOS 🎉

### Resumen

#### ¿Cómo conectarse desde una Mac a una instancia de Amazon?

Conectarse a una instancia de Amazon desde tu Mac puede parecer complicado, pero con los pasos adecuados se convierte en un proceso sencillo y eficaz. Aquí vamos a guiarte a través del proceso, asegurándonos de que puedas establecer una conexión segura y efectiva con la instancia que creaste.

#### ¿Cómo acceder a la IP de tu instancia?

1. Abre tu navegador y dirígete a la sección de Compute S2.
2. Busca las instancias que se están ejecutando y selecciona la que acabas de crear.
3. Copia la dirección IP de la instancia. Esto es fundamental para conectarte desde la terminal de tu Mac.

#### ¿Cómo preparar tu terminal para la conexión?

La terminal es una herramienta esencial para ejecutar comandos. Puedes acceder a ella de la siguiente manera:

- Busca el ícono de la terminal a través de Spotlight. Simplemente escribe "terminal" y selecciona la aplicación.

Antes de conectar, asegúrate de estar en el directorio donde se encuentra la llave que creaste al configurar tu instancia. En este ejemplo, estamos en un directorio llamado Amazon, aunque es común encontrar la llave en el directorio "Descargas" o "Downloads".

#### ¿Cuál es el comando para conectarse?

Para conectar, utilizamos el siguiente comando en la terminal:

`ssh -i nombre_de_la_llave.pem ec2-user@IP_de_la_instancia`

- **-i**: Especifica el archivo de identidad (llave PEM) que usaste para configurar tu instancia.
- **ec2-user**: Es el nombre de usuario por defecto para instancias basadas en Amazon Linux.
- **@IP_de_la_instancia**: La dirección IP que copiaste anteriormente.

#### ¿Cómo solucionar problemas de permisos en la llave?

Al intentar conectar, puede que te encuentres con un problema de permisos en la llave. Para solucionarlo:

1. Cambia los permisos de la llave ejecutando:

`chmod 660 nombre_de_la_llave.pem`

2. Repite el comando `ssh` anterior para intentar la conexión nuevamente.

#### ¿Cómo convertirse en root en la instancia?

Una vez conectado, puede que necesites permisos de administrador (root) para instalar software o realizar configuraciones adicionales. Para ello, simplemente ejecuta:

`sudo su`

Ahora, puedes proceder a instalar cualquier programa necesario, como **IMAX** o cualquier otra herramienta que requieras.

Con estos pasos, ya estás preparado para conectarte con éxito a tu instancia desde una Mac. Este acceso te permite un control completo de tu máquina virtual y la capacidad de iniciar proyectos, entre ellos, proyectos en Python, que abordaremos en sesiones futuras. ¡Sigue explorando y aprendiendo, cada paso cuenta para dominar este entorno!

## Subiendo un proyecto: Clonando un repositorio de GitHub a nuestra Instancia de EC2

¡Perfecto! Aquí tienes una guía paso a paso para **clonar un repositorio de GitHub en tu instancia EC2**, una vez que ya estás conectado vía SSH. Esto es útil para subir tu proyecto, por ejemplo, una aplicación Node.js, Python, etc.


### 🛠️ Requisitos previos

- Ya estás conectado a tu instancia EC2 desde macOS, Windows o Linux.
- Tu instancia tiene acceso a internet (vía una **IP pública** o saliendo por un **NAT Gateway** si está en una VPC).
- El repositorio en GitHub es **público** o **tienes una forma de autenticarte** (token o SSH key) si es privado.


### 🧩 Paso a paso

### 1. 📥 Instalar Git (si no está instalado)

Ejecuta en tu instancia:

```bash
sudo yum update -y            # Amazon Linux / RHEL
sudo yum install git -y
```

O para Ubuntu:

```bash
sudo apt update
sudo apt install git -y
```

### 2. 🔎 Verifica que Git esté disponible

```bash
git --version
```

Deberías ver algo como: `git version 2.4.11` o superior.
### 3. 📂 Elige un directorio para tu proyecto

Por ejemplo:

```bash
cd /home/ec2-user
mkdir proyectos
cd proyectos
```

### 4. 🧬 Clonar el repositorio

Para repositorios públicos:

```bash
git clone https://github.com/usuario/nombre-repo.git
```

Para repositorios privados (vía HTTPS):

```bash
git clone https://<token>@github.com/usuario/nombre-repo.git
```

(El `<token>` es un [Personal Access Token](https://github.com/settings/tokens))

### 5. 📁 Navega al proyecto

```bash
cd nombre-repo
ls
```

## ✅ ¡Listo!

Tu proyecto está ahora dentro de la instancia EC2. Puedes instalar dependencias, iniciar servidores, compilar código, etc.

### Resumen

Para clonar repositorios desde tu instancia de AWS necesitas instalar [git](https://git-scm.com/ "git"):

```bash
# Permisos de super usuario
sudo su
# Instalación de git
yum install git
```

Teniendo git instalado ya podemos clonar el repositorio que necesitemos:

`git clone [URL_DEL_PROYECTO]`

En el siguiente link vas a encontrar el repositorio con el que vamos a trabajar en la clase: [https://github.com/mauropm/aws-platzi-python](https://github.com/mauropm/aws-platzi-python "https://github.com/mauropm/aws-platzi-python").

## Subiendo un proyecto a nuestra instancia de EC2: Ejecutar nuestro proyecto

¡Claro! Una vez que ya **clonaste tu proyecto en la instancia EC2**, estás listo para **ejecutarlo**. Aquí te muestro cómo hacerlo dependiendo del tipo de aplicación que tengas (Node.js, Python, etc.).

### 🚀 1. Accede a tu proyecto clonado

```bash
cd /home/ec2-user/proyectos/nombre-repo
```

Asegúrate de estar en el directorio correcto:

```bash
ls
```

### 🧪 2. Instala las dependencias

### Si es un proyecto **Node.js**:
```bash
npm install
```

### Si es un proyecto **Python**:
Primero crea un entorno virtual (opcional pero recomendado):

```bash
python3 -m venv env
source env/bin/activate
```

Luego instala las dependencias:
```bash
pip install -r requirements.txt
```

### 🟢 3. Ejecuta tu aplicación

### Para **Node.js**:
```bash
npm start
```
O si usas un archivo específico (por ejemplo `index.js`):
```bash
node index.js
```

### Para **Python**:
```bash
python main.py
```
(Sustituye `main.py` con el nombre de tu script principal)

### 🌐 4. Habilita acceso externo (si tu app es web)

Abre el puerto necesario en el **Security Group** de tu instancia EC2. Ejemplo:

- **Puerto 80** para HTTP
- **Puerto 5000, 3000, 8000** para apps web locales

Ve a la consola de AWS → EC2 → Instancias → Grupo de seguridad → Reglas de entrada → Editar → Agrega el puerto.

### 🧩 5. Ejecutar en segundo plano (opcional)

Si no quieres que la app se detenga cuando cierres la consola SSH, puedes usar:

```bash
nohup node index.js > output.log 2>&1 &
```

O para Python:

```bash
nohup python main.py > output.log 2>&1 &
```

### Resumen

Ahora que clonamos nuestro repositorio, vamos a instalar las dependencias necesarias para que funcione. El proyecto que trabajamos en clase sólo necesita una dependencia:

```bash
# Permisos de super usuario
sudo su
# Instalación de flask
pip install flask
```

Con las dependencias instaladas podemos correr nuestro servidor, en el caso de nuestro proyecto es con el siguiente comando:

`python app.py`

Para acceder por internet a nuestro servidor tenemos que buscar nuestra instancia de EC2 y copiar la IP pública, pero no es suficiente. Debemos entrar a `Security Groups`, `Inbound` y añadir una regla `Custom TCP Rule`, escribe el puerto en el que corre tu servidor (que para nuestro proyecto es el 5000) y en la opción *Source elije Anywhere*.

Recuerda que para encontrar tu IP pública puedes entrar a [whatismyip.com](http://whatismyip.com/ "whatismyip.com").

**Lecturas recomendadas**

[What Is My IP? Shows your real IP - IPv4 - IPv6 - WhatIsMyIP.com®](https://www.whatismyip.com/)

## ¿Qué es Lambda y Serverless?

¡Buena pregunta! Vamos por partes y con ejemplos simples 👇

### 🚀 **¿Qué es AWS Lambda?**

**AWS Lambda** es un servicio de computación **Serverless** de Amazon que te permite ejecutar código **sin tener que preocuparte por servidores**. Solo escribes tu función, la subes, y Lambda se encarga del resto:

- No tienes que crear ni administrar servidores.
- Solo pagas por el tiempo que tu código se ejecuta (¡por milisegundos!).
- Escala automáticamente, desde 1 llamada hasta millones.

### Ejemplo:
Supón que subes una imagen a un bucket S3, y quieres generar una miniatura automáticamente.

✅ Puedes conectar ese evento a una función Lambda que:
- Reciba la imagen
- Cree la miniatura
- La suba a otro bucket

### 🌩️ **¿Qué es Serverless?**

"Serverless" (sin servidor) **no significa que no haya servidores**, sino que tú **no tienes que administrarlos**.

### Ventajas del enfoque Serverless:
- ⛅ **Escala automáticamente**
- 💸 **Pago por uso** (cero uso = cero pago)
- 🧼 **Sin mantenimiento de servidores**
- ⚡ **Implementación rápida y modular**

### 🎯 ¿Dónde encaja Lambda en Serverless?

Lambda es **una de las piezas clave** del ecosistema Serverless de AWS. Otras herramientas serverless que suelen trabajar con Lambda:

- **API Gateway** (para exponer funciones como APIs REST/HTTP)
- **S3** (para subir archivos que pueden disparar Lambdas)
- **SQS/SNS** (mensajería asincrónica entre funciones)
- **DynamoDB** (base de datos serverless)

### Resumen

Lambda es un proyecto de AWS muy relacionado con el concepto de [Serverless](https://platzi.com/blog/serverless-el-inicio-de-una-revolucion/ "Serverless"), dejar la administración de tus servidores en manos de Amazon para solo encargarte de las funciones de código que ejecutara tu aplicación.

#### ¿Qué son?

Imagina lambda como un lugar donde puedes ejecutar funciones de tu código.

##### Serverless

No existe un servidor como vimos en EC2, es decir, solo está el código en lamba y AWS se encarga de ejecutarlo cuando necesites.

Profundiza sobre [qué es serverless framework](https://platzi.com/blog/introduccion-a-serverless-framework/ "qué es serverless framework").

##### Lenguajes soportados

Puedes programar funciones lamba en Nodejs (JavaScript), Python, Java (8), C# (.Net Core) y Go.

Recuerda tener en cuenta los siguientes puntos:

- **Memoria**: Mínima de 128MB, máxima 3000MB con incrementos de 64MB.
- **Límites de ejecución y espacio**: Puedes correr tu aplicación hasta 300 segundos y tienes un /tmp limitado a 512MB.
- **Ejecución paralela**: Esta limitada a 1000 ejecuciones concurrentes (a un mismo tiempo), no tiene límite en ejecuciones secuenciales (una detrás de otra).

#### Ventajas de Lambda:

- **Seguridad**: Al ser una infraestructura compartida, no tienes que preocuparte de seguridad: AWS maneja todo.
- **Performance**: AWS está monitoreando constantemente la ejecución de tus funciones y se encarga de que siempre tenga el mejor performance.
- **Código aislado**: Tu código, aún estando en una infraestructura compartida, corre en un ambiente virtual exclusivo, aislado de las demás ejecuciones lamba.

Recuerda que AWS te regala 1 millón de peticiones lamba gratis el primer año.

## Creación de Funciones Lambda en Python para AWS

¡Claro! Crear funciones Lambda en Python para AWS es un proceso bastante directo y muy útil para construir aplicaciones serverless. Aquí te explico los pasos principales y te doy un ejemplo:

**Pasos para Crear una Función Lambda en Python:**

1.  **Escribir el Código de la Función Lambda:**
    * Tu código Python debe incluir una función "handler". Este es el punto de entrada que AWS Lambda ejecutará cuando se invoque tu función.
    * La función handler generalmente toma dos argumentos: `event` y `context`.
        * `event`: Un diccionario que contiene los datos de entrada para la función Lambda. El formato de este diccionario dependerá del servicio de AWS que active la función (por ejemplo, S3, API Gateway, etc.).
        * `context`: Un objeto que proporciona información sobre la invocación, la función y el entorno de ejecución.
    * Tu código puede importar otras bibliotecas estándar de Python o las que incluyas en tu paquete de despliegue.

2.  **Crear un Paquete de Despliegue (Opcional pero Común):**
    * Si tu función Lambda utiliza bibliotecas que no están incluidas en el entorno de ejecución de Lambda (como `requests`, `pandas`, etc.), necesitarás crear un paquete de despliegue. Este es un archivo ZIP que contiene tu código Python y las dependencias necesarias.
    * Para crear el paquete de despliegue, puedes usar `pip install -t ./package <nombre_de_la_biblioteca>` en un directorio local llamado `package`, y luego comprimir el contenido de ese directorio.

3.  **Crear la Función Lambda en la Consola de AWS o con la AWS CLI:**

    * **Usando la Consola de AWS:**
        * Inicia sesión en la consola de AWS y ve al servicio Lambda.
        * Haz clic en "Crear función".
        * Elige "Crear desde cero".
        * Configura los siguientes parámetros:
            * **Nombre de la función:** Dale un nombre descriptivo a tu función Lambda.
            * **Tiempo de ejecución:** Selecciona "Python 3.x" (elige la versión que corresponda a tu código).
            * **Arquitectura:** Selecciona la arquitectura adecuada (generalmente x86\_64).
            * **Permisos:** Configura el rol de ejecución. Puedes crear un nuevo rol con permisos básicos de Lambda o seleccionar un rol existente que tenga los permisos necesarios para que tu función interactúe con otros servicios de AWS.
        * En la sección "Código fuente", puedes:
            * Cargar un archivo .zip (si creaste un paquete de despliegue).
            * Pegar el código directamente en el editor en línea (para funciones sencillas sin dependencias externas).
        * Configura otras opciones como la memoria asignada, el tiempo de espera, las variables de entorno, etc., según tus necesidades.
        * Haz clic en "Crear función".

    * **Usando la AWS CLI:**
        * Asegúrate de tener la AWS CLI instalada y configurada con tus credenciales de AWS.
        * Crea un archivo ZIP con tu código (y dependencias si las hay).
        * Utiliza el comando `aws lambda create-function`:

        ```bash
        aws lambda create-function \
            --function-name mi-funcion-lambda \
            --runtime python3.9 \
            --zip-file fileb://mi_paquete.zip \
            --handler mi_script.handler \
            --role arn:aws:iam::123456789012:role/mi-rol-lambda \
            --memory-size 128 \
            --timeout 30
        ```

        Reemplaza los siguientes valores:
        * `mi-funcion-lambda`: El nombre que quieres darle a tu función.
        * `python3.9`: El tiempo de ejecución de Python.
        * `mi_paquete.zip`: La ruta a tu archivo ZIP de despliegue.
        * `mi_script.handler`: El nombre del archivo Python y el nombre de la función handler (por ejemplo, si tu archivo es `lambda_function.py` y tu función es `mi_handler`, sería `lambda_function.mi_handler`).
        * `arn:aws:iam::...:role/mi-rol-lambda`: El ARN del rol de IAM que tiene los permisos necesarios.
        * `128`: La cantidad de memoria en MB.
        * `30`: El tiempo de espera en segundos.

4.  **Configurar Triggers (Desencadenadores):**
    * Una vez creada la función Lambda, necesitas configurar qué evento o servicio de AWS la invocará. Esto se hace a través de los "Triggers" en la consola de AWS o mediante la AWS CLI.
    * Los triggers pueden ser servicios como API Gateway, S3, DynamoDB, CloudWatch Events (EventBridge), SNS, SQS, etc.
    * La configuración del trigger dependerá del servicio que elijas. Por ejemplo, para un trigger de API Gateway, definirás las rutas y métodos HTTP. Para un trigger de S3, especificarás el bucket y los eventos (como la creación de un objeto).

5.  **Probar la Función Lambda:**
    * La consola de AWS proporciona una interfaz para probar tu función Lambda. Puedes proporcionar un evento de prueba (en formato JSON) para simular una invocación.
    * Revisa los logs de ejecución en CloudWatch Logs para verificar si la función se ejecutó correctamente y si hubo algún error.

**Ejemplo Sencillo de Función Lambda en Python (sin dependencias externas):**

Supongamos que quieres crear una función Lambda que tome un nombre como entrada y devuelva un saludo.

**`lambda_function.py`:**

```python
import json

def handler(event, context):
    """
    Esta función Lambda recibe un evento con un 'nombre' y devuelve un saludo.
    """
    nombre = event.get('nombre', 'Mundo')
    mensaje = f"¡Hola, {nombre} desde AWS Lambda!"

    return {
        'statusCode': 200,
        'body': json.dumps({'mensaje': mensaje})
    }
```

**Pasos para desplegar este ejemplo usando la consola de AWS:**

1.  Copia y pega el código anterior directamente en el editor en línea al crear la función Lambda.
2.  Dale un nombre a tu función (por ejemplo, `saludar`).
3.  Selecciona un tiempo de ejecución de Python (por ejemplo, `Python 3.9`).
4.  Crea un rol de ejecución básico de Lambda.
5.  Haz clic en "Crear función".
6.  Para probar la función, haz clic en el botón "Probar".
7.  Crea un nuevo evento de prueba con un JSON como este:

    ```json
    {
      "nombre": "TuNombre"
    }
    ```

8.  Haz clic en "Probar". Deberías ver una respuesta con un código de estado 200 y un cuerpo JSON que contiene el saludo.

**Consideraciones Importantes:**

* **Roles de IAM:** Asegúrate de que el rol de ejecución de tu función Lambda tenga los permisos necesarios para acceder a otros servicios de AWS que pueda necesitar (por ejemplo, leer de S3, escribir en DynamoDB, etc.).
* **Variables de Entorno:** Puedes usar variables de entorno para configurar tu función sin necesidad de hardcodear valores sensibles o que puedan cambiar.
* **Logging:** Utiliza la biblioteca `logging` de Python para escribir logs que se enviarán a CloudWatch Logs. Esto es fundamental para la depuración y el monitoreo.
* **Manejo de Errores:** Implementa un manejo de errores robusto en tu código Lambda para evitar fallos inesperados.
* **Rendimiento y Costos:** Considera la memoria asignada y el tiempo de espera de tu función, ya que estos factores afectan tanto el rendimiento como los costos.
* **Serverless Framework o AWS SAM:** Para proyectos más complejos, considera usar herramientas como Serverless Framework o AWS SAM (Serverless Application Model) para definir y desplegar tu infraestructura serverless de manera más organizada y automatizada.

Espero que esta explicación te sea útil para empezar a crear tus propias funciones Lambda en Python para AWS. ¡No dudes en preguntar si tienes alguna otra duda!

### Resumen

#### ¿Cómo crear una función Lambda en AWS utilizando Python?

El creciente ecosistema de Amazon Web Services (AWS) se presenta como una solución poderosa para desarrolladores que buscan optimizar procesos y automatizar tareas. Un componente esencial de este ecosistema es AWS Lambda, que permite ejecutar código sin necesidad de gestionar servidores, gracias al enfoque serverless. En este tutorial, exploraremos cómo crear una función Lambda utilizando Python, desde la consola de AWS.

#### ¿Qué es AWS Lambda?

AWS Lambda es un servicio de computación que permite ejecutar código sin la necesidad de aprovisionar o administrar servidores. Funciona mediante el modelo de ejecución bajo demanda, permitiendo a los desarrolladores cargar y ejecutar código solo cuando sea necesario, lo que convierte a Lambda en una opción muy eficiente y coste-efectiva para diversas aplicaciones.

#### ¿Cómo iniciar con AWS Lambda?

1. **Acceder a la consola de AWS**: Dirígete a la consola de AWS y asegúrate de estar en la sección correcta:

 - Ve a "All Services" (Todos los servicios).
 - Busca y selecciona "Lambda" bajo la categoría de "Compute" (Cómputo).

2. **Crear una nueva función Lambda**:

 - Selecciona "Crear una función" y elige "From scratch" (Desde cero).
 - Asigna un nombre a tu función, por ejemplo, latsifond.
 - Elige Python 3.6 como el lenguaje de ejecución, aunque AWS Lambda soporta varios lenguajes como C, Java, y Go.
 
3. **Configurar el rol de ejecución**: Define un rol que permita a tu función Lambda ejecutar de forma segura. Esta configuración es crucial para garantizar que tu función tenga los permisos necesarios para interactuar con otros servicios de AWS.

#### ¿Cómo escribir y desplegar el código de una función Lambda en AWS?

Desarrollar una función en AWS Lambda implica definir una función de controlador, conocida como `lambda_handler`, que recibe dos argumentos: `event` y `context`.

```python
def lambda_handler(event, context):
    # Código para manejar el evento
    what_to_print = event['what_to_print']
    how_many_times = event['how_many_times']

    if how_many_times > 0:
        for _ in range(how_many_times):
            print(what_to_print)
```

- `what_to_print` y `how_many_times`: Estas son variables de ambiente que determinan qué se imprime y cuántas veces.
- El código comprueba una condición simple y ejecuta acciones según los parámetros recibidos.

#### ¿Cómo configurar las variables de entorno y parámetros adicionales?

1. **Variables de ambiente**: En AWS Lambda, puedes establecer variables de ambiente que serán accesibles para tu función sin necesidad de codificarlas directamente. Ejemplo:

 - `what_to_print` = "Hola desde Platzi"
 - `how_many_times` = 6
 
2. **Configurar la memoria y concurrencia**:

 - **Memoria**: Puedes ajustar la memoria RAM dedicada a tu función, que por defecto es 128 MB y puede expandirse hasta 3 GB. AWS Lambda ajusta automáticamente la memoria basada en el uso histórico.
- **Concurrencia**: AWS Lambda permite hasta 1000 ejecuciones concurrentes por defecto. Si necesitas más, puedes contactar el soporte de AWS para analizar tus necesidades.

#### ¿Cómo probar y validar las funciones Lambda?

Una vez configurada tu función, es vital asegurarse de que funcione correctamente creando eventos de prueba y ejecutando la función. Después de la ejecución, AWS Lambda proporciona información detallada sobre el rendimiento, incluidos el tiempo de ejecución y la memoria utilizada.

1. **Crear un evento de prueba**:

 - Crea un nuevo evento de prueba, nómbralo adecuadamente, como hola.
 - Ejecuta el evento para verificar la salida y el rendimiento de tu función.

2. **Evaluar los resultados**: Asegúrate de revisar el área superior de la consola para ver los resultados de ejecución, que indican la duración, uso de memoria y otras métricas vitales.

#### Consejos adicionales y resolución de problemas

- **Atención al detalle**: Los resultados de las ejecuciones suelen aparecer al tope en la consola, lo que puede ser confuso al principio. Haz scroll hacia arriba para confirmar que los resultados se muestran correctamente.
- **Optimización**: Observa el uso de recursos y ajusta la configuración en función de tus necesidades reales.
- **Feedback y soporte**: Si tienes dudas, contacta con AWS Support para recibir orientación y ajusta los parámetros según el feedback recibido.

AWS Lambda y Python proporcionan una combinación poderosa para automatizar procesos y construir aplicaciones eficientes en la nube. Este tutorial es solo un comienzo; experimenta y adapta estas herramientas para satisfacer tus objetivos de desarrollo.

[github](https://github.com/mauropm/aws-platzi-python)

## Escalabilidad Dinámica en Amazon Elastic Beanstalk

¡Absolutamente! La escalabilidad dinámica es una característica fundamental de Amazon Elastic Beanstalk que te permite ajustar automáticamente la capacidad de tu entorno para mantener un rendimiento constante y optimizar los costos. Aquí te explico cómo funciona y cómo configurarla:

**¿Qué es la Escalabilidad Dinámica en Elastic Beanstalk?**

La escalabilidad dinámica (también conocida como Auto Scaling) en Elastic Beanstalk se refiere a la capacidad de tu entorno de agregar o eliminar automáticamente instancias de Amazon EC2 en función de las métricas de rendimiento de tu aplicación. Esto asegura que tengas suficientes recursos para manejar los picos de tráfico y que no pagues por capacidad inactiva durante los períodos de baja demanda.

**Componentes Clave de la Escalabilidad Dinámica en Elastic Beanstalk:**

1.  **Grupo de Auto Scaling (Auto Scaling Group - ASG):** Elastic Beanstalk crea y gestiona un grupo de Auto Scaling para tu entorno con balanceo de carga. Este grupo es el encargado de lanzar y terminar instancias EC2 según las políticas de escalado que definas. En un entorno de instancia única, el ASG asegura que siempre haya una instancia en ejecución.

2.  **Políticas de Escalado (Scaling Policies):** Estas políticas definen las condiciones bajo las cuales el ASG debe escalar (aumentar o disminuir) el número de instancias. Puedes basar tus políticas en varias métricas de Amazon CloudWatch.

3.  **Disparadores (Triggers) o Alarmas de CloudWatch:** Las políticas de escalado se activan cuando una métrica de CloudWatch cruza un umbral definido durante un período específico. Elastic Beanstalk utiliza alarmas de CloudWatch para monitorear estas métricas y activar las acciones de escalado.

**Tipos de Políticas de Escalado Dinámico:**

Elastic Beanstalk te permite configurar políticas de escalado basadas en:

* **Métricas de Utilización:**
    * **CPU Utilization:** Escala en función del porcentaje promedio de uso de la CPU en tus instancias.
    * **Network In/Out:** Escala según la cantidad promedio de tráfico de red entrante o saliente por instancia.
    * **Disk I/O:** Escala basado en las operaciones de lectura/escritura en disco por instancia.
    * **Memory Utilization (requiere configuración adicional del agente de CloudWatch):** Escala según el porcentaje de memoria utilizada en tus instancias.
    * **Request Count:** Escala según el número de solicitudes HTTP completadas por el balanceador de carga por instancia.
    * **Latency:** Escala basado en la latencia promedio de las solicitudes atendidas por el balanceador de carga.
* **Métricas Personalizadas:** Puedes definir tus propias métricas en CloudWatch y utilizarlas para escalar tu entorno.

**Configuración de la Escalabilidad Dinámica en Elastic Beanstalk:**

Puedes configurar la escalabilidad dinámica de tu entorno de Elastic Beanstalk de varias maneras:

1.  **Consola de AWS:**
    * Ve al servicio Elastic Beanstalk en la consola de AWS.
    * Selecciona tu entorno.
    * En el panel de navegación, elige "Configuración".
    * En la categoría "Capacidad", haz clic en "Editar".
    * Aquí podrás configurar:
        * **Rango de Instancias:** El número mínimo y máximo de instancias que tu entorno puede tener.
        * **Disparadores de Escalado:** Define las métricas, los umbrales, la duración de la infracción y el número de instancias que se deben agregar o eliminar.
        * **Opciones de Escalado Adicionales:** Como el tiempo de espera de enfriamiento (cooldown) entre las operaciones de escalado.

2.  **AWS CLI o EB CLI:**
    * Puedes usar los comandos de la AWS CLI o la EB CLI para crear o actualizar la configuración de Auto Scaling de tu entorno. Por ejemplo, con la AWS CLI, puedes usar el comando `aws elasticbeanstalk update-environment` con las opciones de configuración del espacio de nombres `aws:autoscaling:asg` y `aws:autoscaling:trigger`.

3.  **.ebextensions:**
    * Puedes definir la configuración de Auto Scaling en archivos de configuración dentro del directorio `.ebextensions` de tu paquete de código fuente. Esto te permite versionar la configuración de tu infraestructura junto con tu aplicación.

**Ejemplo de Configuración con `.ebextensions`:**

Crea un archivo llamado `autoscaling.config` dentro de `.ebextensions`:

```yaml
option_settings:
  aws:autoscaling:asg:
    MinSize: 2
    MaxSize: 5
  aws:autoscaling:trigger:
    MeasureName: CPUUtilization
    Statistic: Average
    Unit: Percent
    Period: 60
    BreachDuration: 5
    UpperThreshold: 70
    LowerThreshold: 30
    ScaleUpIncrement: 1
    ScaleDownIncrement: 1
```

Este ejemplo configura un grupo de Auto Scaling con un mínimo de 2 instancias y un máximo de 5. También define una política de escalado que agrega una instancia si la utilización promedio de la CPU supera el 70% durante 5 minutos, y elimina una instancia si cae por debajo del 30% durante 5 minutos.

**Mejores Prácticas para la Escalabilidad Dinámica:**

* **Elige las Métricas Correctas:** Selecciona las métricas que mejor reflejen la carga de trabajo de tu aplicación. La utilización de la CPU suele ser un buen punto de partida, pero considera otras métricas como la latencia o el número de solicitudes para aplicaciones con alta carga de E/S o muchas solicitudes.
* **Configura Umbrales Adecuados:** Establece umbrales que permitan a tu entorno escalar antes de que los usuarios experimenten problemas de rendimiento, pero evita umbrales demasiado sensibles que puedan resultar en un escalado innecesario.
* **Considera el Tiempo de Enfriamiento (Cooldown):** Configura un tiempo de enfriamiento adecuado para evitar que el Auto Scaling reaccione de forma exagerada a picos de carga transitorios.
* **Prueba tu Configuración de Escalado:** Realiza pruebas de carga para asegurarte de que tu configuración de escalado funciona como se espera y que tu aplicación puede manejar el aumento de instancias.
* **Monitorea tu Escalado:** Utiliza CloudWatch para monitorear las métricas de tu aplicación y las actividades de escalado del Auto Scaling para realizar ajustes si es necesario.
* **Considera el Escalado Basado en la Programación (Scheduled Scaling):** Para cargas de trabajo predecibles (por ejemplo, picos de tráfico a ciertas horas del día), puedes combinar el escalado dinámico con el escalado basado en la programación para asegurar que tienes la capacidad adecuada antes de que ocurra el aumento de tráfico.

La escalabilidad dinámica en Elastic Beanstalk es una herramienta poderosa para construir aplicaciones resilientes y rentables en la nube de AWS. Configurarla correctamente te permitirá manejar el crecimiento y las fluctuaciones de tráfico de manera eficiente.

### Resumen

Elastic Beanstalk es una plataforma donde en pocos pasos, obtienes un balanceador de cargas y tantas instancias EC2 como tu quieras.

Este ambiente puede escalar de manera dinámica de acuerdo al tiempo de respuesta a los usuarios, uso de CPU, uso de RAM, etc.

Esta herramienta soporta los siguientes ambientes:

- Docker Image
- Go
- Java SE
- Java con Tomcat
- .NET + Windows Server + IIS
- Nodejs
- PHP
- Python
- Ruby

## Almacenamiento y Gestión de Archivos con Amazon S3

¡Excelente tema! Amazon Simple Storage Service (S3) es un servicio de almacenamiento de objetos altamente escalable, seguro, duradero y de alto rendimiento ofrecido por AWS. Es fundamental para muchas arquitecturas en la nube debido a su versatilidad y bajo costo. Aquí te explico los aspectos clave del almacenamiento y la gestión de archivos con S3:

**Conceptos Fundamentales de Amazon S3:**

* **Buckets (Contenedores):** Son los contenedores de nivel superior para almacenar tus objetos (archivos). Los buckets tienen un nombre único a nivel global dentro de la región de AWS donde se crean. Piensa en ellos como carpetas de alto nivel.
* **Objects (Objetos):** Son los archivos que almacenas en los buckets. Cada objeto tiene una clave (su nombre dentro del bucket) y, opcionalmente, metadatos.
* **Claves (Keys):** Son los nombres únicos de los objetos dentro de un bucket. La clave identifica el objeto. Puedes pensar en las claves como las rutas de archivo dentro de una estructura de carpetas (aunque S3 es un almacenamiento de objetos plano, la clave puede simular una estructura jerárquica usando prefijos).
* **Regiones:** Los buckets se crean en una región específica de AWS. Elegir la región correcta es importante por razones de latencia, costos y cumplimiento normativo.

**Almacenamiento de Archivos con S3:**

1.  **Creación de Buckets:**
    * Puedes crear buckets a través de la consola de AWS, la AWS CLI, los SDKs de AWS o herramientas de infraestructura como código (IaC) como AWS CloudFormation o Terraform.
    * Al crear un bucket, debes elegir un nombre único y una región de AWS.

2.  **Subida de Objetos (Archivos):**
    * Puedes subir archivos a tus buckets utilizando la consola de AWS (arrastrar y soltar o seleccionar archivos), la AWS CLI (`aws s3 cp`, `aws s3 sync`), los SDKs de AWS (para integrar la funcionalidad en tus aplicaciones) o herramientas de terceros.
    * Al subir un objeto, puedes especificar su clave, metadatos (pares clave-valor que describen el objeto) y la clase de almacenamiento.

3.  **Clases de Almacenamiento:** S3 ofrece varias clases de almacenamiento optimizadas para diferentes casos de uso y patrones de acceso, con diferentes costos y niveles de disponibilidad y durabilidad:
    * **S3 Standard:** Para acceso frecuente, alta disponibilidad y durabilidad. Es la clase predeterminada.
    * **S3 Intelligent-Tiering:** Mueve automáticamente los datos entre niveles de acceso frecuente, infrecuente y de archivo en función de los patrones de acceso cambiantes, optimizando los costos.
    * **S3 Standard-Infrequent Access (S3 Standard-IA):** Para datos a los que se accede con menos frecuencia pero que requieren una disponibilidad y un rendimiento similares a los de S3 Standard. Tiene un costo de almacenamiento más bajo pero un costo de recuperación más alto.
    * **S3 One Zone-Infrequent Access (S3 One Zone-IA):** Similar a S3 Standard-IA pero almacena los datos en una única zona de disponibilidad, lo que reduce los costos pero también la disponibilidad y durabilidad. No se recomienda para datos críticos.
    * **S3 Glacier Instant Retrieval:** Para datos archivados a los que se accede ocasionalmente con requisitos de recuperación en milisegundos.
    * **S3 Glacier Flexible Retrieval (anteriormente S3 Glacier):** Para archivado a largo plazo con opciones de recuperación flexibles que van desde minutos hasta horas. Costo de almacenamiento muy bajo.
    * **S3 Glacier Deep Archive:** La clase de almacenamiento de menor costo, diseñada para el archivado de datos a largo plazo a los que se accede muy raramente. Los tiempos de recuperación son de horas.

**Gestión de Archivos con S3:**

1.  **Organización:**
    * Aunque S3 no tiene una estructura de carpetas tradicional, puedes usar prefijos en las claves de los objetos para simular una jerarquía. Por ejemplo, las claves `fotos/2023/enero/imagen1.jpg` y `fotos/2023/febrero/imagen2.jpg` crean una organización lógica.
    * La consola de AWS y los SDKs a menudo interpretan estos prefijos como carpetas para facilitar la navegación.
    * **S3 Object Tags:** Puedes asignar etiquetas (pares clave-valor) a los objetos para categorizarlos y administrarlos. Las etiquetas pueden usarse para políticas de control de acceso y administración del ciclo de vida.

2.  **Control de Acceso:** S3 ofrece varios mecanismos para controlar el acceso a tus buckets y objetos:
    * **Políticas de Bucket:** Permiten definir reglas de acceso a nivel de bucket, especificando qué principios (usuarios, cuentas de AWS, servicios) tienen qué permisos (lectura, escritura, eliminación, etc.) y bajo qué condiciones. Se escriben en formato JSON.
    * **Listas de Control de Acceso (ACLs):** Un mecanismo más antiguo que permite conceder permisos básicos (lectura, escritura, control total) a usuarios y grupos de AWS a nivel de bucket y objeto. Se recomienda usar políticas de bucket en lugar de ACLs para un control de acceso más granular.
    * **Políticas de IAM (Identity and Access Management):** Puedes crear roles y usuarios de IAM con políticas que les otorguen permisos específicos para interactuar con buckets y objetos de S3. Es la forma más recomendada de gestionar permisos para usuarios y aplicaciones.
    * **AWS KMS Encryption:** Puedes cifrar tus objetos en reposo utilizando claves administradas por S3 (SSE-S3), claves administradas por AWS KMS (SSE-KMS) o claves proporcionadas por el cliente (SSE-C). También puedes habilitar el cifrado en tránsito mediante HTTPS.
    * **Bucket Policies y CORS (Cross-Origin Resource Sharing):** Si tu aplicación web necesita acceder a recursos en un bucket de S3 desde un dominio diferente, deberás configurar el CORS en el bucket.

3.  **Administración del Ciclo de Vida (Lifecycle Management):**
    * Las políticas de ciclo de vida te permiten automatizar la transición de objetos entre diferentes clases de almacenamiento o su eliminación después de un período de tiempo especificado. Esto es crucial para optimizar costos y cumplir con políticas de retención.
    * Puedes definir reglas basadas en prefijos de clave, etiquetas de objeto o la antigüedad del objeto. Por ejemplo, puedes mover automáticamente los objetos de S3 Standard a S3 Standard-IA después de 30 días y luego a S3 Glacier después de un año.

4.  **Versioning:**
    * Habilitar el versionamiento en un bucket conserva todas las versiones de un objeto, incluso si se sobrescriben o eliminan. Esto proporciona una capa adicional de protección contra la pérdida de datos y permite restaurar versiones anteriores de un objeto.

5.  **Replicación (Cross-Region Replication - CRR y Same-Region Replication - SRR):**
    * La replicación te permite copiar automáticamente objetos entre buckets en diferentes regiones (CRR) o en la misma región (SRR). Esto puede ser útil para recuperación ante desastres, cumplimiento normativo o acceso de baja latencia en diferentes ubicaciones geográficas.

6.  **Notificaciones de Eventos (S3 Event Notifications):**
    * Puedes configurar notificaciones para que S3 envíe mensajes a otros servicios de AWS (como AWS Lambda, Amazon SQS o Amazon SNS) cuando ocurren ciertos eventos en tu bucket (por ejemplo, la creación de un objeto, la eliminación). Esto permite construir flujos de trabajo basados en eventos.

7.  **Consultas en el Lugar con S3 Select y S3 Glacier Select:**
    * S3 Select te permite recuperar solo los datos que necesitas de un objeto almacenado en S3 utilizando consultas SQL sencillas. Esto puede mejorar significativamente el rendimiento y reducir los costos de recuperación para archivos grandes. S3 Glacier Select ofrece una funcionalidad similar para datos archivados en S3 Glacier.

**Acceso a S3:**

Puedes acceder a tus buckets y objetos de S3 de diversas maneras:

* **Consola de AWS:** Una interfaz gráfica de usuario basada en web.
* **AWS CLI:** Una interfaz de línea de comandos para interactuar con los servicios de AWS.
* **SDKs de AWS:** Bibliotecas específicas para diferentes lenguajes de programación (Python, Java, .NET, etc.) que te permiten integrar la funcionalidad de S3 en tus aplicaciones.
* **API de REST de S3:** S3 proporciona una API de REST que puedes utilizar directamente a través de solicitudes HTTP.
* **Herramientas de Terceros:** Existen varias herramientas de terceros que facilitan la gestión de buckets y objetos de S3.

**Consideraciones Importantes:**

* **Seguridad:** Implementa el principio de "privilegio mínimo" al configurar las políticas de acceso. Revisa y audita regularmente tus políticas de bucket y de IAM.
* **Costo:** Comprende los modelos de precios de S3, que se basan en el almacenamiento, las solicitudes, la transferencia de datos y la recuperación (para las clases de almacenamiento de acceso infrecuente y archivo). Optimiza tus costos utilizando las clases de almacenamiento adecuadas y las políticas de ciclo de vida.
* **Rendimiento:** S3 está diseñado para ofrecer un alto rendimiento. Considera las directrices de rendimiento de AWS si necesitas optimizar la velocidad de carga y descarga para cargas de trabajo de alto rendimiento.
* **Durabilidad y Disponibilidad:** S3 está diseñado para una durabilidad del 99.999999999% (11 nueves) de los objetos y una alta disponibilidad. Las diferentes clases de almacenamiento ofrecen diferentes niveles de disponibilidad.

En resumen, Amazon S3 proporciona una plataforma robusta y flexible para el almacenamiento y la gestión de archivos en la nube. Comprender sus conceptos, clases de almacenamiento y mecanismos de gestión te permitirá construir aplicaciones escalables, seguras y rentables en AWS. ¡No dudes en preguntar si tienes alguna otra duda!

### Resumen

Existen dos grandes opciones para almacenamiento en AWS:

- **S3**: Es un repositorio de archivos rápido y perfecto para uso de una aplicación a la hora de crear, manipular y almacenar datos.
- **Glacier**: Es un servicio de almacenamiento en la nube para archivar datos y realizar copias de seguridad a largo plazo.

Con S3, AWS te permite guardar archivos en su plataforma, de tal forma, tus instancias EC2, Lamba u otras son efímeras y puedes borrarlas sin preocupación alguna. Tambien te permite hacer respaldos en tiempo prácticamente real en otras regiones de AWS.