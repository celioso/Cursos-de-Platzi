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