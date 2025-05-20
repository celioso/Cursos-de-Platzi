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

## Subida y Hosting de Sitios Web Estáticos en AWS S3

### ¿Cómo subir contenido a S3 y crear un sitio web estático?

En el vasto ecosistema de servicios que ofrece Amazon Web Services (AWS), S3 es uno de los más versátiles y esenciales. No solo puedes almacenar archivos, sino también crear sitios web estáticos fácilmente. Esta guía es para aquellos que desean emprender el camino de explorar esta útil herramienta, logrando no solo almacenar datos, sino también compartirlos con el mundo.

### ¿Cómo se crea un bucket en S3?

Comencemos por abrir la consola de AWS, donde puedes encontrar el servicio S3 bajo la sección de "storage". Siguen estos pasos:

1. **Creación del bucket**: Si un bucket relacionado con Elastic Beanstalk aparece, es porque lo utilizaste anteriormente. Si no, simplemente crea uno nuevo. Recuerda:

- Elige un nombre único que respete las reglas de nombres de dominios (sin caracteres especiales como @, #, espacios, etc.).
- Selecciona la región donde se ubicará el bucket.

2. **Configuraciones adicionales**: Hay varias opciones al crear un bucket, como:

- **Versionado**: Permite guardar múltiples versiones de cada archivo en el bucket.
- **Logs de acceso**: Puedes activar la generación de registros de acceso para monitorizar qué archivos se solicitan.
- **Encriptación**: S3 también ofrece cifrado automatizado para mayor seguridad.

3. Establecer permisos: La configuración predeterminada del bucket es privada, pero puedes hacer que los archivos sean públicos. Una advertencia aparecerá, señalando el riesgo de publicarlas.

### ¿Cómo subir archivos en S3?

Una vez creado el bucket, proceder a cargar archivos es un proceso simple. Supongamos que tienes un proyecto con un archivo index.html y una imagen:

1. Inicia sesión en el bucket y selecciona "Añadir archivos".
2. Carga ambos archivos: el index.html y la imagen.
3. Los archivos podrán configurar sus permisos para hacerlos públicos después de la carga.

### ¿Cómo configurar un hosting web estático?

S3 ofrece la opción de activar el hosting estático directamente en las propiedades del bucket:

1. **Activación del hosting estático**: Dirígete a la sección de hosting estático en las propiedades del bucket.
2. **Configuración de archivos de índice**: Indica el archivo que actuará como el index.html.
3. **Guardar cambios**: Una vez guardados los cambios, tu sitio estará listo para ser publicado con una URL generada por AWS, aunque algo compleja.

### ¿Cómo habilitar replicación entre regiones?

La replicación en S3 es una función poderosa para asegurar que los datos estén disponibles en múltiples ubicaciones geográficas:

1. En "Propiedades", selecciona la opción de replicación.
2. Crear un nuevo bucket en otra región, como Oregón si actualmente trabajas en Ohio.
3. Selecciona el rol adecuado y guarda las configuraciones.

### Puntos a considerar para mejorar la experiencia con S3

- **Diversificación del sitio**: La herramienta AWS Route 53 te permitirá crear un alias para la URL generada, personalizando el acceso.
- **Versionado**: Actualizar tu sitio en diferentes buckets facilita cambios y revertir a versiones anteriores si necesario.

AWS S3 no solo es eficiente para almacenar datos, sino que ofrece un enfoque unificado al crear, gestionar y replicarse como un hosting estático. Ya sea para proyectos pequeños o escala mayor, S3 se presenta como una solución robusta. Sigue explorando y practicando. ¡Cada paso te acerca más a dominar AWS!

**Lecturas recomendadas**

[aws-platzi-python/templates at master · mauropm/aws-platzi-python · GitHub](https://github.com/mauropm/aws-platzi-python/tree/master/templates)

## Almacenamiento Económico en Glacier para Archivos Históricos

AWS tiene un tipo de almacenamiento más económico, pero también más lento que S3 llamado Glacier. Es una muy buena opción si tienes que guardar algún tipo de archivo histórico, como documentos de transacciones de años pasados.

Glacier podrá entregarte tus datos y/o archivos con tiempos de entre 2 y 15 minutos por archivo.

## Bases de Datos en Amazon RDS: Motores y Prestaciones Principales

Amazon RDS (Relational Database Service) es un servicio administrado de base de datos que permite configurar, operar y escalar bases de datos relacionales en la nube de forma sencilla. A continuación te explico los **motores disponibles** y las **prestaciones principales** que ofrece:

### 🔧 **Motores de Base de Datos Compatibles en Amazon RDS**

Amazon RDS soporta los siguientes motores de bases de datos populares:

1. **Amazon Aurora**  
   - Compatible con MySQL y PostgreSQL.
   - Rendimiento hasta 5 veces mayor que MySQL y 3 veces mayor que PostgreSQL.
   - Alta disponibilidad y recuperación automática.

2. **MySQL**  
   - Versión tradicional del motor relacional de código abierto más usado.
   - Fácil migración a Aurora si se desea mejorar el rendimiento.

3. **PostgreSQL**  
   - Motor avanzado con fuerte soporte para funciones complejas, extensiones y cumplimiento de estándares SQL.

4. **MariaDB**  
   - Variante de MySQL enfocada en el rendimiento y la compatibilidad.

5. **Oracle**  
   - Soporta características empresariales, requiere licencia de Oracle (o uso con licencia incluida en RDS).

6. **Microsoft SQL Server**  
   - Compatible con versiones Express, Web, Standard y Enterprise.
   - Permite integración con herramientas Microsoft (como SSMS y Active Directory).

### 🚀 **Prestaciones Principales de Amazon RDS**

1. **Administración Simplificada**
   - No necesitas gestionar hardware, parches o respaldos manuales.

2. **Escalabilidad Automática**
   - Puedes escalar el almacenamiento y la capacidad de cómputo verticalmente con pocos clics o de forma automática.

3. **Alta Disponibilidad y Réplicas**
   - Opciones de Multi-AZ (zonas de disponibilidad) para tolerancia a fallos.
   - Réplicas de lectura para mejorar el rendimiento de consultas.

4. **Seguridad**
   - Cifrado en reposo y en tránsito.
   - Integración con AWS IAM y Amazon VPC.
   - Autenticación mediante Kerberos (para algunos motores).

5. **Backups Automáticos**
   - Snapshots automáticos y manuales.
   - Recuperación punto en el tiempo.

6. **Monitoreo y Mantenimiento**
   - Integración con Amazon CloudWatch, EventBridge y métricas RDS.
   - Mantenimiento automático con opción de ventanas definidas por el usuario.

### 📊 Casos de Uso Comunes

- Aplicaciones web y móviles que requieren una base de datos confiable.
- Migración de bases de datos on-premise a la nube.
- Aplicaciones empresariales con alta disponibilidad y cumplimiento de normativas.

**Resumen**

AWS creó un producto llamado RDS que optimiza el funcionamiento de un motor de bases de datos. Este servicio incluye mantenimiento a tu base de datos, respaldos diarios, optimización para tu tipo de uso, etc.

RDS tiene varias opciones de motores de bases de datos, como: Aurora PG, Aurora MySQL, MySQL, MariaDB, PostgreSQL, Oracle y Microsoft SQL Server.

Recuerda que AWS te da 750 horas de servicio gratis de RDS, incluyendo cualquiera de los motores de bases de datos.

## Administración de RDS Postgres en AWS: Seguridad y Optimización

Administrar **RDS PostgreSQL** en AWS requiere buenas prácticas tanto en **seguridad** como en **optimización de rendimiento**. Aquí tienes una guía clara y estructurada sobre ambos aspectos:

### 🔐 **Seguridad en RDS PostgreSQL**

### 1. **Control de Acceso con IAM y VPC**
- **Acceso a nivel de red (VPC + Security Groups):**
  - Asegúrate de que tu instancia RDS esté en una **VPC privada**.
  - Usa **Security Groups** para permitir solo el tráfico desde IPs o instancias autorizadas (por ejemplo, tu aplicación backend).
- **IAM para gestión (no para conexión directa):**
  - Usa **IAM roles** para permitir o denegar acceso a acciones administrativas sobre RDS (ej. backups, snapshots).

### 2. **Autenticación Segura**
- **Autenticación tradicional con usuarios y contraseñas.**
- **IAM database authentication** *(opcional)*:
  - Permite autenticarse con credenciales temporales de IAM, eliminando contraseñas estáticas.

### 3. **Cifrado**
- **En reposo:** habilita **cifrado de almacenamiento** con KMS al crear la instancia.
- **En tránsito:** habilita **SSL** en la conexión al RDS PostgreSQL.

### 4. **Gestión de Usuarios**
- Crea usuarios con privilegios mínimos necesarios.
- Evita usar el usuario `postgres` para aplicaciones.
- Usa `pg_roles` y `pg_hba.conf` (configurado por AWS internamente) para roles específicos si es necesario.

### 5. **Auditoría**
- Activa los **logs de auditoría** y consúltalos a través de CloudWatch.
- Usa **Amazon RDS Enhanced Monitoring** y **CloudTrail** para seguimiento de operaciones.

### ⚙️ **Optimización de Rendimiento en RDS PostgreSQL**

### 1. **Instancia y Almacenamiento**
- Usa **tipos de instancia adecuados** (ej. `db.m6g`, `db.r6g`) según carga de trabajo.
- Activa **autoscaling de almacenamiento** si esperas crecimiento de datos.
- Usa **storage tipo GP3 o IO1** si necesitas alto rendimiento de IOPS.

### 2. **Parámetros del Servidor**
- Usa el **Parameter Group** para ajustar:
  - `work_mem`, `shared_buffers`, `maintenance_work_mem`, `autovacuum_*`
- Ejemplo: para cargas intensivas de lectura, aumenta `work_mem` para mejorar performance de joins y sorts.

### 3. **Índices y VACUUM**
- Crea **índices adecuados** para las consultas más frecuentes.
- Asegúrate de que el **autovacuum** esté bien configurado.
- Revisa periódicamente los `bloat` (inflamiento de tablas) con herramientas como `pg_stat_user_tables`.

### 4. **Logging de Consultas Lentas**
- Activa `log_min_duration_statement` para identificar consultas lentas.
- Usa **Performance Insights** para analizar métricas de uso de CPU, I/O y wait events.

### 5. **Replicación y Escalabilidad**
- Usa **Read Replicas** para distribuir carga de lectura.
- Considera Amazon Aurora PostgreSQL si necesitas **autoescalado** y mejor rendimiento.

### ✅ Recomendaciones Finales

| Aspecto              | Acción recomendada                                 |
|----------------------|-----------------------------------------------------|
| **Acceso seguro**    | VPC privada + IAM + SSL                             |
| **Auditoría**        | Logs + CloudWatch + RDS Performance Insights        |
| **Performance**      | Ajustes de memoria + índice + autovacuum            |
| **Alta disponibilidad** | Multi-AZ o Aurora PostgreSQL                     |
| **Mantenimiento**    | Configurar ventana de mantenimiento y backups       |

### Resumen

AWS implementa el motor de PostgreSQL (RDS PG) en una instancia optimizada para correr con la máxima eficacia.

RDS PG incluye, por omisión, **tareas de optimización** como vacuum, recuperación de espacio en el disco duro y planificación de queries. Tambien te permite hacer **respaldos** diarios (o incluso más seguido) de tu base de datos.

Otras ventajas de RDS PG son:

- **Cifrado** a tu elección, tu base de datos puede estar cifrada en disco duro
- **Migración asistida**: RDS PG tiene mecanismos que te ayudan a migrar tu información en caso de que tu ya cuentes con una base de datos con otro proveedor.
- **Alta disponibilidad**: RDS PG te permite fácilmente configurar un ambiente de alta disponibilidad al ofrecerte diversas zonas para tu base de datos.

Recuerda que Amazon RDS provee de seguridad por omisión tan alta que no podrás conectarte a tu DB hasta que explícitamente lo permitas.

## Creación y Configuración de Bases de Datos en Amazon RDS

Claro, aquí tienes una guía paso a paso para la **creación y configuración de bases de datos en Amazon RDS**, enfocada en buenas prácticas y aplicable a motores como **PostgreSQL**, **MySQL**, **MariaDB**, **Oracle** y **SQL Server**.

### 🏗️ **1. Crear una Base de Datos en Amazon RDS**

### 🔹 Paso 1: Iniciar la creación
- Ve a la consola de AWS > **RDS > Databases > Create database**
- Selecciona el modo de creación:
  - **Standard Create** (recomendado para control completo)

### 🔹 Paso 2: Elegir el motor de base de datos
- PostgreSQL, MySQL, MariaDB, Oracle o SQL Server
- Ejemplo: **PostgreSQL**

### 🔹 Paso 3: Configurar detalles de la instancia
- **Nombre de la instancia**: `mibasededatos`
- **Credenciales del administrador**:
  - Usuario maestro (ej. `admin`)
  - Contraseña segura (o genera automáticamente con Secrets Manager)

### 🔹 Paso 4: Elegir tipo de instancia
- Elige según carga de trabajo:
  - **t4g.micro / db.t3.micro** para desarrollo/pruebas
  - **db.m6g / db.r6g** para producción

### 🔹 Paso 5: Configurar almacenamiento
- Tipo: **General Purpose (gp3)** o **Provisioned IOPS (io1)** si necesitas rendimiento alto
- Tamaño inicial (ej. 20 GiB) + opción de **autoscaling** del almacenamiento

### 🔒 **2. Configurar conectividad y seguridad**

### 🔹 Red
- Selecciona una **VPC privada** (recomendado)
- Habilita o desactiva el acceso público (según si se accede desde internet)

### 🔹 Grupo de seguridad
- Crea o selecciona un **Security Group** que permita tráfico desde IPs autorizadas (ej. tu servidor de aplicación)

### 🔹 Opciones avanzadas de seguridad
- **Habilitar cifrado en reposo** con KMS (marcar si es necesario)
- **Autenticación con IAM** (opcional, para conexiones sin contraseña)

### ⚙️ **3. Configuración adicional**

### 🔹 Opciones de base de datos
- Nombre de base de datos inicial (ej. `appdb`)
- Puerto por defecto: PostgreSQL (5432), MySQL (3306), etc.

### 🔹 Backup y mantenimiento
- Configura backups automáticos (recomendado: 7 días)
- Configura ventana de mantenimiento y backups automáticos

### 🔹 Monitoreo
- Habilita **Enhanced Monitoring** y **Performance Insights** si es posible

### 🧪 **4. Finalizar y lanzar**
- Revisa toda la configuración
- Haz clic en **Create database**

La creación tomará unos minutos.


### ✅ **5. Acceder a la base de datos**

Una vez creada:
1. Ve a la consola > selecciona la base de datos
2. Copia el **endpoint DNS** y el **puerto**
3. Usa un cliente como **pgAdmin**, **DBeaver**, **MySQL Workbench** o **psql**:
   ```bash
   psql -h <endpoint> -U admin -d appdb -p 5432
   ```

### 🧰 **6. Recomendaciones adicionales**

| Elemento                 | Recomendación                            |
|--------------------------|------------------------------------------|
| Seguridad                | No usar el usuario maestro en la app     |
| Backups                  | Activar + probar restauración            |
| Rendimiento              | Crear índices, activar logging de lentas|
| Escalabilidad            | Habilitar replicas de lectura si es necesario |
| Alta disponibilidad      | Usar opción Multi-AZ                    |

### Resumen

### ¿Cómo crear una base de datos en Amazon RDS?

Crear una base de datos en Amazon RDS es una tarea sencilla que puedes lograr en pocos pasos, y te ofrece una base sólida para experimentar con datos en un entorno seguro. A continuación, te guiaré paso a paso en el proceso, incluyendo configuraciones y consideraciones importantes.

### ¿Qué es RDS y cómo accedemos a él?

Amazon Relational Database Service (RDS) es un servicio gestionado por Amazon Web Services que facilita la configuración, operación y escalabilidad de bases de datos en la nube. Para comenzar, accede a la consola de Amazon Web Services y escribe "RDS" en la barra de búsqueda. Haz clic en el servicio para ingresar.

### ¿Cómo crear la base de datos en RDS?

Con el acceso al servicio RDS, sigue estos pasos para crear una base de datos:

1. **Seleccionar 'DB Instances**': Una vez en la pantalla principal de RDS, haz clic en 'DB Instances' y luego selecciona 'Crear una base de datos'.

2. **Configuración inicial:**

- **Opciones de tipo de base de datos**: Si tu objetivo es experimentar sin costo, asegúrate de seleccionar bases de datos gratuitas.
- **Motor de base de datos**: Elige Postgres, que es una de las opciones más usadas y versátiles para manejar proyectos.

3. Configuración de la instancia:

- Se te asignará una instancia por defecto, pero puedes elegir otra si tus necesidades son diferentes. Cambiar la instancia afectará los costos.
- **Nombre y usuario**: Define el nombre para tu base de datos, por ejemplo, "platzidB", y un usuario con el mismo identificador.

4. Accesibilidad y seguridad:

- Habilita la opción de que sea "accesible públicamente" aún si se requiere un paso adicional en el grupo de seguridad más adelante.
- Configura los puertos y nombre de la base de datos. El puerto por defecto para Postgres suele ser 5432.

### ¿Cómo optimizar la base de datos para producción?

Si en el futuro planeas usar tu base de datos en un entorno de producción, considera lo siguiente:

- **Ajustes de almacenamiento**: Aunque la opción gratuita ofrece veinte gigas, analiza si necesitas más y prepárate para ajustar.
- **Mantenimiento y backups**:
 - Configura ventanas de mantenimiento cuando la base de datos no esté en uso frecuente, como de madrugada.
 - Establece políticas de backup regulares para prevención contra pérdidas de datos.

### ¿Por qué es importante deshabilitar las actualizaciones automáticas?

Deshabilitar las actualizaciones automáticas puede evitar interrupciones no planeadas. Realiza manualmente las actualizaciones en momentos propicios, y siempre haz un backup antes de efectuar un cambio significativo. Así, podrías restaurar el estado previo si algo falla.

### ¿Cuánto tiempo toma la creación de la base de datos?

Una vez configurada toda la información, Amazon RDS comenzará a crear la base de datos. Aunque el estado podría cambiar a "lista", puede tomar entre 5 a 10 minutos adicionales para que esté completamente operativa.

Crear y gestionar bases de datos en Amazon RDS es un proceso que, aunque puede parecer complejo al principio, se vuelve mucho más intuitivo con práctica. ¡Anímate a experimentar y descubre todas las posibilidades que te ofrece este potente recurso!

## Migración de Bases de Datos PostgreSQL a AWS usando Dump

Claro, aquí tienes una guía básica para realizar la **migración de una base de datos PostgreSQL a AWS (usualmente a Amazon RDS para PostgreSQL)** usando el método de **dump**:

### 1. Preparativos

* **Acceso a la base de datos origen (local o en otro servidor)**: Debes tener usuario y contraseña con permisos para hacer dump (exportar).
* **Base de datos destino en AWS RDS PostgreSQL** creada y accesible (con endpoint, usuario y contraseña).
* **Herramientas necesarias**: `pg_dump` y `psql` instalados en tu máquina local o servidor donde harás la migración.

### 2. Exportar la base de datos origen con `pg_dump`

`pg_dump` genera un volcado (dump) de la base de datos en formato SQL o personalizado.

Ejemplo para crear un dump en formato SQL:

```bash
pg_dump -h origen_host -U usuario_origen -d nombre_basedatos_origen -F p -f dump.sql
```

* `-h`: Host donde está la base de datos origen.
* `-U`: Usuario de la base de datos.
* `-d`: Nombre de la base de datos a exportar.
* `-F p`: Formato plain (SQL).
* `-f`: Archivo donde guardar el dump.

Si quieres comprimir el dump:

```bash
pg_dump -h origen_host -U usuario_origen -d nombre_basedatos_origen -F c -f dump.backup
```

* `-F c`: Formato custom, útil para restaurar con `pg_restore`.

### 3. Crear la base de datos destino en AWS RDS (si no existe)

Conéctate a AWS RDS (usando `psql` o consola) y crea la base de datos vacía:

```sql
CREATE DATABASE nombre_basedatos_destino;
```

### 4. Importar el dump en AWS RDS

#### a) Si el dump es en formato SQL (plain):

Usa `psql` para importar directamente:

```bash
psql -h aws_rds_endpoint -U usuario_destino -d nombre_basedatos_destino -f dump.sql
```

#### b) Si el dump está en formato custom (`.backup`):

Usa `pg_restore`:

```bash
pg_restore -h aws_rds_endpoint -U usuario_destino -d nombre_basedatos_destino -v dump.backup
```

### 5. Verificar la migración

* Conéctate a la base de datos AWS RDS y verifica tablas, datos y funciones.
* Revisa permisos y roles si es necesario.


### Consejos adicionales

* Configura adecuadamente el grupo de seguridad de RDS para permitir conexión desde tu IP.
* Puedes usar `--no-owner` y `--no-privileges` en `pg_dump` si quieres evitar conflictos con roles.
* Si la base de datos es muy grande, considera usar AWS Database Migration Service (DMS) para migración en vivo.
* Asegúrate de tener backups antes de hacer migraciones.

**recursos**

[https://github.com/mauropm/aws-platzi-python](https://github.com/mauropm/aws-platzi-python)

## Rendimiento y ventajas de Amazon Aurora PostgreSQL

Amazon Aurora PostgreSQL es una versión administrada y optimizada de PostgreSQL ofrecida por AWS, diseñada para ofrecer **mayor rendimiento, escalabilidad y disponibilidad** en comparación con una instalación tradicional de PostgreSQL.

### 🚀 **Ventajas clave de Amazon Aurora PostgreSQL**

### 1. **📈 Rendimiento Mejorado**

* Hasta **3 veces más rápido** que PostgreSQL estándar.
* Usa una arquitectura de almacenamiento distribuido y tolerante a fallos.
* **Almacenamiento en paralelo** y ejecución eficiente de consultas.
* Optimización automática de caché, índices y uso de CPU.

### 2. **🔄 Escalabilidad Automática**

* El almacenamiento se escala automáticamente de 10 GB hasta 128 TB, sin tiempo de inactividad.
* Soporte para hasta **15 réplicas de lectura** con baja latencia.
* Posibilidad de usar **Aurora Serverless v2**, que escala automáticamente la capacidad según la carga.

### 3. **🔒 Alta Disponibilidad y Recuperación**

* Multi-AZ (zonas de disponibilidad): replica automáticamente en 3 zonas para tolerancia a fallos.
* **Failover automático** en menos de 30 segundos.
* Backups automáticos, snapshots manuales y punto de recuperación temporal (PITR).

### 4. **🛡️ Seguridad**

* Cifrado en reposo con KMS y en tránsito con SSL.
* Integración con IAM y VPC para control de acceso detallado.
* Compatible con grupos de seguridad, ACLs, y opciones de autenticación externa.

### 5. **🛠️ Compatibilidad con PostgreSQL**

* Compatible con muchas extensiones de PostgreSQL como `PostGIS`, `pg_stat_statements`, `uuid-ossp`, etc.
* Puedes migrar desde PostgreSQL sin modificar tu aplicación.

### 6. **💰 Costo-Eficiencia**

* Pagas solo por lo que usas (por hora o por capacidad consumida con Aurora Serverless).
* Menor carga operativa: no necesitas gestionar backups, replicación ni mantenimiento de instancias.

### 🧪 Casos de uso comunes

* Aplicaciones web escalables.
* Análisis de datos con alto rendimiento.
* Reemplazo de bases de datos on-premise.
* Backends para aplicaciones móviles y microservicios.

### 📊 Comparación rápida: PostgreSQL vs Aurora PostgreSQL

| Característica           | PostgreSQL en EC2/RDS | Aurora PostgreSQL         |
| ------------------------ | --------------------- | ------------------------- |
| Rendimiento              | Estándar              | Hasta 3x más rápido       |
| Escalabilidad de Storage | Manual                | Automática hasta 128 TB   |
| Réplicas de lectura      | Limitadas             | Hasta 15                  |
| Recuperación rápida      | Manual                | Failover automático       |
| Administración           | Semi-manual           | Totalmente gestionada     |
| Serverless disponible    | No                    | Sí (Aurora Serverless v2) |

### Resumen

Aurora PG es una nueva propuesta en bases de datos, AWS toma el motor de Postgres, instancias de nueva generación, optimizaciones varias en el kernel/código y obtiene un Postgres 3x más rápido.

Aurora PG es compatible con `Postgres 9.6.x.`

Antes de migrar a Aurora PG debes considerar los siguientes puntos:

- Usar Aurora RDS PG **no es gratis** en ningún momento.
- AWS RDS PG es **eficiente** por varias razones:
 - Modificaciones al código mismo del motos de bases de datos.
 - Instancias de última generación.
- Aurora PG estará por omisión en una configuración de alta disponibilidad con distintas zonas, es decir, en 3 centros de datos a un mismo tiempo.

## Creación y gestión de bases de datos en Aurora PostgreSQL

La **creación y gestión de bases de datos en Amazon Aurora PostgreSQL** combina lo mejor de la compatibilidad con PostgreSQL y las ventajas de una base de datos totalmente administrada por AWS. Aquí tienes una guía práctica paso a paso sobre cómo crear, administrar y optimizar tu base de datos en Aurora PostgreSQL.

### 🏗️ **1. Creación de una Base de Datos Aurora PostgreSQL**

### 🔹 Opción 1: Desde la Consola de AWS

1. Ve a **RDS > Bases de datos > Crear base de datos**.
2. Selecciona:

   * **Motor**: Aurora
   * **Edición**: Aurora PostgreSQL-Compatible Edition
3. Escoge el modo de aprovisionamiento:

   * **Provisión estándar** (instancias dedicadas) o
   * **Aurora Serverless v2** (escalado automático)
4. Configura:

   * Nombre del clúster
   * Nombre de usuario maestro y contraseña
   * Parámetros de red (VPC, subredes, grupo de seguridad, etc.)
   * Configuraciones adicionales como backups, monitoreo, rendimiento, etc.
5. Clic en **Crear base de datos**.

### 🔹 Opción 2: Usando AWS CLI

```bash
aws rds create-db-cluster \
  --db-cluster-identifier mi-cluster-aurora \
  --engine aurora-postgresql \
  --master-username admin \
  --master-user-password TuPasswordSegura123 \
  --vpc-security-group-ids sg-12345678 \
  --db-subnet-group-name mi-subnet-group
```

Luego creas la instancia:

```bash
aws rds create-db-instance \
  --db-instance-identifier mi-instancia-aurora \
  --db-cluster-identifier mi-cluster-aurora \
  --engine aurora-postgresql \
  --db-instance-class db.r6g.large
```

### ⚙️ **2. Gestión de la Base de Datos**

### 🔒 Seguridad

* Usa **grupos de seguridad VPC** para controlar el tráfico.
* Habilita **cifrado en reposo y en tránsito** (SSL/TLS).
* Integra con **IAM para autenticación** y control de acceso.

### 🔁 Backups y recuperación

* Aurora realiza **backups automáticos continuos** hasta por 35 días.
* Puedes crear **snapshots manuales** y clonarlos.
* Usa la recuperación a un punto en el tiempo (PITR) si ocurre un error lógico.

### 🔎 Monitoreo

* Habilita **Enhanced Monitoring** y **Performance Insights**.
* Usa **CloudWatch Logs** para auditoría y diagnóstico.

### 🔄 Escalabilidad

* Añade **réplicas de lectura** para distribuir la carga.
* Cambia de instancia primaria a réplica en caso de mantenimiento o fallos.
* Usa **Aurora Global Databases** si necesitas replicación entre regiones.

### 🧪 **3. Operaciones Comunes**

### 📥 Conexión a la base de datos

```bash
psql -h <endpoint> -U <usuario> -d <nombre_bd> -p 5432
```

### 📂 Crear una base de datos adicional

```sql
CREATE DATABASE mi_nueva_bd;
```

### 👤 Crear usuarios y asignar roles

```sql
CREATE USER mario WITH PASSWORD 'seguro123';
GRANT CONNECT ON DATABASE mi_nueva_bd TO mario;
```

### ⚙️ Configurar parámetros personalizados

Usa un **DB Parameter Group** en la consola de RDS y asócialo al clúster Aurora.

### 📌 Recomendaciones Finales

* ✅ Usa **Aurora Serverless v2** si tienes cargas variables o impredecibles.
* ✅ Siempre activa la autenticación SSL.
* ✅ Usa **réplicas de lectura** para consultas intensivas.
* ✅ Automatiza tareas con **AWS Backup** y **EventBridge** para tareas programadas.
* ✅ Supervisa el uso con **CloudWatch y Performance Insights** para detectar cuellos de botella.

### Resumen

####¿Cómo crear una base de datos en Amazon Aurora compatible con PostgreSQL?

Crear una base de datos en Amazon Aurora compatible con PostgreSQL es un proceso sencillo pero poderoso, que permite aprovechar al máximo las ventajas que ofrece esta plataforma. A continuación, te guiaré a través de los pasos esenciales para configurar tu base de datos desde la consola de Amazon RDS.

#### Configuración inicial de la base de datos

1. **Selecciona el servicio adecuado**: Accede a la consola de RDS de Amazon y elige crear una nueva instancia de base de datos. Asegúrate de seleccionar Amazon Aurora y luego opta por la compatibilidad con PostgreSQL.

2. **Elegir versión y tamaño**: Es importante destacar que Aurora PostgreSQL no es parte del nivel gratuito de AWS. La versión de uso común es PostgreSQL 9.6, que ofrece ventajas como un almacenamiento elástico de hasta 64 TB y es hasta tres veces más rápida que una instancia normal de RDS PostgreSQL.

3. **Especifica el hardware**: Aunque no es gratuito, la opción más económica de Aurora PostgreSQL viene equiparada con dos CPUs y 15 GB de RAM, lo cual proporciona un rendimiento significante desde el primer momento.

### Creación de la base de datos

- **Réplica automática**: Aurora creará automáticamente copias de seguridad en diferentes zonas dentro de la misma región, proporcionando redundancia y protección contra fallos. Este sistema inteligente actuará mediante el DNS dinámico, redirigiendo tráficos si alguna copia falla.

- **Identificación y credenciales**: Define el nombre de la base de datos (ej., PlatziDB2), asigna un nombre de usuario y una contraseña, evitando caracteres especiales problemáticos como comillas y barras.

- **Accesibilidad y cifrado**: Configura si la base de datos es accesible públicamente y habilita el cifrado de datos y los respaldos automáticos para maximizar la seguridad.

#### Gestión y seguridad de conexiones

- **Políticas de seguridad**: Una vez creada, personaliza el grupo de seguridad para definir quiénes pueden conectarse a la base de datos. Para un acceso más abierto, permite conexiones desde cualquier IP, o limita el acceso a IPs específicas de oficina o casa.

- **Monitoreo y actualizaciones**: Deshabilitar actualizaciones automáticas te permite mantener el control absoluto sobre los cambios en la base de datos, evitando interrupciones no planificadas.

#### ¿Cómo insertar datos en la base de datos Aurora PostgreSQL?

Una vez que tu base de datos está creada y configurada correctamente, el siguiente paso es comenzar a poblarla con datos útiles. A continuación te mostramos cómo hacerlo.

#### Conexión e inserción de datos

1. **Conectar a la base de dato**s: Usa herramientas de conexión, como una consola SSH, y actualiza el host con el nuevo endpoint de Aurora PostgreSQL. Asegúrate de ingresar el usuario y la contraseña correctos.

2. **Importar SQL**: Descarga y carga un archivo de volcado de PostgreSQL para poblar la base de datos. Esto se puede hacer con el comando wget seguido del URL del archivo, y luego cargándolo con los comandos psql.

3. **Verificación de tablas**: Usa el comando \dt para listar las tablas en la base de datos y confirmar que tu importación fue exitosa.

#### Migración entre RDS PostgreSQL y Aurora PostgreSQL

Uno de los grandes beneficios de trabajar con Aurora PostgreSQL es la facilidad para migrar desde una instancia de RDS PostgreSQL. Esto es útil cuando tu aplicación supera la capacidad de una RDS estándar y necesitas una solución más robusta sin cambiar mucho tu infraestructura existente.

- **Simplicidad de migración**: Dado que ambas plataformas son compatibles con PostgreSQL, puedes migrar sin esfuerzo y sin modificar el código, conservando beneficios como redundancia y velocidad.

- Rendimiento y redundancia: Al migrar a Aurora, puedes disfrutar de un aumento significativo en el rendimiento y la seguridad, manteniendo tu servicio operativo incluso en caso de fallos en un centro de datos.

La implementación eficaz de Amazon Aurora compatible con PostgreSQL puede llevar tu gestión de bases de datos al siguiente nivel, brindando robustas funcionalidades en términos de rendimiento y confiabilidad. Si implementas los pasos descritos, podrás manejar tus datos de manera más eficiente y segura.

## Prácticas de Respaldo y Recuperación con Amazon RDS

Las **prácticas de respaldo y recuperación con Amazon RDS** son fundamentales para garantizar la disponibilidad y seguridad de tus datos ante errores humanos, fallos del sistema o ciberataques. A continuación, te presento un resumen completo y práctico sobre cómo realizar respaldos y cómo recuperarse eficientemente en Amazon RDS (incluyendo motores como PostgreSQL, MySQL, MariaDB, Oracle y SQL Server).

### 📦 **1. Tipos de Respaldo en Amazon RDS**

### 🔁 **Backups Automáticos**

* Se habilitan por defecto al crear una instancia.
* Conservan copias **diarias completas** y **logs de transacciones** (PITR).
* Se almacenan en S3 gestionado por AWS.
* Retención: entre **1 y 35 días** (puedes configurarlo).
* Te permiten restaurar a un **punto exacto en el tiempo** dentro del período de retención.

> ✅ Recomendación: Establece al menos **7 días de retención** para cubrir errores comunes de usuarios.

### 🧩 **Snapshots Manuales**

* Son respaldos completos realizados bajo demanda.
* Se retienen indefinidamente hasta que los borres manualmente.
* Pueden ser usados para restaurar una instancia en cualquier momento.
* Puedes compartir snapshots entre cuentas y regiones.

```bash
aws rds create-db-snapshot \
  --db-snapshot-identifier mi-snapshot \
  --db-instance-identifier mi-instancia
```

## 🔁 **2. Restauración de una Base de Datos**

### ⏱️ Restaurar a un punto en el tiempo (PITR)

Crea una nueva instancia restaurada desde los backups automáticos:

```bash
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier mi-instancia \
  --target-db-instance-identifier instancia-restaurada \
  --restore-time "2025-05-18T15:00:00Z"
```

> Ideal para recuperar datos antes de un error humano, como una eliminación accidental.

### 💾 Restaurar desde Snapshot Manual

```bash
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier instancia-restaurada \
  --db-snapshot-identifier mi-snapshot
```

### 🔐 **3. Buenas Prácticas de Respaldo y Recuperación**

### 🔒 Seguridad

* Habilita **cifrado en los snapshots** (en reposo y en tránsito).
* Usa **KMS (AWS Key Management Service)** para cifrado personalizado.
* Configura políticas de acceso (IAM) para controlar quién puede crear/borrar snapshots.

### 🌐 Replicación y Alta Disponibilidad

* Usa la opción **Multi-AZ** para recuperación automática ante fallos.
* Replica snapshots a **otras regiones** para DR (Disaster Recovery):

```bash
aws rds copy-db-snapshot \
  --source-db-snapshot-identifier arn:snapshot:region:mi-snapshot \
  --target-db-snapshot-identifier mi-snapshot-copia \
  --source-region us-east-1
```

### 📅 Automatización con AWS Backup

* Usa **AWS Backup** para centralizar y automatizar respaldos.
* Permite establecer políticas de retención y copia entre regiones.

### 🧪 4. Validación y Pruebas

> 📌 **Probar regularmente tus planes de recuperación** es tan importante como hacer respaldos.

* Restaura tus snapshots en una instancia de prueba.
* Valida integridad de datos y funcionalidad de la aplicación.
* Documenta tiempos estimados de recuperación (RTO y RPO).

### 📋 Ejemplo de Estrategia de Backup

| Tipo               | Frecuencia | Retención | Cifrado  | Objetivo                        |
| ------------------ | ---------- | --------- | -------- | ------------------------------- |
| Backup automático  | Diario     | 7 días    | Sí (KMS) | PITR                            |
| Snapshot manual    | Semanal    | 4 semanas | Sí       | Restauración controlada         |
| Snapshot replicado | Mensual    | 3 meses   | Sí       | Recuperación ante desastre (DR) |

### Resumen

#### ¿Cuáles son las mejores prácticas para el uso de RDS?

El uso eficiente de Amazon Relational Database Service (RDS) es esencial para cualquier negocio que dependa de bases de datos robustas, ya sea en producción o en análisis. Siempre es recomendable seguir ciertas mejores prácticas para garantizar la integridad y disponibilidad de tus datos.

#### ¿Por qué es crucial realizar respaldos frecuentes?

Realizar respaldos con frecuencia es una estrategia fundamental en la administración de cualquier base de datos. Esto no solo protege contra la pérdida de datos, sino también sirve como un salvavidas en caso de fallas inesperadas del sistema. Considera las siguientes recomendaciones:

- **Frecuencia de los respaldos**: Idealmente, deberías respaldar tus datos diariamente. Esto asegura que, en el peor escenario, solo perderías un día de información. Si tus operaciones son críticas, considera aumentar la frecuencia según tus necesidades.
- **Formato de los respaldos**: Los respaldos en RDS se guardan como "snapshots", imágenes de la máquina en la que reside tu base de datos. Estas instantáneas permiten restaurar rápidamente una nueva instancia sin complicaciones.
- **Escenarios especializados**: Si bien el respaldo diario puede ser suficiente para algunas aplicaciones, en sectores como el financiero o de salud, donde se maneja información crítica, es esencial realizar copies de transacciones más frecuentemente.

#### ¿Cómo podemos optimizar la recuperación de datos?

La capacidad de recuperación rápida y eficaz de datos es vital, especialmente en sectores que requieren alta disponibilidad y pérdida mínima de información. Amazon RDS ofrece soluciones diseñadas para tal propósito:

- **Replicación entre regiones**: Con RDS, puedes configurar una réplica en otra región geográfica para asegurar que tienes un respaldo inmediato si algo catastrófico ocurre. Esta "read replica" sincroniza constantemente con la base de datos principal.
- **Minimización de pérdida de datos**: Dependiendo de la latencia y volumen de información, es posible alcanzar eficiencias donde la pérdida de datos se minimiza a unos pocos minutos. Esto es crucial para servicios financieros, donde la pérdida de información crítica debe ser insignificante.

#### ¿Cómo gestionar eficiente los cambios de DNS?

A la hora de restaurar datos de un snapshot, puedes optar por dos caminos distintos:

- **Cambio de DNS**: Al restaurar una nueva instancia, actualizar el DNS es una solución rápida que garantiza la continuidad sin necesidad de ajustes significativos en las configuraciones del cliente.
- **Acceso directo a la información restaurada**: En lugar de cambiar el DNS, una opción es obtener la información necesaria desde la nueva instancia restaurada sin editar conexiones existentes. Esto puede ser útil en situaciones donde solo se requiere datos específicos o históricos.

Implementar una estrategia integral de respaldo y recuperación en RDS te asegurará un manejo más seguro y eficiente de tus datos. Incorpora estas prácticas y optimiza el rendimiento y disponibilidad de tus sistemas, garantizando así la continuidad de tu negocio frente a cualquier eventualidad.

## Gestión de DNS y dominios con Amazon Route 53

La **gestión de DNS y dominios con Amazon Route 53** es una parte clave en la arquitectura de aplicaciones modernas en AWS. Route 53 es un servicio altamente disponible y escalable que te permite registrar dominios, gestionar registros DNS y configurar enrutamiento inteligente de tráfico.

### 🌐 ¿Qué es Amazon Route 53?

Amazon Route 53 es un **servicio de DNS (Domain Name System)** administrado por AWS que proporciona:

* **Registro de dominios**.
* **Resolución DNS pública y privada**.
* **Balanceo geográfico o por latencia**.
* **Monitoreo del estado de recursos (health checks)**.
* **Integración con AWS para dominios personalizados de servicios como API Gateway, S3, ELB, etc.**

### 🛠️ Funciones Principales de Route 53

### 1. ✅ **Registro de Dominios**

Puedes registrar dominios directamente desde la consola de AWS.

```bash
aws route53domains register-domain \
  --domain-name mi-sitio.com \
  --duration-in-years 1 \
  --admin-contact file://contact.json \
  --registrant-contact file://contact.json \
  --tech-contact file://contact.json
```

> 📌 Puedes transferir dominios existentes a Route 53 si ya los tienes con otro proveedor.

### 2. 🧭 **Gestión de Zonas Hosted Zones**

Una **hosted zone** es un contenedor para registros DNS de un dominio.

#### Crear una zona hospedada:

```bash
aws route53 create-hosted-zone \
  --name mi-sitio.com \
  --caller-reference "$(date +%s)"
```

#### Tipos comunes de registros:

| Tipo  | Uso                                    |
| ----- | -------------------------------------- |
| A     | Apunta a una dirección IP IPv4         |
| AAAA  | Apunta a una IP IPv6                   |
| CNAME | Alias a otro nombre (no raíz)          |
| MX    | Correo electrónico                     |
| TXT   | Verificación de dominio (SPF, DKIM)    |
| NS    | Nameservers (automático al crear zona) |

### 3. 🚦 **Routing Policies (Políticas de Enrutamiento)**

| Tipo                  | Descripción                                            |
| --------------------- | ------------------------------------------------------ |
| **Simple**            | Apunta a una única IP o nombre                         |
| **Weighted**          | Balanceo por peso entre varios recursos                |
| **Latency-based**     | Redirige al recurso con menor latencia                 |
| **Geolocation**       | Basado en la ubicación del cliente                     |
| **Failover**          | Enrutamiento activo/pasivo con health checks           |
| **Multivalue Answer** | Devuelve múltiples valores A o AAAA con disponibilidad |

> Ejemplo: Puedes redirigir usuarios en Sudamérica a una instancia EC2 en São Paulo y usuarios en Europa a Frankfurt.

### 4. 💡 **Health Checks**

Permiten monitorear el estado de endpoints HTTP/HTTPS/TCP.

* Si un recurso está caído, puedes redirigir tráfico a un recurso alterno.
* Puedes usarlos en conjunto con políticas **failover** o **multivalue**.

### 5. 🔒 **DNS Privado con VPC**

Route 53 puede actuar como DNS interno de una **VPC**:

* Solo accesible dentro de tu red privada.
* Ideal para microservicios internos, bases de datos, etc.

```bash
aws route53 create-hosted-zone \
  --name miapp.local \
  --vpc VPCRegion=us-east-1,VPCId=vpc-abc123 \
  --hosted-zone-config Comment="DNS privado",PrivateZone=true
```

### 6. 📦 Integración con Otros Servicios AWS

* **CloudFront**: asociar nombres personalizados.
* **API Gateway**: dominios personalizados con SSL.
* **Elastic Load Balancer (ELB)**: usar registros alias.
* **S3**: para sitios web estáticos.
* **ACM**: para certificados SSL/TLS validados por DNS.

### 🔐 Seguridad y Mejores Prácticas

* Usa **MFA** en la cuenta raíz.
* **Bloquea transferencias de dominio** si no estás migrando.
* Configura registros **CAA** para autorizar certificados SSL de ciertas entidades.
* Usa **CloudTrail** para auditar cambios en dominios y zonas.

### ✅ Caso de uso típico

1. Registrar dominio: `miempresa.com`
2. Crear zona hospedada.
3. Configurar registros A/CNAME para apuntar a:

   * un ELB (`myapp-123456.elb.amazonaws.com`)
   * un bucket S3 (`miempresa.com.s3-website-us-east-1.amazonaws.com`)
4. Validar dominio en ACM para emitir certificado TLS.
5. Apuntar dominio personalizado en CloudFront o API Gateway

### Resumen

Existen muchos servicios de redes en AWS, uno de los más interesantes es Route 53.

AWS te permite tener un DNS muy avanzado a tu disposición, con el podrás hacer subdominios asignados a instancias y verlos reflejados en segundos.

**Route 53** está disponible en todas las regiones de AWS, por lo que funcionará excelente aún en caso de que alguna de las regiones se pierda.

## Gestión de Usuarios y Recursos en Amazon AWS

La **Gestión de Usuarios y Recursos en Amazon AWS** es esencial para administrar de forma segura quién tiene acceso a qué servicios y cómo se utilizan los recursos en la nube. Esta gestión gira principalmente en torno a **AWS Identity and Access Management (IAM)**, pero también involucra herramientas como **AWS Organizations**, **AWS Resource Groups** y **etiquetado (tagging)** de recursos.

### 🧑‍💼 1. Gestión de Usuarios con IAM (Identity and Access Management)

### ✅ ¿Qué es IAM?

AWS IAM te permite crear y gestionar **usuarios, grupos, roles y políticas** para controlar el acceso a los recursos de AWS.

### 🧾 Elementos clave de IAM:

* **Usuarios**: Identidades permanentes para personas u otros servicios.
* **Grupos**: Colecciones de usuarios que comparten permisos.
* **Roles**: Identidades temporales usadas por servicios o usuarios externos (como federación o acceso entre cuentas).
* **Políticas**: JSONs que definen permisos (acciones, recursos y condiciones).

### 🛡️ Buenas prácticas de seguridad en IAM:

| Práctica                                    | Descripción                                       |
| ------------------------------------------- | ------------------------------------------------- |
| 🚫 No usar la cuenta raíz                   | Solo para tareas excepcionales. Protege con MFA.  |
| 🔐 Activar MFA                              | Obligatorio para usuarios con permisos sensibles. |
| 📄 Usar políticas con privilegios mínimos   | “Menos es más” para evitar accesos indebidos.     |
| ⏳ Usar roles temporales                     | Especialmente en entornos productivos.            |
| 🧪 Revisar permisos con IAM Access Analyzer | Detecta accesos no intencionales.                 |

### 🗃️ 2. Gestión de Recursos con Etiquetas (Tags)

### ¿Qué son las etiquetas?

Son **pares clave-valor** que puedes asignar a casi cualquier recurso en AWS (EC2, S3, RDS, Lambda, etc.).

### ¿Para qué sirven?

* **Organización** por proyecto, entorno, dueño, etc.
* **Filtrado** y agrupación de recursos en la consola.
* **Control de costos** (Cost Explorer puede agrupar por tags).
* **Aplicar políticas IAM basadas en tags**.

```json
{
  "Resource": "arn:aws:ec2:us-east-1:123456789012:instance/*",
  "Condition": {
    "StringEquals": {
      "aws:RequestTag/project": "mi-app"
    }
  }
}
```

### 🏢 3. Gestión de Cuentas con AWS Organizations

### ¿Qué es AWS Organizations?

Permite **agrupar varias cuentas AWS** bajo una jerarquía centralizada y aplicar políticas de control (SCPs).

### Beneficios:

* **Control centralizado** de facturación y acceso.
* **Separación por entornos** (producción, desarrollo, testing).
* **Aplicación de políticas restrictivas** a nivel de cuenta o unidad organizativa.
* **Consolidación de costos**.

### 🧰 4. Resource Groups

Permiten **agrupar recursos de distintos tipos** (EC2, RDS, Lambda, etc.) bajo un mismo conjunto lógico, usualmente con tags comunes.

* Filtra recursos fácilmente.
* Aplica acciones sobre múltiples recursos.
* Útil para administración a escala.

### 📊 5. Monitoreo y Auditoría

| Servicio                | Uso                                                                 |
| ----------------------- | ------------------------------------------------------------------- |
| **CloudTrail**          | Audita toda la actividad (quién hizo qué, cuándo y desde dónde).    |
| **IAM Access Analyzer** | Revisa quién puede acceder a tus recursos desde fuera de tu cuenta. |
| **Config**              | Historiza configuraciones de recursos y detecta cambios.            |
| **CloudWatch Logs**     | Monitorea y genera alertas sobre comportamiento del sistema.        |

### 🧠 Ejemplo de flujo para una empresa:

1. Se crean **cuentas independientes** para cada departamento con **AWS Organizations**.
2. En cada cuenta:

   * Se crean **grupos IAM** como "developers", "ops", etc.
   * Cada grupo tiene permisos limitados a sus recursos (por tags o por servicios).
   * Se crean **roles con MFA obligatorio** para acciones administrativas.
3. Todos los recursos se etiquetan por proyecto, entorno y dueño.
4. Se usan **SCPs** para evitar creación de ciertos servicios fuera de política.
5. Se audita actividad con CloudTrail y se revisa mensualmente con Access Analyzer.

### Resumen

Existen muchas herramientas de administración en AWS muy útiles, las siguientes tres son las más importantes:

1. **IAM** te permite administrar todos los permisos de acceso de usuarios y máquinas sobre máquinas.

2. **CloudWatch** te mostrará diversos eventos relacionados con tu infraestructura o servidores, para tener un lugar centralizado de logs e información.

3. **Cloudtrail** es una herramienta de auditoria que permite ver quién o qué hizo que actividad en tu cuenta de AWS.

Cada uno de los productos de AWS tienen diversas alternativas para acceder a más logs, estas opciones cuentan con almacenamiento histórico y hacen un gran trabajo al tratar la información para auditar actividades y deshabilitar usuario.

## Creación y Configuración de Usuarios IAM Programáticos en AWS

La **creación y configuración de usuarios IAM programáticos en AWS** es fundamental cuando deseas que aplicaciones, scripts o herramientas externas accedan a tus servicios de AWS de manera segura, sin necesidad de iniciar sesión en la consola.

### 🔐 ¿Qué es un usuario IAM programático?

Un **usuario IAM programático** es un usuario que no necesita acceso a la consola web de AWS, sino que se autentica y opera mediante **credenciales de acceso** (Access Key ID y Secret Access Key) para utilizar servicios como S3, EC2, DynamoDB, etc., vía CLI, SDKs o APIs.

### ✅ Pasos para crear y configurar un usuario IAM programático

### 1. **Accede a la consola de IAM**

Ve a: [https://console.aws.amazon.com/iam](https://console.aws.amazon.com/iam)

### 2. **Crea un nuevo usuario IAM**

* Ir a **Users (Usuarios)** → clic en **“Add user” (Agregar usuario)**
* Asigna un nombre (ej. `app-user-s3`)
* **Selecciona solo acceso programático** (✔ *Access key - Programmatic access*)

### 3. **Asignar permisos**

Tienes tres opciones:

#### a. **Adjuntar directamente políticas existentes**

Selecciona una política como:

* `AmazonS3ReadOnlyAccess`
* `AmazonDynamoDBFullAccess`
* O crea una política personalizada

#### b. **Agregar al grupo con permisos**

Si tienes un grupo predefinido (ej. `developers-s3-access`), agrégalo al grupo.

#### c. **Crear política personalizada**

Ejemplo para acceso limitado a un bucket S3:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject"],
      "Resource": "arn:aws:s3:::mi-bucket/*"
    }
  ]
}
```

### 4. **Revisar y crear**

* Revisa la configuración
* Haz clic en **“Create user”**

### 5. **Guardar credenciales**

Se mostrará:

* `Access Key ID`
* `Secret Access Key`

⚠️ **Guárdalas de inmediato** (descarga el `.csv` o copia manualmente). El Secret Access Key **no se vuelve a mostrar**.

### 6. **Usar las credenciales programáticamente**

#### a. Con AWS CLI:

```bash
aws configure
```

Introduce el `Access Key ID`, `Secret Access Key`, región y formato de salida.

#### b. Variables de entorno:

```bash
export AWS_ACCESS_KEY_ID=AKIAXXXXXXXXX
export AWS_SECRET_ACCESS_KEY=abc123xxxxxxxx
```

#### c. En tu código (Python con `boto3`):

```python
import boto3

s3 = boto3.client(
    's3',
    aws_access_key_id='AKIAXXXXXX',
    aws_secret_access_key='abc123XXXXX'
)
s3.list_buckets()
```

### 🛡️ Buenas prácticas de seguridad

| Práctica                    | Recomendación                                 |
| --------------------------- | --------------------------------------------- |
| 🔐 Rotación de claves       | Cambia cada 90 días                           |
| 🧹 Elimina claves no usadas | Revisa su uso en IAM                          |
| 🧾 Usa permisos mínimos     | Evita políticas `*:*`                         |
| 🔍 Monitorea con CloudTrail | Audita el uso del usuario                     |
| 🤝 Considera usar roles     | Para cargas en EC2, Lambda o EKS (más seguro) |

### Resumen

#### ¿Cómo utilizar IAM para gestionar usuarios en AWS?

La gestión de usuarios en Amazon Web Services (AWS) mediante IAM (Identity and Access Management) es fundamental para mantener la seguridad y el manejo adecuado de recursos en la nube. Al crear usuarios, grupos y claves de acceso, puedes controlar quién tiene acceso a qué recursos y cómo interactúan con ellos. A continuación, te describimos cómo puedes crear un usuario programático y gestionar sus permisos.

#### ¿Qué es un acceso programático y para qué se usa?

El acceso programático se refiere a la capacidad de interactuar con AWS mediante la línea de comandos o programas externos, en lugar de hacerlo a través de la consola web. Esto es especialmente útil para:

- Automatizar tareas repetitivas.
- Integrar servicios de AWS en aplicaciones.
- Ejecutar scripts o aplicaciones que necesitan acceder a AWS.

#### ¿Cómo crear un usuario programático en IAM?

Para crear un usuario programático en AWS, sigue estos pasos:

1. **Accede a la consola IAM**: Inicia sesión en tu cuenta de AWS y navega a la sección de seguridad, identidad y cumplimiento, y luego a IAM.

2. **Crea un usuario nuevo**: Selecciona "Crear usuario" y asigna un nombre; en este ejemplo, usaremos "Platzi key".

3. **Define el tipo de acceso**: Especifica que el acceso será programático.

4. Establece permisos:

 - Crea un grupo (por ejemplo, "Platzigrupo").
 - Asigna políticas de permisos, como que el grupo tenga acceso a S3 para lectura/escritura.

5. Revisa y crea el usuario: AWS ofrecerá un resumen con la opción de descargar una key de acceso y un secreto, elementos necesarios para la conexión programática.

#### ¿Por qué es importante el access key y el secret key?

Una vez creado el usuario, AWS proporciona un `access key` y un `secret key`. Estos son esenciales para establecer conexiones de manera segura con los servicios de AWS desde aplicaciones externas. Es crucial guardar esta información de forma segura, ya que AWS no permite ver el secret key una vez cerrada la ventana inicial de creación. Se recomienda guardar esta información en un archivo CSV, enviarla por correo, o almacenarla en una USB.

#### ¿Cómo probar el acceso programático con aplicaciones externas?

Para demostrar la funcionalidad del usuario programático, se puede utilizar la aplicación Cyberduck para conectarse a AWS S3:

1. **Descarga e instala Cyberduck**: Disponible para Windows y Mac, permite gestionar archivos en la nube.

2. **Configura una conexión**:

 - Abre Cyberduck y selecciona S3 como el tipo de conexión.
 - Ingresa el access key y el secret key cuando sea solicitado.

3. **Verifica la conexión**: Podrás ver y gestionar los recursos disponibles en tu bucket S3, como subir o descargar archivos, y realizar otras acciones.

#### ¿Cuáles son las recomendaciones para gestionar las claves de acceso?

1. **Seguridad primero**: Almacena siempre tus claves de acceso de manera segura, evitando posibles filtraciones o pérdidas.

2. **Cierre de sesión y copias de seguridad**: Realiza copias de seguridad del archivo CSV con las claves, y almacénalas en un lugar seguro.

3. **Acceso controlado**: Considera crear cuentas individuales para cada persona que necesite acceso, con permisos específicos según su rol.

Utilizar IAM de manera correcta y segura permite una gestión efectiva de los recursos y usuarios dentro de AWS. Experimenta con estos pasos para comprender mejor cómo gestionar usuarios y sus accesos, asegurando así el correcto funcionamiento de tus aplicaciones en la nube. ¡Sigue así y continúa explorando las posibilidades que AWS te ofrece!

## Monitoreo de Actividades en AWS con Cloudwatch

El **monitoreo de actividades en AWS con Amazon CloudWatch** es clave para observar, registrar y reaccionar ante eventos y métricas que afectan el rendimiento y seguridad de tus recursos en la nube. A continuación, te explico sus componentes principales, cómo funciona y cómo usarlo de manera efectiva.

### 🎯 ¿Qué es Amazon CloudWatch?

Es un **servicio de monitoreo** nativo de AWS que recopila datos de rendimiento en tiempo real (métricas, logs, eventos y alarmas) de servicios como EC2, Lambda, RDS, DynamoDB, S3, entre otros.

### 🧱 Componentes principales de CloudWatch

| Componente         | Función                                                                          |
| ------------------ | -------------------------------------------------------------------------------- |
| **Métricas**       | Valores numéricos (ej. CPU, memoria, latencia) recolectados cada 1 o 5 min       |
| **Logs**           | Archivos de eventos generados por tus apps o servicios (ej. errores, peticiones) |
| **Alarmas**        | Permiten reaccionar ante condiciones críticas (ej. CPU > 80%)                    |
| **Dashboards**     | Visualizaciones personalizadas de métricas en tiempo real                        |
| **Events / Rules** | Automatización basada en eventos (ej. reiniciar instancia EC2 si falla)          |

### 🔧 ¿Qué puedes monitorear con CloudWatch?

| Servicio         | Ejemplo de Métricas                        |
| ---------------- | ------------------------------------------ |
| **EC2**          | Uso de CPU, disco, red, estado del sistema |
| **RDS / Aurora** | Latencia, conexiones, uso de CPU/disco     |
| **Lambda**       | Duración, invocaciones, errores            |
| **DynamoDB**     | Read/Write Capacity, ThrottledRequests     |
| **API Gateway**  | Conteo de solicitudes, errores, latencia   |
| **S3**           | Bytes almacenados, peticiones, errores     |

### ✅ Ejemplo de flujo de monitoreo

1. **Recopilación de métricas:**
   AWS genera métricas por defecto. Puedes enviar métricas personalizadas usando SDK o CLI.

2. **Creación de alarmas:**
   Ejemplo: Si `CPUUtilization > 80%` por 5 minutos, envía notificación SNS.

3. **Visualización con dashboards:**
   Crea gráficos personalizados para múltiples métricas en una sola vista.

4. **Reacción automatizada:**
   Usa **CloudWatch Events o Alarm Actions** para:

   * Reiniciar instancias
   * Llamar a una Lambda
   * Escalar grupos de Auto Scaling
   * Notificar por correo o Slack vía SNS

### 📦 Logs con CloudWatch Logs

Puedes enviar logs desde:

* **EC2** (mediante CloudWatch Agent)
* **Lambda** (logs automáticos)
* **ECS, Fargate** (con FireLens)
* **Aplicaciones personalizadas** (SDK)

Ejemplo: Enviar logs de una app en EC2:

```bash
sudo yum install amazon-cloudwatch-agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard
```

### 📊 Crear una alarma con AWS Console

1. Ir a CloudWatch > Alarms > Create Alarm
2. Seleccionar la métrica (ej. CPU de EC2)
3. Configurar condición (ej. > 80%)
4. Añadir acción: enviar notificación SNS o ejecutar Lambda
5. Crear

### 🧠 Buenas prácticas

* **Crear dashboards por ambiente** (producción, pruebas)
* **Usar etiquetas (tags)** para filtrar métricas por proyecto o equipo
* **Agrupar logs con filtros métricos** (ej. errores 500 en API)
* **Configurar retención adecuada** de logs (ej. 30 o 90 días)
* **Combinar con CloudTrail** para auditar actividades sospechosas

### 🚀 ¿Quieres un ejemplo de monitoreo automático con Terraform o CloudFormation?

Puedo ayudarte a generar el código para:

* Enviar métricas personalizadas
* Crear una alarma y un SNS topic
* Automatizar el despliegue de dashboards o CloudWatch Agent

### Resumen

#### ¿Cómo utilizar IAM para gestionar usuarios en AWS?

La gestión de usuarios en Amazon Web Services (AWS) mediante IAM (Identity and Access Management) es fundamental para mantener la seguridad y el manejo adecuado de recursos en la nube. Al crear usuarios, grupos y claves de acceso, puedes controlar quién tiene acceso a qué recursos y cómo interactúan con ellos. A continuación, te describimos cómo puedes crear un usuario programático y gestionar sus permisos.

#### ¿Qué es un acceso programático y para qué se usa?

El acceso programático se refiere a la capacidad de interactuar con AWS mediante la línea de comandos o programas externos, en lugar de hacerlo a través de la consola web. Esto es especialmente útil para:

- Automatizar tareas repetitivas.
- Integrar servicios de AWS en aplicaciones.
- Ejecutar scripts o aplicaciones que necesitan acceder a AWS.

#### ¿Cómo crear un usuario programático en IAM?

Para crear un usuario programático en AWS, sigue estos pasos:

1. **Accede a la consola IAM**: Inicia sesión en tu cuenta de AWS y navega a la sección de seguridad, identidad y cumplimiento, y luego a IAM.

2. **Crea un usuario nuevo**: Selecciona "Crear usuario" y asigna un nombre; en este ejemplo, usaremos "Platzi key".

3. **Define el tipo de acceso**: Especifica que el acceso será programático.

4. Establece permisos:

 - Crea un grupo (por ejemplo, "Platzigrupo").
 - Asigna políticas de permisos, como que el grupo tenga acceso a S3 para lectura/escritura.

5. Revisa y crea el usuario: AWS ofrecerá un resumen con la opción de descargar una key de acceso y un secreto, elementos necesarios para la conexión programática.

#### ¿Por qué es importante el access key y el secret key?

Una vez creado el usuario, AWS proporciona un `access key` y un `secret key`. Estos son esenciales para establecer conexiones de manera segura con los servicios de AWS desde aplicaciones externas. Es crucial guardar esta información de forma segura, ya que AWS no permite ver el secret key una vez cerrada la ventana inicial de creación. Se recomienda guardar esta información en un archivo CSV, enviarla por correo, o almacenarla en una USB.

#### ¿Cómo probar el acceso programático con aplicaciones externas?

Para demostrar la funcionalidad del usuario programático, se puede utilizar la aplicación Cyberduck para conectarse a AWS S3:

1. **Descarga e instala Cyberduck**: Disponible para Windows y Mac, permite gestionar archivos en la nube.

2. **Configura una conexión**:

 - Abre Cyberduck y selecciona S3 como el tipo de conexión.
 - Ingresa el access key y el secret key cuando sea solicitado.

3. **Verifica la conexión**: Podrás ver y gestionar los recursos disponibles en tu bucket S3, como subir o descargar archivos, y realizar otras acciones.

#### ¿Cuáles son las recomendaciones para gestionar las claves de acceso?

1. **Seguridad primero**: Almacena siempre tus claves de acceso de manera segura, evitando posibles filtraciones o pérdidas.

2. **Cierre de sesión y copias de seguridad**: Realiza copias de seguridad del archivo CSV con las claves, y almacénalas en un lugar seguro.

3. **Acceso controlado**: Considera crear cuentas individuales para cada persona que necesite acceso, con permisos específicos según su rol.

Utilizar IAM de manera correcta y segura permite una gestión efectiva de los recursos y usuarios dentro de AWS. Experimenta con estos pasos para comprender mejor cómo gestionar usuarios y sus accesos, asegurando así el correcto funcionamiento de tus aplicaciones en la nube. ¡Sigue así y continúa explorando las posibilidades que AWS te ofrece!

## Monitoreo de Actividades en AWS con CloudTrail

El **monitoreo de actividades en AWS con AWS CloudTrail** permite auditar, registrar y analizar todas las acciones realizadas en tu cuenta, ya sea por usuarios, servicios o roles. Es fundamental para la **seguridad, cumplimiento y análisis forense** en entornos AWS.

### 🛡️ ¿Qué es AWS CloudTrail?

CloudTrail es un servicio de **auditoría y registro de eventos** que captura todas las **acciones realizadas en la consola de AWS, la CLI, SDKs y APIs**. Guarda estos registros en S3, y se pueden consultar directamente, enviar a CloudWatch Logs, o analizar con Athena.

### 📌 ¿Qué tipo de actividades registra CloudTrail?

CloudTrail graba eventos como:

| Actividad                               | Ejemplo                                               |
| --------------------------------------- | ----------------------------------------------------- |
| **Acciones administrativas**            | Crear/Eliminar usuarios IAM, roles, políticas         |
| **Cambios en recursos**                 | Iniciar/detener instancias EC2, modificar RDS, etc.   |
| **Acciones automatizadas de servicios** | Auto Scaling, Lambda, CloudFormation                  |
| **Eventos de autenticación**            | Inicio de sesión, intentos fallidos, cambios de clave |

### 🔍 ¿Cómo funciona?

1. **Registro de eventos:** Cada vez que alguien o algo hace una solicitud a un servicio AWS compatible, CloudTrail registra el evento.
2. **Almacenamiento en S3:** Los eventos se almacenan en un bucket de Amazon S3.
3. **Envío a CloudWatch (opcional):** Puedes enviarlos para análisis en tiempo real o activar alarmas.
4. **Consulta con Athena (opcional):** Realiza consultas SQL a los registros para investigar eventos.

### 🧱 Componentes de CloudTrail

| Componente          | Función                                                                            |
| ------------------- | ---------------------------------------------------------------------------------- |
| **Trail**           | Conjunto de configuraciones que define qué eventos registrar y dónde almacenarlos. |
| **Event history**   | Historial de eventos de los últimos 90 días (sin configuración adicional).         |
| **S3 Bucket**       | Almacén persistente de eventos (recomendado para auditoría a largo plazo).         |
| **CloudWatch Logs** | Permite monitoreo y creación de alarmas en tiempo real.                            |

### ✅ Crear un Trail con AWS Console

1. Ir a **CloudTrail > Trails > Create trail**
2. Nombre del trail: `trail-global`
3. Marcar **Enable for all regions**
4. Elegir o crear un **S3 bucket**
5. (Opcional) Enviar eventos a CloudWatch Logs
6. Confirmar y crear

### 🧪 Ejemplo de evento CloudTrail (formato JSON)

```json
{
  "eventTime": "2025-05-18T17:45:00Z",
  "eventName": "RunInstances",
  "awsRegion": "us-east-1",
  "userIdentity": {
    "type": "IAMUser",
    "userName": "admin-user"
  },
  "sourceIPAddress": "192.0.2.0",
  "requestParameters": {
    "instanceType": "t2.micro"
  }
}
```

### 📊 Integración con Athena para análisis

1. CloudTrail guarda eventos en S3.
2. Usa **AWS Glue** para catalogar los datos.
3. Consulta los logs con **Athena**:

```sql
SELECT eventName, userIdentity.userName, eventTime
FROM cloudtrail_logs
WHERE eventName = 'DeleteBucket'
ORDER BY eventTime DESC
```

### 🧠 Buenas prácticas con CloudTrail

* **Habilita CloudTrail en todas las regiones** (protección completa).
* **Protege el bucket S3 de logs** con políticas y cifrado.
* **Activa el envío a CloudWatch** para alertas inmediatas.
* **Monitorea eventos sospechosos** como:

  * `ConsoleLogin` fallidos
  * Creación/eliminación de claves de acceso
  * Modificación de políticas IAM

### 🚨 Ejemplo de uso: Detección de intentos sospechosos

Puedes configurar una **alarma en CloudWatch** que se dispare si hay varios `ConsoleLogin` fallidos en 10 minutos, y enviar notificación con SNS o ejecutar una Lambda.

### Resumen

#### ¿Qué es CloudTrail y para qué sirve?

CloudTrail es una poderosa herramienta de monitoreo en la plataforma de Amazon Web Services (AWS). Esta herramienta permite rastrear y registrar cada acción o evento que se realiza en tu cuenta de AWS. Imagina que es como tener un detective digital dentro de tu cuenta que te informa sobre quién hizo qué y cuándo. Este recurso es invaluable para la gestión de la seguridad y cumplimiento al ofrecer un registro detallado de las actividades en el entorno de AWS.

#### ¿Cómo utilizar CloudTrail en la consola de AWS?

Al acceder a la consola de AWS, puedes buscar CloudTrail para comenzar a explorar sus capacidades. Al ingresar, tendrás acceso a una visión general de los eventos recientes, incluyendo detalles como el nombre de usuario y el momento del evento. Para una experiencia más enriquecedora, puedes dirigirte a la sección de historial que presenta una interfaz más amigable y detallada para analizar la actividad específica de usuarios individuales.

#### ¿Cómo rastrear actividades específicas de un usuario?

1. **Buscar actividades por usuario**: Por ejemplo, si se creó un usuario llamado "Platziki" para acceder a S3 y subir archivos de manera programática, puedes buscar este usuario en CloudTrail. Verás eventos como la solicitud de ver la ubicación de un bucket, listar el bucket y la carga de archivos.

2. **Monitoreo de actividades constantes**: Podrás notar actividades constantes como "describe instance health", que monitorean el estado de instancia como parte de Elastic Beanstalk. Esto verifica regularmente si las instancias están activas, lo cual es crucial para mantener la aplicación funcionando correctamente.

3. **Análisis de cambios en configuraciones**: Si realizas cambios en Elastic Beanstalk —por ejemplo, pasar de un solo nodo a un sistema multinodo con balanceadores de carga—, CloudTrail te mostrará detalles de estos cambios, registrando alteraciones en configuraciones y grupos de autoescalamiento, entre otras acciones.

#### ¿Qué información adicional ofrece CloudTrail sobre las instancias?

Además de las actividades programáticas, CloudTrail puede ofrecer datos sobre instancias EC2 y sus AMIs (Amazon Machine Images) específicas:

- Creación de imágenes e instancias: Podrás ver entradas que registran actividades como la creación de nuevas imágenes (AMIs) y la ejecución de instancias, todos vinculados a la cuenta raíz de AWS.

##### Recomendaciones para aprovechar CloudTrail al máximo

1. **Configura alertas personalizadas**: Utiliza CloudWatch para recibir notificaciones sobre actividades sospechosas o no autorizadas, elevando la seguridad de tu infraestructura.

2. **Auditoría regular**: Realiza auditorías periódicas de los registros para asegurarte de que todas las actividades sean esperadas y autorizadas.

3. **Automatización de respuestas**: Implementa reglas en AWS Lambda para automatizar respuestas a ciertos eventos críticos capturados por CloudTrail.

4. **Educación continua**: Mantente actualizado sobre las mejores prácticas de seguridad en AWS y las nuevas características de CloudTrail.

CloudTrail es una herramienta esencial para cualquier administrador de cuentas en AWS que quiera asegurar la total transparencia y seguridad en sus operaciones. Te aliento a explorar todas sus funcionalidades y continuar aprendiendo para maximizar la eficiencia de tu infraestructura en la nube. ¡Sigue adelante en tu camino de aprendizaje en la gestión de AWS!

## Certificación SSL y Seguridad AWS con GuardDuty

Aquí tienes una guía clara y concisa sobre **Certificación SSL** y el uso de **Amazon GuardDuty** para mejorar la seguridad en AWS:

### 🔐 Certificación SSL en AWS

### ¿Qué es una Certificación SSL?

SSL (Secure Sockets Layer) o TLS (su sucesor) es un protocolo de seguridad que cifra la comunicación entre el navegador del usuario y el servidor web. En AWS, puedes usar certificados SSL para proteger sitios web, aplicaciones y APIs.

### Opciones en AWS:

| Servicio                            | Descripción                                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------------------------- |
| **AWS Certificate Manager (ACM)**   | Provisión y administración gratuita de certificados SSL públicos para dominios verificados. |
| **Certificados privados (ACM PCA)** | Crear una Autoridad Certificadora Privada (CA) para certificados internos.                  |
| **Importar certificados**           | Puedes cargar certificados SSL/TLS propios desde otras autoridades.                         |

### Usos comunes:

* **Elastic Load Balancer (ELB)**
* **CloudFront**
* **API Gateway**
* **AWS Amplify / App Runner**
* **Custom Domains con ACM**

### ¿Cómo obtener un certificado SSL con ACM?

1. Ve a **AWS Certificate Manager > Request certificate**
2. Elige **Public certificate**
3. Ingresa tu dominio (ej. `example.com`, `*.example.com`)
4. Verifica el dominio (DNS o Email)
5. Usa el certificado en CloudFront, ELB, etc.

### 🛡️ Amazon GuardDuty: Detección Inteligente de Amenazas

### ¿Qué es Amazon GuardDuty?

Es un servicio de **detección continua de amenazas** que analiza logs de seguridad para identificar comportamientos maliciosos o no autorizados en tu cuenta de AWS.

### ¿Qué analiza GuardDuty?

| Fuente de datos               | Ejemplos analizados                             |
| ----------------------------- | ----------------------------------------------- |
| **VPC Flow Logs**             | Tráfico de red sospechoso                       |
| **AWS CloudTrail**            | Acciones API no autorizadas o inusuales         |
| **DNS Logs**                  | Consultas DNS maliciosas o inesperadas          |
| **EKS Audit Logs (opcional)** | Actividades sospechosas en clústeres Kubernetes |

### 🔎 Ejemplos de amenazas detectadas

| Tipo de amenaza                           | Descripción                                               |
| ----------------------------------------- | --------------------------------------------------------- |
| `UnauthorizedAccess:IAMUser/ConsoleLogin` | Intento de inicio de sesión sospechoso                    |
| `Recon:EC2/PortProbeUnprotectedPort`      | Escaneo de puertos en instancias EC2                      |
| `CryptoCurrency:EC2/BitcoinTool.B!DNS`    | Minería de criptomonedas detectada en EC2                 |
| `Persistence:EC2/MetadataDNSRebind`       | Técnica para acceder a metadatos EC2 mediante ataques DNS |

### 🚀 ¿Cómo habilitar GuardDuty?

1. Ve a **Amazon GuardDuty > Enable GuardDuty**
2. Selecciona la región o activa en todas las regiones
3. (Opcional) Integra con **AWS Organizations** para múltiples cuentas
4. Visualiza los hallazgos en el panel

### 🧠 Buenas prácticas combinadas

| Práctica               | Acción recomendada                                                                |
| ---------------------- | --------------------------------------------------------------------------------- |
| **Seguridad SSL**      | Usa ACM para emitir certificados y habilita HTTPS en servicios públicos           |
| **Monitoreo continuo** | Habilita GuardDuty y configura notificaciones vía Amazon SNS                      |
| **Automatización**     | Crea reglas que activen AWS Lambda ante hallazgos críticos                        |
| **Cumplimiento**       | Guarda certificados y eventos de GuardDuty en S3 con cifrado y acceso restringido |

### 📦 Extra: Integración con otros servicios

* **AWS Security Hub**: Agrega hallazgos de GuardDuty con otros servicios de seguridad.
* **Amazon Detective**: Investiga hallazgos de GuardDuty en profundidad.
* **AWS Config**: Audita configuraciones que puedan afectar la seguridad SSL.

### Resumen

Existen varias herramientas de seguridad en AWS. Vamos a ver las más importantes:

**Certificate Manager**: AWS te permite crear nuevos certificados cuando necesites (o importar alguno que ya tengas) y te sera fácil usarlos en balanceadores de cargas.

**GuardDuty**: AWS permite que hagas una auditoria constante de todos los intentos de conexiones que tienen tus equipos de computo.

## Monitoreo de Seguridad en AWS con Good Duty

Parece que te refieres a **Amazon GuardDuty** (a veces confundido como “Good Duty”). Aquí tienes una guía clara y práctica sobre el **Monitoreo de Seguridad en AWS con Amazon GuardDuty**:

### 🔍 ¿Qué es Amazon GuardDuty?

**Amazon GuardDuty** es un servicio administrado de seguridad que **detecta amenazas y actividades maliciosas** en tu cuenta de AWS. Usa inteligencia artificial, aprendizaje automático y fuentes de amenazas de AWS para **identificar comportamientos sospechosos** sin que tengas que instalar o configurar software adicional.

### 🧩 ¿Qué monitorea GuardDuty?

| Fuente de datos               | Ejemplos de lo que analiza                            |
| ----------------------------- | ----------------------------------------------------- |
| **CloudTrail**                | Actividad sospechosa con las APIs de AWS              |
| **VPC Flow Logs**             | Tráfico de red inusual o hacia direcciones maliciosas |
| **DNS Logs**                  | Consultas DNS hacia dominios maliciosos               |
| **EKS Audit Logs (opcional)** | Actividad sospechosa en clústeres Kubernetes          |

### ⚠️ Tipos de hallazgos comunes

| Categoría                 | Ejemplo de hallazgo                                    |
| ------------------------- | ------------------------------------------------------ |
| **Reconocimiento**        | `PortProbeUnprotectedPort` — escaneo de puertos        |
| **Acceso no autorizado**  | `UnauthorizedAccess:IAMUser/ConsoleLogin`              |
| **Ransomware/Minado**     | `CryptoCurrency:EC2/BitcoinTool.B!DNS`                 |
| **Persistencia**          | `EC2/MetadataDNSRebind` — acceso indebido a metadatos  |
| **Exfiltración de datos** | `S3/AnomalousBehavior` — actividad no común en buckets |

### ✅ ¿Cómo activar Amazon GuardDuty?

1. Ve a la consola de AWS.
2. Busca **GuardDuty**.
3. Haz clic en **"Enable GuardDuty"**.
4. (Opcional) Actívalo en todas las regiones y configura integración con **AWS Organizations** para múltiples cuentas.
5. Revisa los hallazgos en el panel.

### 🔐 ¿Cómo responde a amenazas?

GuardDuty **no actúa por sí solo**. Puedes automatizar respuestas con:

* **AWS Lambda**: ejecutar acciones (ej. apagar una instancia EC2 sospechosa).
* **Amazon SNS**: enviar notificaciones por email, SMS, etc.
* **AWS Security Hub**: centralizar hallazgos con otros servicios.
* **Amazon EventBridge**: crear reglas para actuar según el tipo de hallazgo.

### 🎯 Buenas prácticas

| Recomendación                                | Detalle                                       |
| -------------------------------------------- | --------------------------------------------- |
| Habilita en todas las regiones               | Para detectar amenazas globales               |
| Integra con AWS Organizations                | Monitoreo centralizado para múltiples cuentas |
| Revisa hallazgos regularmente                | Clasifica y prioriza los más críticos         |
| Automatiza respuestas con Lambda/EventBridge | Para reducir el tiempo de reacción            |
| Exporta hallazgos a S3                       | Para cumplimiento y auditoría                 |

### 🧠 Ejemplo de uso

**Escenario**: GuardDuty detecta `UnauthorizedAccess:EC2/SSHBruteForce`.

**Respuesta automática**:

1. EventBridge detecta el hallazgo.
2. Llama a una Lambda.
3. Lambda detiene la instancia comprometida y bloquea la IP en NACLs o Security Groups.

### Resumen

#### ¿Qué es Good Duty y cómo puede mejorar la seguridad en la nube de Amazon?

Good Duty es un potente sistema de Amazon diseñado para mejorar la seguridad de tus recursos en la nube. Proporciona información detallada sobre quién intenta conectarse a tus recursos, como servidores y bases de datos. Al utilizar Good Duty, puedes estar al tanto de los accesos no autorizados y las posibles amenazas a tus recursos.

####¿Cómo empezar a usar Good Duty?

Para comenzar a usar Good Duty, accede a la consola de Amazon Web Services (AWS) y sigue estos pasos básicos:

1. Escribe `GuardDuty` en la barra de búsqueda de la consola de AWS y selecciona el servicio.
2. Haz clic en "Get Started" para activar Good Duty.
3. Otorga los permisos necesarios para que Good Duty pueda monitorear las actividades de tu cuenta.

Al iniciar Good Duty, no verás información inmediatamente. Sin embargo, el sistema comenzará a escanear tus sistemas y conexiones para ofrecerte datos relevantes sobre actividades sospechosas.

#### ¿Qué tipo de ataques pueden detectarse con Good Duty?

Good Duty es eficaz para identificar varios tipos de ataques, incluyendo:

- **Escaneo de puertos**: Identificación de intentos de escaneo de puertos abiertos, lo cual puede indicar preparaciones para un ataque.

- **Ataques de fuerza bruta**: Detección de ataques por diccionario que buscan acceder a través de SSH (Secure Shell). Amazon protege las conexiones SSH mediante el uso de claves, por lo que los ataques de fuerza bruta son inútiles sin la clave correcta.

- **Script kiddies**: Identificación de intentos de acceso automatizados a través de scripts, conocidos como ataques de "script kiddie", que prueban la seguridad del sistema buscando vulnerabilidades.

#### ¿Qué hacer en caso de detectar un ataque con problemas de alta severidad?

Si Good Duty detecta un ataque con alta severidad, actúa rápidamente siguiendo estas recomendaciones:

1. **Pausar la instancia afectada**: Evalúa la situación sin interrupciones adicionales.
2. **Revisar el respaldo más reciente**: Asegúrate de que tus datos estén protegidos.
3. **Eliminar la IP pública afectada**: Para prevenir más ataques, retira cualquier IP pública comprometida.
4. **Crear una nueva instancia segura**: Utiliza la IP interna para conectar y restaurar la seguridad desde dentro de Amazon si es necesario.
5. **Realizar un análisis detallado**: Determina el alcance del ataque y realiza un seguimiento de las actividades sospechosas.

#### ¿Con qué frecuencia debes revisar los hallazgos de Good Duty?

Para mantener la seguridad de tus recursos en Amazon, revisa periódicamente los hallazgos de Good Duty. Puedes optar por un seguimiento semanal o mensual, dependiendo de tu nivel de uso y las amenazas potenciales que enfrentas. Este hábito de revisión regular te ayudará a:

- Restablecer snapshots de seguridad antes de cualquier intrusión exitosa.
- Entender la naturaleza de los ataques y responder oportunamente a las amenazas.

Al mantener Good Duty bajo constante supervisión, garantizas la protección continua de tus recursos y te aseguras de tomar medidas proactivas para mitigar posibles ataques. Por último, archiva los hallazgos una vez revisados para mantener la claridad en momentos de auditoría y revisión futuras.

## Análisis de Imágenes con Amazon Rekognition

Amazon Rekognition es un servicio de inteligencia artificial de AWS que permite analizar imágenes y videos utilizando aprendizaje profundo. A continuación te presento un resumen detallado sobre **Análisis de Imágenes con Amazon Rekognition**:

### 🔍 Análisis de Imágenes con Amazon Rekognition

### ¿Qué es Amazon Rekognition?

Amazon Rekognition es un servicio de análisis de imágenes y videos que detecta objetos, escenas, texto, caras, emociones y actividades, sin necesidad de experiencia previa en machine learning.

### 🧠 Funciones Principales en Análisis de Imágenes

1. **Detección de Objetos y Escenas**

   * Identifica objetos comunes (vehículos, animales, armas, etc.)
   * Reconoce contextos como “playa”, “ciudad”, “oficina”, etc.

2. **Detección y Análisis Facial**

   * Detecta rostros humanos en una imagen.
   * Analiza emociones (feliz, triste, sorprendido, etc.)
   * Determina características faciales como género, apertura de ojos o boca, uso de gafas, barba, etc.

3. **Comparación de Rostros**

   * Compara dos imágenes para saber si son de la misma persona.
   * Útil en control de acceso, autenticación o sistemas de vigilancia.

4. **Reconocimiento de Texto (OCR)**

   * Extrae texto de imágenes (carteles, documentos, matrículas).
   * Soporta múltiples idiomas.

5. **Moderación de Contenido**

   * Detecta contenido explícito o potencialmente ofensivo.
   * Ideal para redes sociales, aplicaciones de usuario, etc.

6. **Etiquetado Automático**

   * Asigna etiquetas automáticas a una imagen para su clasificación.

### 🛠️ ¿Cómo Funciona?

1. **Carga la imagen** a través de la consola AWS, SDK o API.
2. El servicio analiza la imagen con modelos de deep learning preentrenados.
3. Devuelve un JSON con la información detectada: etiquetas, rostros, texto, etc.

### 🧪 Ejemplo de Análisis de Imagen

```json
{
  "Labels": [
    {
      "Name": "Dog",
      "Confidence": 98.5
    },
    {
      "Name": "Pet",
      "Confidence": 92.1
    }
  ],
  "Faces": [
    {
      "Gender": "Male",
      "Emotions": [
        {
          "Type": "HAPPY",
          "Confidence": 99.0
        }
      ]
    }
  ]
}
```

### 🔐 Seguridad y Privacidad

* Compatible con AWS IAM para control de accesos.
* Cifrado en tránsito (SSL/TLS) y en reposo.
* No almacena imágenes de forma permanente por defecto.

### 💡 Casos de Uso

| Caso de uso              | Aplicación                                   |
| ------------------------ | -------------------------------------------- |
| Seguridad y vigilancia   | Identificación facial en entradas/salidas    |
| Comercio electrónico     | Búsqueda de productos por imagen             |
| Redes sociales           | Moderación de imágenes y contenido           |
| Recursos Humanos / RRHH  | Validación de identidad                      |
| Medios y entretenimiento | Clasificación y búsqueda de contenido visual |

### 🧩 Integraciones

* Puede integrarse con **S3** (almacenamiento de imágenes), **Lambda** (procesamiento automático), **SNS/SQS** (notificaciones), y otros servicios de AWS.

### 💵 Precios

Amazon Rekognition cobra por número de imágenes procesadas y tipo de análisis (detección de etiquetas, rostros, texto, etc.). Tiene una capa gratuita limitada útil para pruebas iniciales.

### Resumen

#### ¿Qué es Amazon Recognition y cómo puede ser utilizado?

Amazon Recognition es un servicio de inteligencia artificial basado en deep learning. Esta herramienta es capaz de detectar personas, objetos, acciones, celebridades y tipos de actividades en fotografías. Entre sus múltiples aplicaciones, se incluye la moderación automatizada de imágenes, detección de objetos, análisis facial para conocer las emociones de las personas, y el reconocimiento de textos presentes en las imágenes.

#### ¿Cuáles son las funcionalidades clave de Amazon Recognition?

- **Moderación de contenido**: Permite identificar y filtrar contenido inapropiado, como desnudos, en imágenes.
- **Detección de objetos y escenas**: Identifica objetos y escenas en fotografías, como deportes, ciudades, o naturaleza.
- **Análisis facial**: Detecta emociones en los rostros como felicidad, tristeza o enfado. Esto se podría utilizar para analizar las reacciones de usuarios a ciertos estímulos o servicios.
- **Reconocimiento de celebridades**: Identifica figuras públicas en imágenes, lo cual es útil en sistemas de archivos multimedia.
- **Detección de texto**: Permite extraer texto de las imágenes, facilitando la conversión de los contenidos visuales a formatos editables.

#### ¿Cómo funciona el demo de Amazon Recognition?

Al probar el demo de Amazon Recognition, uno puede experimentar con imágenes de ejemplo o subir sus propias fotos. El sistema ofrece descripciones detalladas de cada imagen, identificando aspectos como deportes, tipos de entorno urbano, emociones faciales, entre otros. Esta funcionalidad es rápida, ofreciendo resultados casi instantáneamente.

Por ejemplo, en una imagen de alguien practicando skateboarding, el resultado incluye reconocimiento del deporte y el número de personas. En otro caso, una foto de la ciudad puede listar elementos como edificios y áreas urbanas.

#### ¿Qué ventajas ofrece el uso de SDKs con Amazon Recognition?

Las SDKs (Software Development Kits) ofrecen la posibilidad de integrar Amazon Recognition en tus propias aplicaciones. Esto significa que puedes llamar a las funciones de Amazon Recognition desde cualquier programa desarrollado en lenguajes como .NET, Java o PHP. El servicio te responde con un JSON que contiene información detallada sobre el contenido de la imagen, como la presencia de personas, vehículos, marcas específicas y más.

Algunas aplicaciones prácticas incluyen:

- **Automatización de procesos de reconocimiento**: Analizar grandes volúmenes de imágenes y clasificar contenido de manera eficiente.
- **Mejora de la experiencia del usuario**: Personalizar experiencias de usuario según sus reacciones detectadas en tiempo real.
- **Seguridad y control de acceso**: Podría utilizarse para sistemas de seguridad que detectan rostros y emociones en tiempo real.

Amazon Recognition es, sin duda, una herramienta poderosa con aplicaciones diversas que facilitan tanto la automatización como el análisis de contenido visual. Con su ayuda, es posible transformar la manera en que interactuamos y entendemos el mundo visual que nos rodea. Si estás interesado en la inteligencia artificial y el procesamiento de imágenes, explorar Amazon Recognition podría aportar grandes beneficios a tus proyectos.

## Creación y Configuración de Postgrex en Amazon RDS

Aquí tienes una guía detallada para la **Creación y Configuración de PostgreSQL en Amazon RDS** (a menudo se le llama "Postgrex" de forma informal, pero el nombre correcto del motor es **PostgreSQL**):

### 🛠️ Creación y Configuración de PostgreSQL en Amazon RDS

### 📌 ¿Qué es Amazon RDS?

Amazon RDS (Relational Database Service) es un servicio administrado por AWS para bases de datos relacionales como PostgreSQL, MySQL, MariaDB, Oracle y SQL Server.

### ✅ Paso a Paso para Crear PostgreSQL en RDS

#### 1. **Ingresar a la Consola de AWS**

* Ir a: [https://console.aws.amazon.com](https://console.aws.amazon.com)
* Buscar "RDS" en el buscador superior.

#### 2. **Crear una Nueva Instancia**

* Clic en **"Create database"**.
* Elegir método: `Standard create`.
* Elegir motor de base de datos: **PostgreSQL**.
* Seleccionar la versión de PostgreSQL que desees (por ejemplo, 15.3).

#### 3. **Configurar la Instancia**

* **DB instance identifier**: Un nombre para tu instancia (ej. `pg-rds-demo`).
* **Master username**: Nombre del usuario administrador (ej. `admin`).
* **Master password**: Contraseña segura.
* Puedes habilitar la autenticación IAM si deseas manejar usuarios con roles de AWS.

#### 4. **Seleccionar Tipo de Instancia**

* Clase de instancia: ej. `db.t3.micro` (apto para pruebas, elegible para capa gratuita).
* Almacenamiento: general SSD (gp2 o gp3).
* Configura el tamaño inicial del disco y habilita el escalado si lo deseas.

#### 5. **Configuración de Conectividad**

* VPC: usa una VPC existente o crea una nueva.
* Subnet group y zona de disponibilidad: puedes dejar los valores predeterminados.
* **Public access**: Elige “Yes” si deseas acceder desde internet, o “No” si será privada.
* Configura el **grupo de seguridad** para permitir tráfico desde tu IP pública en el puerto **5432** (PostgreSQL).

#### 6. **Opciones Adicionales**

* Habilitar backups automáticos.
* Configurar monitoreo con CloudWatch.
* Activar mantenimiento automático si deseas actualizaciones programadas.

#### 7. **Crear la Base de Datos**

* Clic en “Create database”.
* Espera unos minutos hasta que el estado esté en **“Available”**.

### 🔌 Conexión a PostgreSQL en RDS

#### Desde línea de comandos:

```bash
psql -h tu-endpoint.rds.amazonaws.com -U admin -d postgres
```

#### Desde Python con `psycopg2`:

```python
import psycopg2

conn = psycopg2.connect(
    host="tu-endpoint.rds.amazonaws.com",
    database="postgres",
    user="admin",
    password="tu-contraseña"
)
```

### 🔒 Seguridad Recomendaciones

* Usa **grupos de seguridad** para restringir IPs.
* Cifra en tránsito con **SSL/TLS**.
* Habilita **cifrado en reposo** (KMS).
* Configura copias automáticas y **multi-AZ** para alta disponibilidad.

### 📊 Buenas Prácticas

* Habilita métricas de CloudWatch (CPU, IOPS, conexiones).
* Usa parámetros personalizados con **parameter groups** si necesitas cambiar configuraciones como `work_mem`, `max_connections`, etc.
* Realiza pruebas de rendimiento y escalabilidad si tu carga de trabajo crecerá.

### Resumen

#### ¿Cómo crear un servicio de PostgreSQL en RDS?

Si estás buscando una manera sencilla y efectiva de gestionar bases de datos a través de la nube, utilizar RDS de Amazon para crear un servicio de PostgreSQL puede ser una excelente opción. A continuación, te guiaré a través del proceso para configurar y crear una instancia de PostgreSQL en RDS, asegurando que puedas aprovechar al máximo las características de este servicio.

#### ¿Cómo comienzo con la configuración de RDS en Amazon?

Para iniciar, lo primero que deberás hacer es acceder a la consola de Amazon. Busca el servicio RDS en la consola. Esto te redirigirá a una pantalla donde podrás empezar a configurar una nueva instancia de base de datos.

Una recomendación útil si estás comenzando y deseas experimentar sin incurrir en costos adicionales es seleccionar las opciones gratuitas que Amazon ofrece. Esto te permitirá familiarizarte con el servicio sin preocupaciones financieras.

#### ¿Cuál es la importancia de escoger la versión correcta de PostgreSQL?

La elección de la versión de PostgreSQL es crucial, especialmente si ya cuentas con una base de datos existente que buscas migrar. Asegúrate de que la versión que selecciones en RDS sea compatible con la que ya tienes, lo que facilitará el proceso de migración y evitará problemas de compatibilidad. Si estás comenzando una nueva base de datos, opta por la versión más reciente compatible con tus necesidades.

#### ¿Cómo configurar los detalles de la instancia de base de datos?

Una vez elegida la versión, deberás configurar los detalles básicos de la instancia:

1. **Nombre de la instancia y base de datos**: Se recomienda utilizar nombres fáciles de recordar y consistentes. Por ejemplo, "testplatzi2".
2. **Nombre del usuario maestro**: Utiliza un nombre sencillo, idéntico o similar al de la instancia, para ahorrar tiempo y evitar confusiones.
3. **Password**: Para garantizar la seguridad, usa un generador de contraseñas para crear un password fuerte y apúntalo en un lugar seguro.

Esta planificación meticulosa te ayudará a evitar confusiones futuras y garantizará que los datos sensibles estén protegidos.

#### ¿Qué configuraciones adicionales están disponibles?

Amazon RDS ofrece una serie de configuraciones avanzadas que puedes considerar:

- **Cifrado**: Aunque e inicialmente puede no estar disponible en el modo gratuito, el cifrado es esencial cuando deseas proteger tus datos en entornos de producción.
- **Copias de seguridad**: Configura la frecuencia de backups según tus necesidades para proteger tus datos.
- **Monitoreo**: Decide si necesitas monitorear la base de datos y configurar alertas según el rendimiento o requisitos específicos.

Luego de ajustar estas configuraciones, procede a crear la base de datos. RDS te confirmará la creación y te permitirá ver los detalles de la instancia.

#### ¿Cómo conectar la base de datos a otras instancias de Amazon o desde tu computadora?

Inicialmente, la conexión solo será posible desde instancias creadas dentro de Amazon. Sin embargo, puedes modificar la configuración para permitir conexiones desde tu computadora local o cualquier otra ubicación externa. Este paso es esencial para desarrollar aplicaciones que dependan de la base de datos y para el acceso conveniente de desarrolladores y administradores desde múltiples ubicaciones.

Con estos conocimientos, estarás preparado para implementar y gestionar un servicio de PostgreSQL en Amazon RDS de manera eficiente. Si tienes curiosidad o necesitas asistencia adicional, continúa explorando recursos y herramientas que te ayuden a expandir tu dominio en bases de datos en la nube. ¡Feliz aprendizaje y buena suerte en tus proyectos!

[Generador de password](https://getmypassword.com/)

## Conexión y Gestión de Datos en RDS con PostgreSQL

Con gusto, aquí tienes un resumen claro y práctico sobre **Conexión y Gestión de Datos en Amazon RDS con PostgreSQL**:

### 🔗 **Conexión y Gestión de Datos en RDS con PostgreSQL**

### 🛠️ 1. **Creación de la Instancia RDS (PostgreSQL)**

Desde la consola de AWS:

* Ve a **RDS > Crear base de datos**
* Elige **PostgreSQL**
* Selecciona “**Solo configuración básica**” (si es para pruebas)
* Define:

  * Usuario administrador (`masterusername`)
  * Contraseña
  * Puerto (por defecto 5432)
  * Nombre de la base de datos inicial (opcional)
* Asegúrate de:

  * Habilitar el acceso público si vas a conectarte desde tu PC
  * Seleccionar un grupo de seguridad que permita conexiones entrantes en el puerto 5432

### 💻 2. **Conexión desde tu PC**

#### 🧱 Requisitos:

* Tener instalado `psql` (cliente de PostgreSQL)
* Tener la IP pública o DNS de la instancia RDS

#### 🧪 Comando para conectarse:

```bash
psql -h <host> -U <usuario> -d <nombre_basedatos> -p 5432
```

**Ejemplo:**

```bash
psql -h database-1.abc123xyz.us-east-1.rds.amazonaws.com -U admin -d postgres -p 5432
```

> Si no creaste una base específica, usa `postgres` como nombre.

### 🔐 3. **Seguridad y acceso**

* Revisa el grupo de seguridad (Security Group) asociado a la instancia RDS:

  * Asegúrate de tener una regla de entrada que permita el tráfico al **puerto 5432** desde tu IP pública.
* Revisa que la opción **"acceso público"** esté habilitada.

### 📂 4. **Gestión de Datos**

#### ✅ Crear tabla:

```sql
CREATE TABLE empleados (
    id SERIAL PRIMARY KEY,
    nombre TEXT,
    cargo TEXT,
    salario NUMERIC
);
```

#### ✅ Insertar datos:

```sql
INSERT INTO empleados (nombre, cargo, salario) VALUES
('Mario Vargas', 'Ingeniero', 4200),
('Ana Pérez', 'Diseñadora', 3700);
```

#### ✅ Consultar datos:

```sql
SELECT * FROM empleados;
```

#### ✅ Exportar/Importar datos:

* **Importar archivo SQL:**

```bash
psql -h <host> -U <usuario> -d <bd> -f archivo.sql
```

* **Exportar (dump):**

```bash
pg_dump -h <host> -U <usuario> -d <bd> > respaldo.sql
```

### 🧹 5. **Buenas prácticas**

* **Habilita backups automáticos**
* **Monitorea el rendimiento con CloudWatch**
* **Activa alertas de espacio y carga**
* **Usa roles IAM si accedes desde Lambda o EC2**
* **Configura mantenimiento automático en horarios nocturnos**

### Resumen

#### ¿Cómo conectarse a una instancia de RDS en Postgres?

Conectar a una instancia de RDS (Amazon Relational Database Service) en Postgres puede parecer complicado al principio, pero con los pasos correctos, esta tarea se vuelve sencilla. En este apartado, aprenderás cómo realizar esta conexión y a verificar los datos necesarios para lograrlo.

Para empezar, accede a la consola de Amazon y navega hasta tu instancia de RDS. Aquí, en la sección "Connect", encontrarás un "endpoint", que es esencial para tu conexión.

1. **Descarga de Software**:

 - Descarga e instala PGAdmin, una herramienta gráfica de administración para bases de datos Postgres.

2. **Configuración en PGAdmin**:

 - Usa el nombre de tu instancia, nombre de usuario y contraseña que configuraste en Amazon RDS.
 - Copia el "endpoint" obtenido de la consola y configúralo como el "host" en PGAdmin.
 - Asegúrate de usar el puerto correcto (generalmente 5432) y la base de datos a la que deseas conectarte.

3. **Errores comunes al conectar**:

 - Si experimentas errores, una causa frecuente es que el servidor no está escuchando en el puerto indicado.
 
#### ¿Cómo modificar la configuración de tu instancia para permitir conexiones?

Para facilitar las conexiones externas a tu instancia de RDS, es necesario modificar ciertos parámetros que inicialmente restringen las conexiones solo a la red de Amazon.

1. **Hacer pública la instancia**:

 - En la consola de RDS, selecciona la instancia y elige la opción de modificar.
 - Cambia la configuración para que la instancia sea accesible públicamente.

2. **Editar el grupo de seguridad**:

 - Utiliza el grupo de seguridad de tu instancia para permitir conexiones desde direcciones IP externas.
 - En la sección "Inbound", modifica las reglas para permitir conexiones desde cualquier IP o limita a direcciones específicas para mayor seguridad.

3. **Monitorear cambios**:

 - Recarga la página de la consola para verificar que los cambios están aplicados y que la base de datos es ahora accesible.

#### ¿Cómo crear y administrar tablas con SQL en PGAdmin?

Una vez establecida la conexión con tu base de datos, el siguiente paso es comenzar a crear y administrar tablas, lo cual se realiza utilizando SQL (Structured Query Language).

1. **Apertura del editor SQL**:

 - Utiliza el ícono de SQL en PGAdmin para abrir una ventana donde puedas introducir comandos y ejecutar queries.

2. **Creación de tablas**:

 - Copia y pega el script SQL necesario para crear una nueva tabla. Un ejemplo básico:
 
```sql
CREATE TABLE films (
    code CHAR(5),
    title VARCHAR(40),
    did INTEGER,
    date_prod DATE,
    kind VARCHAR(10),
    len INTERVAL HOUR TO MINUTE
);
```

3. **Inserción de datos**:

 - Una vez creada la tabla, inserta datos usando el siguiente tipo de query:
 
```sql
INSERT INTO films (code, title, did, date_prod, kind, len)
VALUES ('12345', 'Nombre de la película', 1, '2023-10-12', 'Drama', '02:00');
```

4. **Consulta de datos**:

 - Para verificar que los datos se insertaron correctamente, usa un SELECT:
 
`SELECT * FROM films;`

Recuerda que la práctica constante y la familiarización con SQL te ayudarán a gestionar eficientemente tus bases de datos en RDS. No dudes en consultar recursos adicionales o tutoriales más detallados para seguir avanzando en tus habilidades. ¡Sigue adelante y mejora tu dominio sobre bases de datos!

## Fundamentos de Amazon: Opciones y Servicios Básicos

Claro, aquí tienes una guía clara y resumida sobre los **Fundamentos de Amazon Web Services (AWS): Opciones y Servicios Básicos**:

### 🧭 **Fundamentos de Amazon AWS: Opciones y Servicios Básicos**

### ☁️ ¿Qué es AWS?

Amazon Web Services (AWS) es la plataforma de computación en la nube de Amazon. Proporciona más de 200 servicios integrales de infraestructura y plataforma bajo demanda, pagando solo por lo que usas.

---

### 🧱 **Principales Categorías de Servicios**

| Categoría                     | Servicio Clave | Descripción breve                                     |
| ----------------------------- | -------------- | ----------------------------------------------------- |
| **Cómputo**                   | **EC2**        | Servidores virtuales escalables                       |
|                               | Lambda         | Computación sin servidor (serverless)                 |
| **Almacenamiento**            | S3             | Almacenamiento de objetos (archivos)                  |
|                               | EBS            | Discos duros para EC2                                 |
|                               | Glacier        | Almacenamiento a largo plazo y bajo costo             |
| **Bases de Datos**            | RDS            | Bases de datos relacionales (MySQL, PostgreSQL, etc.) |
|                               | DynamoDB       | Base de datos NoSQL rápida y escalable                |
| **Red y Entrega**             | VPC            | Red privada virtual                                   |
|                               | Route 53       | Sistema DNS y gestión de dominios                     |
|                               | CloudFront     | Red de distribución de contenido (CDN)                |
| **Gestión de Usuarios**       | IAM            | Gestión de usuarios, roles y políticas de acceso      |
| **Herramientas de Monitoreo** | CloudWatch     | Supervisión y métricas de servicios                   |
|                               | CloudTrail     | Auditoría y registro de actividades de cuenta         |

### 🔐 **Seguridad y Acceso**

* **IAM (Identity and Access Management)**: Define usuarios, permisos y políticas para acceso controlado.
* Autenticación multifactor (MFA)
* Políticas de acceso granular (por servicio, acción, recurso)

### 🧰 **Herramientas de Gestión**

* **AWS Management Console**: Interfaz gráfica web.
* **AWS CLI**: Línea de comandos para automatización.
* **AWS SDKs**: Librerías para programar en Python, Node.js, Java, etc.

### 🧪 **Modelos de Uso Común**

| Escenario           | Servicios involucrados     |
| ------------------- | -------------------------- |
| Sitio web estático  | S3 + CloudFront + Route 53 |
| Web app dinámica    | EC2 / Lambda + RDS + S3    |
| Big Data / Análisis | EMR, Athena, Redshift      |
| IoT                 | AWS IoT Core               |
| Machine Learning    | SageMaker, Rekognition     |

### 📈 **Ventajas de AWS**

* Escalabilidad automática
* Alta disponibilidad y redundancia
* Paga solo por lo que usas
* Seguridad de nivel empresarial
* Presencia global con regiones y zonas de disponibilidad

### Resumen

####¿Qué aprendimos en el curso básico de Amazon?

El curso que acabamos de terminar nos proporcionó una introducción esencial a los fundamentos de Amazon como plataforma. Con más de cincuenta servicios y opciones disponibles, Amazon es un ecosistema en constante evolución, y este curso fue solo el primer paso hacia un entendimiento más amplio.

#### ¿Qué sigue después de este curso?

Amazon es una plataforma vasta y compleja, y este curso apenas desentrañó la superficie de lo que se puede lograr con ella. Planificamos continuar con más cursos que profundizarán en temas específicos y funcionalidades avanzadas. Estos futuros cursos explorarán:

- Servicios adicionales que Amazon ofrece.
- Mejoras en la integración de servicios para optimizar operaciones.
- Cómo adaptarse a las constantes actualizaciones tecnológicas de Amazon.

#### ¿Cómo podemos mejorar juntos?

Estamos comprometidos en el aprendizaje continuo y en la mejora de nuestros cursos. Tu participación y feedback son fundamentales para nosotros. Aquí algunas formas en las que puedes contribuir:

- **Comentarios**: No dudes en dejar tus preguntas, dudas, o sugerencias de mejora en la sección de comentarios de nuestro curso. Estamos aquí para ayudarte y para asegurarnos de que tengas una experiencia de aprendizaje efectiva.
- **Sugerencias de contenido**: Cuéntanos qué servicios de Amazon te gustaría explorar en profundidad en futuros cursos. Tu opinión nos ayuda a diseñar contenido relevante y útil.
- **Redes sociales**: Comparte tus logros y la obtención de tu certificado con nosotros en redes como Twitter o Facebook.

#### ¿Por qué es importante seguir aprendiendo?

El mundo digital y las plataformas como Amazon están en constante cambio. Aprender y adaptarse a estas evoluciones es crucial para mantenerse competitivo. Al continuar con tu educación en esta área, te aseguras de:

- Estar al día con las últimas tendencias y novedades tecnológicas.
- Potenciar tu perfil profesional con habilidades actualizadas.
- Aprovechar nuevas oportunidades que surgen con cada innovación tecnológica.

Te animamos a que sigas aprendiendo con nosotros, aportes tus ideas y, en conjunto, descubramos todo lo que Amazon tiene para ofrecer. ¡Emprendamos este viaje de conocimiento juntos!