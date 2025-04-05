# Curso de Serverless Framework en AWS

## Bienvenida al curso de Serverless Framework

El **Serverless Framework** es una herramienta de desarrollo open-source que facilita la creación, despliegue y administración de aplicaciones **serverless** en plataformas como **AWS Lambda**, **Azure Functions**, **Google Cloud Functions**, entre otras.

### 🚀 ¿Qué es el Serverless Framework?

Es un **framework de infraestructura como código (IaC)** que te permite:

- Crear funciones Lambda y recursos de infraestructura desde un archivo de configuración (`serverless.yml`)
- Desplegar tu aplicación con un solo comando (`sls deploy`)
- Integrarte fácilmente con servicios como API Gateway, DynamoDB, S3, etc.
- Gestionar múltiples entornos (dev, test, prod)
- Monitorear y depurar tus funciones en la nube

### 🧱 Arquitectura típica

Una app usando Serverless Framework puede incluir:

- **Funciones Lambda** para lógica de negocio
- **API Gateway** para exponer endpoints HTTP
- **DynamoDB** o RDS como base de datos
- **S3** para almacenamiento de archivos
- **IAM Roles** configurados automáticamente

### 📁 Ejemplo básico (`serverless.yml` con AWS)

```yaml
service: mi-api-serverless

provider:
  name: aws
  runtime: nodejs18.x
  region: us-east-1

functions:
  hola:
    handler: handler.hola
    events:
      - http:
          path: hola
          method: get
```

```js
// handler.js
module.exports.hola = async (event) => {
  return {
    statusCode: 200,
    body: JSON.stringify({ mensaje: "¡Hola mundo serverless!" }),
  };
};
```

### ✅ Ventajas

- ⚡ **Despliegue rápido**
- 🧩 Soporte para múltiples cloud providers
- 🔧 Reutilizable y modular
- 📊 Integración con monitoreo y métricas
- 🔒 Buen manejo de seguridad y permisos vía IAM

### 📦 Instalación

```bash
npm install -g serverless
```

### 🧪 Comandos útiles

| Comando | Descripción |
|--------|-------------|
| `sls create --template aws-nodejs` | Crea un nuevo proyecto |
| `sls deploy` | Despliega la aplicación |
| `sls invoke -f hola` | Ejecuta una función |
| `sls logs -f hola` | Muestra logs de la función |
| `sls remove` | Elimina los recursos desplegados |

### Resumen

#### ¿Qué es Serverless Framework y cómo puede beneficiar tus proyectos?

Serverless Framework es una poderosa herramienta de desarrollo y despliegue que revoluciona la manera en que construimos aplicaciones en la nube. Facilitando la eliminación de la gestión de infraestructura tradicional, permite a los desarrolladores centrarse en escribir código y mejorar las funcionalidades de sus aplicaciones. Al combinarlo con Amazon Web Services (AWS), obtienes una solución escalable y eficiente que optimiza el proceso de desarrollo. Además, gracias a la eliminación de servidores fijos, se logra una utilización más efectiva y adaptativa de los recursos, lo que se traduce en un costo menor y mejor aprovechamiento del entorno cloud.

#### ¿Cuáles son los pilares del ecosistema serverless en AWS?

AWS ofrece un conjunto completo de servicios que complementan el paradigma serverless. Aquí te mostramos algunas de las piezas clave para configurar tu entorno:

- **AWS Lambda**: Ejecuta tu código en respuesta a eventos sin la necesidad de aprovisionar o administrar servidores.
- **API Gateway**: Facilita la creación, publicación, mantenimiento, monitoreo y protección de API a cualquier escala.
- **AWS DynamoDB**: Un servicio de base de datos NoSQL rápido y flexible adecuado para aplicaciones serverless.

Este ecosistema permite a los desarrolladores innovar y escalar sus aplicaciones sin preocuparse por el mantenimiento de los servidores subyacentes. Además, conocer y dominar estas herramientas es esencial para cualquier Cloud Developer que busque destacar en el mundo de la computación en la nube.

#### ¿Cómo ayuda este curso a mejorar tus habilidades como Cloud Developer?

Este curso está diseñado para llevar tus habilidades al siguiente nivel. No solo vas a aprender a usar Serverless Framework para desarrollar aplicaciones, sino que también descubrirás cómo automatizar y optimizar tus procesos de desarrollo. Además, vas a adquirir conocimientos prácticos que incluyen:

- Construcción de aplicaciones CRUD que puedes añadir a tu portafolio.
- Implementaciones avanzadas que superan el desarrollo básico, incorporando automatizaciones y mejoras.
- Consejos esenciales sobre cómo integrar otras tecnologías del mundo serverless, como Docker y Kubernetes, lo que aumenta tu versatilidad como Cloud Engineer.

#### ¿Qué puedes esperar al final del curso?

Al concluir el curso, estarás preparado para mostrar un abanico de capacidades que no solo comprende la creación de aplicaciones serverless, sino también su gestión y optimización. Las habilidades adquiridas te permitirán ofrecer soluciones completas a problemas complejos en la nube, elevando tus oportunidades profesionales y diferenciándote en la industria tecnológica.

Por lo tanto, este curso no es solo una introducción a Serverless, sino un paso hacia convertirte en un experto en AWS, listo para implementar mejoras significativas en tus proyectos. ¡Anímate a explorar el futuro del desarrollo en la nube y lleva tus competencias a nuevos horizontes!

## Presentación de proyecto

### ¿Qué es serverless en AWS?

El enfoque de computación serverless en AWS ofrece muchas ventajas para el desarrollo de aplicaciones escalables, resilientes y con alta concurrencia. En este curso, aprenderás sobre componentes clave de este ecosistema y cómo integrarlos eficazmente para lograr un diseño óptimo y bien arquitecturado. AWS provee servicios como API Gateway, Lambda Functions y DynamoDB, que son esenciales para el desarrollo de aplicaciones bajo este esquema.

### ¿Cómo están organizados los componentes en un proyecto serverless?
#### ¿Qué papel juega el API Gateway?

API Gateway actúa como el intermediario que recibe las solicitudes de los usuarios y las envía a las funciones Lambda correspondientes. Es crucial en el diseño serverless ya que gestiona las peticiones y las respuestas, facilitando una comunicación estable y segura entre el cliente y la lógica del servidor.

### ¿Qué son las funciones Lambda?

Las funciones Lambda son el núcleo de la lógica de nuestra aplicación serverless. Estas funciones son responsables de manejar acciones específicas como crear, leer, actualizar y eliminar, típicamente denominadas operaciones CRUD. Una de las ventajas de las funciones Lambda es su independencia del lenguaje de programación, permitiendo el uso de Python, JavaScript, entre otros, bajo el framework serverless.

### ¿Cuál es el rol de DynamoDB?

DynamoDB es el servicio de almacenamiento que integra este ecosistema. Diseñado para manejar grandes cantidades de datos de una manera rápida y eficiente, es el componente que persiste la información que manipulan nuestras funciones Lambda.

### ¿Por qué es importante el AWS Well-Architected Framework?

El AWS Well-Architected Framework proporciona un conjunto de mejores prácticas para el diseño de aplicaciones seguras, confiables y eficientes. Entre sus pilares se encuentran:

- **Excelencia operacional**: Garantiza que las operaciones se realicen de manera adecuada y replicable.
- **Seguridad**: Asegura que los datos y sistemas estén protegidos contra amenazas.
- **Fiabilidad**: Garantiza que el sistema pueda recuperarse de fallos y adaptarse a cambios.
- **Eficiencia en el rendimiento**: Optimiza el uso de recursos para lograr el máximo rendimiento.
- **Optimización de costos**: Controla el gasto, maximizando el valor económico del sistema.
- **Sostenibilidad**: Asegura el diseño ecoamigable del sistema.

Cumplir con estos pilares mejora significativamente el diseño y mantenimiento de aplicaciones serverless en AWS.

### ¿Qué consideraciones de seguridad se deben tener en cuenta?

En un entorno serverless, aspectos de seguridad como la gestión de puertos y la exposición de aplicaciones a Internet requieren atención. AWS ofrece herramientas y configuraciones que ayudan a asegurar estos puntos críticos, como la protección de puertos y el establecimiento de permisos adecuados para cada componente.

### ¿Cuál es el próximo paso en el curso?

En la siguiente lección, profundizaremos en los conceptos clave y términos específicos relacionados con serverless, el framework serverless y el AWS Well-Architected Framework. Esta comprensión detallada te permitirá aplicar con éxito estos principios en el proyecto del curso, asegurando un desarrollo eficiente y moderno de tus aplicaciones. ¡Sigue aprendiendo y motivándote para dominar serverless en AWS!