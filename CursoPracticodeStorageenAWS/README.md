# Curso Práctico de Storage en AWS

## Introducción al storage en AWS

AWS ofrece una amplia gama de servicios de almacenamiento diseñados para satisfacer diversas necesidades de negocio, desde el almacenamiento de objetos hasta sistemas de archivos distribuidos y almacenamiento en bloque. Estos servicios permiten a las empresas almacenar, gestionar y respaldar sus datos de forma escalable, segura y rentable.

### **1. Almacenamiento de Objetos: Amazon S3**  
- **Amazon S3 (Simple Storage Service)** es un servicio de almacenamiento de objetos ideal para almacenar y recuperar grandes cantidades de datos no estructurados (imágenes, videos, archivos de respaldo, logs, etc.).  
- Ofrece alta durabilidad, disponibilidad y escalabilidad.  
- Se integra con otros servicios de AWS para análisis, machine learning y distribución de contenido.

### **2. Almacenamiento en Bloque: Amazon EBS**  
- **Amazon EBS (Elastic Block Store)** proporciona almacenamiento en bloque persistente para instancias EC2.  
- Es ideal para bases de datos, sistemas de archivos y aplicaciones que requieren almacenamiento de bajo nivel con alta performance.  
- Ofrece diferentes tipos de volúmenes (SSD, HDD) para optimizar costos y rendimiento según la carga de trabajo.

### **3. Almacenamiento de Archivos: Amazon EFS y Amazon FSx**  
- **Amazon EFS (Elastic File System)** ofrece un sistema de archivos escalable y compartido para múltiples instancias EC2, ideal para aplicaciones basadas en Linux.  
- **Amazon FSx** proporciona soluciones de almacenamiento de archivos específicas:  
  - **FSx for Windows File Server:** Diseñado para entornos Windows, ideal para aplicaciones empresariales que dependen de un sistema de archivos Windows.  
  - **FSx for Lustre:** Optimizado para cargas de trabajo intensivas en datos, como análisis de big data y computación de alto rendimiento (HPC).

### **4. Almacenamiento de Archivos a Largo Plazo: Amazon Glacier y S3 Glacier Deep Archive**  
- **Amazon Glacier** y **S3 Glacier Deep Archive** son servicios de almacenamiento a largo plazo y de bajo costo para datos que no se requieren de forma inmediata.  
- Son ideales para archivos de respaldo, archivado de datos históricos y cumplimiento normativo.


### **5. Beneficios del Storage en AWS**  
- **Escalabilidad:** Crece de forma automática según las necesidades.  
- **Seguridad:** Integración con AWS IAM, cifrado en tránsito y en reposo, y políticas de acceso.  
- **Flexibilidad:** Variedad de servicios que se adaptan a distintos casos de uso (objetos, bloque, archivos y archivado).  
- **Costo-Efectividad:** Modelos de pago por uso que permiten optimizar costos según el consumo real de recursos.

### **Conclusión**  
El almacenamiento en AWS ofrece soluciones robustas y escalables para gestionar todo tipo de datos. Ya sea que necesites almacenar grandes volúmenes de datos no estructurados, proporcionar almacenamiento en bloque para instancias EC2, compartir archivos entre múltiples servidores o archivar datos a largo plazo, AWS tiene un servicio adaptado a cada necesidad.

Esta diversidad de servicios te permite diseñar arquitecturas de almacenamiento que aseguran la durabilidad, disponibilidad y seguridad de los datos, a la vez que optimizas costos y mejoras la eficiencia operativa.

## Proyecto de arquitectura e implementación de almacenamiento

¡Hola! Como proyecto de este curso vas a hacer de cuenta que eres el arquitecto de soluciones de una empresa y el CEO te ha pedido diseñar una arquitectura de almacenamiento para la empresa de seguros. Además, debes implementar una prueba de concepto que como mínimo cuente con lo siguiente:

- 2 buckets en diferentes regiones.

- Replicación de archivos entre las regiones.

- Pruebas de replicación entre regiones.

- Configuración del bucket principal con las siguientes funcionalidades:
Versionamento.
Encriptación con KMS.
Ciclo de vida de la siguiente forma (no para objetos versiones anteriores):

 1. Objetos mayores a 2 meses pasan a S3-IA.
 2. Objetos con 6 meses de antigüedad pasan a Glacier.
 3. Objetos con 2 años de antigüedad deben borrarse.
 
- Crear un servidor con un volumen EBS agregado a la instancia.

- A través de la CLI consultar los buckets generados y migrar la información que se
- tiene en el EBS al bucket usando la CLI.

- Genera un snapshot del volumen y mígralo a la región en donde se encuentra el bucket secundario.

## Características de S3

Amazon S3 (Simple Storage Service) es un servicio de almacenamiento de objetos de AWS que se utiliza para almacenar y recuperar cualquier cantidad de datos desde cualquier lugar. Algunas de sus características clave son:

- **Alta Durabilidad:**  
  Diseñado para ofrecer 99.999999999% (11 nueves) de durabilidad, lo que significa que tus datos están protegidos incluso en caso de fallas de hardware.

- **Alta Disponibilidad:**  
  Proporciona disponibilidad robusta para acceder a los datos de manera continua, integrándose con múltiples regiones y zonas de disponibilidad.

- **Escalabilidad:**  
  Permite almacenar y gestionar grandes volúmenes de datos de forma prácticamente ilimitada sin necesidad de aprovisionar infraestructura adicional.

- **Flexibilidad en el Acceso:**  
  Se puede acceder a los datos mediante interfaces web, API REST y SDKs, facilitando su integración con otras aplicaciones y servicios de AWS.

- **Modelos de Pago por Uso:**  
  Solo pagas por la cantidad de almacenamiento que utilizas y las solicitudes que realizas, lo que ayuda a optimizar costos.

- **Seguridad y Cumplimiento:**  
  Soporta cifrado en reposo (integración con AWS KMS) y en tránsito, y se puede gestionar mediante políticas de IAM, listas de control de acceso (ACL) y políticas de bucket.

- **Versionado y Gestión de Ciclo de Vida:**  
  Permite habilitar el versionado de objetos para proteger contra sobrescritura o eliminación accidental, además de definir reglas de ciclo de vida para mover o eliminar datos automáticamente.

- **Integración con otros Servicios de AWS:**  
  Funciona de manera nativa con otros servicios como Amazon Athena, AWS Glue, Amazon Redshift Spectrum, y AWS Lambda, facilitando análisis y procesamiento de datos.

Estas características hacen de Amazon S3 una solución robusta y versátil para almacenar datos de aplicaciones, backups, archivos multimedia, y mucho más.

## Resumen

S3 es almacenamiento de objetos como archivos, PDF’s, imágenes, etc. Dentro de S3 contamos con diferentes tipos de almacenamiento:

- S3 Standar
- S3 IA
- S3 IA One Zone
- Glacier

Dependiendo de la clase de S3 va a variar la durabilidad y disponibilidad.

Bucket es la unidad donde vamos a almacenar la información en S3, su identificador se encuentra compuesto por la región donde fue creado, la dirección de Amazon AWS y el nombre del bucket. Para los casos cuando queramos acceder a un objeto simplemente se le suma el nombre del objeto, este debe ser único, en minúsculas y no se permiten los caracteres especiales salvo _ y -. El nombre de un Bucket debe ser único a nivel global.

## Versionamiento de archivos en S3

El **versionamiento** en Amazon S3 es una característica que te permite mantener múltiples versiones de un mismo objeto (archivo) dentro de un bucket. Esto significa que cada vez que se actualiza o sobrescribe un objeto, S3 guarda una nueva versión en lugar de reemplazar la existente.

### **Beneficios del Versionamiento**

- **Protección contra pérdida de datos:**  
  Permite recuperar versiones anteriores en caso de eliminación o sobrescritura accidental.

- **Auditoría y trazabilidad:**  
  Puedes hacer seguimiento de los cambios realizados en un objeto a lo largo del tiempo.

- **Facilita la recuperación de errores:**  
  Si una actualización genera un error o problema, se puede restaurar una versión anterior.

### **Cómo Habilitar el Versionamiento**

1. **Accede a la consola de S3:**  
   Ingresa a la [Consola de AWS S3](https://console.aws.amazon.com/s3/).

2. **Selecciona el bucket:**  
   Haz clic en el bucket en el que deseas habilitar el versionamiento.

3. **Accede a las propiedades del bucket:**  
   Ve a la pestaña **"Properties"** (Propiedades).

4. **Habilita el versionamiento:**  
   Busca la sección **"Bucket Versioning"** y haz clic en **"Edit"**. Luego, selecciona **"Enable"** (Habilitar) y guarda los cambios.

### **Consideraciones Importantes**

- **Costo:**  
  El versionamiento puede aumentar los costos de almacenamiento, ya que se conservan múltiples copias de cada objeto. Es recomendable configurar políticas de ciclo de vida para eliminar versiones antiguas o moverlas a almacenamiento de menor costo (como S3 Glacier).

- **Administración:**  
  Mantener el control de las versiones puede complicar la gestión de objetos si no se implementa un buen plan de gobernanza de datos.

- **Restauración de datos:**  
  Puedes recuperar una versión anterior de un objeto a través de la consola, AWS CLI o la API de S3.

### **Ejemplo Práctico con AWS CLI**

Para habilitar el versionamiento en un bucket con AWS CLI, puedes usar el siguiente comando:

```bash
aws s3api put-bucket-versioning --bucket nombre-del-bucket --versioning-configuration Status=Enabled
```

Y para listar las versiones de un objeto:

```bash
aws s3api list-object-versions --bucket nombre-del-bucket --prefix nombre-del-objeto
```

El versionamiento en S3 es una herramienta poderosa para mejorar la **resiliencia y seguridad** de tus datos, permitiendo una recuperación fácil en caso de errores o cambios no deseados.

**Resumen**

Tener un control de versiones de tus archivos es importante y necesario cuando manejamos información muy delicada. En los casos donde tenemos un error o cargamos un archivo incompleto siempre podremos volver a la versión anterior de nuestro archivo.

Al momento de ir añadiendo varias versiones de un archivo AWS va a poner un tag al último archivo para tener claro que es esta la última versión. Es importante tener en cuenta que la característica de versionamiento te va a cobrar por el almacenamiento total de tus archivos, es decir la última versión y todas sus versiones anteriores.

**Lecturas recomendadas**

[Uso del control de versiones - Amazon Simple Storage Service](https://docs.aws.amazon.com/es_es/AmazonS3/latest/dev/Versioning.html)

## Sitio web estático

Un **sitio web estático** es aquel que está compuesto únicamente por archivos fijos, como HTML, CSS, JavaScript, imágenes y otros recursos, sin necesidad de procesamiento del lado del servidor para generar contenido dinámico.

### **Opciones para alojar un sitio web estático en AWS**

- **Amazon S3:**  
  Puedes configurar un bucket de S3 para servir como alojamiento para tu sitio web estático. AWS S3 permite habilitar el "Static website hosting", donde se especifica un documento de índice (por ejemplo, `index.html`) y opcionalmente una página de error.
  
- **Amazon CloudFront:**  
  Para mejorar el rendimiento y la disponibilidad, se puede integrar S3 con CloudFront, la red de distribución de contenido (CDN) de AWS. Esto permite entregar el contenido desde ubicaciones cercanas a los usuarios, reduciendo la latencia.

### **Pasos básicos para configurar un sitio web estático en S3:**

1. **Crear un bucket en S3:**
   - Asigna un nombre único al bucket.
   - Selecciona la región adecuada.
   
2. **Subir tus archivos del sitio web:**
   - Carga todos los archivos HTML, CSS, JS, imágenes, etc.

3. **Configurar el bucket para hosting web:**
   - En las propiedades del bucket, habilita "Static website hosting".
   - Especifica el documento de índice (por ejemplo, `index.html`) y, opcionalmente, la página de error.

4. **Configurar permisos:**
   - Ajusta la política del bucket para permitir el acceso público a los archivos, de modo que los usuarios puedan visualizar el sitio.

5. **(Opcional) Configurar CloudFront:**
   - Crea una distribución de CloudFront que tenga como origen tu bucket de S3 para mejorar la entrega del contenido.

### **Beneficios de un sitio web estático en AWS:**

- **Bajo costo:** S3 y CloudFront ofrecen almacenamiento y entrega a un costo muy reducido.
- **Alta disponibilidad y escalabilidad:** AWS se encarga de la infraestructura, permitiendo que el sitio esté disponible globalmente.
- **Simplicidad:** No se requiere administración de servidores ni bases de datos.

En resumen, un sitio web estático es ideal para blogs, portafolios, sitios corporativos y landing pages, y AWS S3 es una solución muy popular para implementarlos de manera eficiente.

[documentacion](https://docs.aws.amazon.com/es_es/AmazonS3/latest/userguide/WebsiteAccessPermissionsReqd.html)
**Al deagtivat el acceso s ecrea un Bucket Policy**

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::dopscloud.click/*"
        }
    ]
}
```

**Resumen**

Podremos utilizar nuestro propio dominio como en cualquier sitio web estático, para ello usaremos Route 53 que es el servicio encargado de la parte de DNS y gestión de dominios en S3.

En los sitios web estáticos debes tener en cuenta que el dominio deberá llamarse igual al bucket, los archivos index y error deben ser públicos, debe ser configurado con el servicio Route 53.

[video de problema de acceso](https://www.youtube.com/watch?v=w5WRs0wgG54&ab_channel=d3vcloud)

**Lecturas recomendadas**

[Alojamiento de un sitio web estÃ¡tico en Amazon S3 - Amazon Simple Storage Service](https://docs.aws.amazon.com/es_es/AmazonS3/latest/dev/WebsiteHosting.html)

## Logs a nivel de objetos

Los **logs a nivel de objetos** en AWS S3 te permiten rastrear y auditar todas las solicitudes que se realizan a objetos específicos en un bucket. Esto es útil para monitorear accesos, detectar patrones de uso o actividad sospechosa, y para cumplir con requisitos de auditoría.

Existen dos enfoques principales para lograr esto:

1. **S3 Server Access Logging**  
   - **Qué hace:** Registra todas las solicitudes HTTP realizadas a un bucket, incluyendo información detallada de cada acceso (por ejemplo, quién accedió, cuándo, desde qué IP, qué operación se realizó, etc.).
   - **Nivel de detalle:** A nivel de bucket, pero incluye detalles para cada objeto accedido.
   - **Uso:** Se configura en el bucket de origen y almacena los registros en otro bucket (preferiblemente separado) para su análisis.

2. **AWS CloudTrail**  
   - **Qué hace:** Monitorea y registra llamadas a la API de S3 (por ejemplo, PutObject, GetObject, DeleteObject) a nivel de objeto.
   - **Nivel de detalle:** Captura actividad detallada de API que afecta a objetos en S3.
   - **Uso:** Es útil para auditorías de seguridad y cumplimiento, ya que registra quién realizó la acción, cuándo y desde dónde.

En resumen, para tener **logs a nivel de objetos** en S3, puedes habilitar **S3 Server Access Logging** o usar **AWS CloudTrail** para capturar la actividad de API a nivel de objeto. Ambas opciones te ayudarán a mantener un registro detallado de las interacciones con tus objetos almacenados en S3.

**Resumen**

Podemos activar el Object-level Logging dentro de un bucket en S3 para llevar registro de todas las acciones que se realicen en este, esta funcionalidad nos sirve cuando tenemos buckets con información crítica. Al activar el Object-level Logging debemos conectarlo a CloudTrail.

**Lecturas recomendadas**

[Online JSON Viewer](http://jsonviewer.stack.hu/)

[¿Como puedo habilitar el registro en el nivel de objeto para un bucket de S3 con eventos de datos de AWS CloudTrail? - Amazon Simple Storage Service](https://docs.aws.amazon.com/es_es/AmazonS3/latest/user-guide/enable-cloudtrail-events.html)

## Transferencia acelerada

**AWS S3 Transfer Acceleration** es una característica de Amazon S3 que mejora la velocidad de transferencia de archivos entre clientes y buckets de S3, aprovechando la red global de **CloudFront**.  

### **¿Cómo Funciona?**  
- **Red de Borde:** Utiliza ubicaciones de borde (edge locations) de CloudFront para acercar físicamente el punto de entrada de tus datos al usuario, reduciendo la latencia.  
- **Rutas Óptimas:** Optimiza la ruta de la transferencia a través de la red global de AWS, lo que puede resultar en velocidades de carga y descarga significativamente mayores, especialmente para distancias largas.  

### **Ventajas:**  
- **Aceleración de Transferencias:** Ideal para cargas de trabajo que implican subir o descargar grandes volúmenes de datos a nivel global.  
- **Mejora en la Experiencia de Usuario:** Reduce el tiempo de espera para usuarios alejados de la región donde se aloja el bucket.  
- **Fácil de Configurar:** Se habilita a nivel de bucket y se utiliza una URL específica (por ejemplo, `s3-accelerate.amazonaws.com`) para aprovechar la aceleración.  

### **Uso y Configuración:**  
1. **Habilitar Transfer Acceleration en el Bucket:**  
   - En la consola de AWS S3, selecciona el bucket deseado.  
   - Ve a la pestaña **"Propiedades"** y busca la opción **"Transfer Acceleration"**.  
   - Actívala y guarda los cambios.
2. **Acceso a través de la URL acelerada:**  
   - Una vez habilitado, puedes usar la URL `bucket-name.s3-accelerate.amazonaws.com` en lugar de la URL estándar para transferir datos.

### **Consideraciones:**  
- **Costo Adicional:** Transfer Acceleration tiene un costo extra basado en la cantidad de datos transferidos y la ubicación de origen/destino.  
- **Evaluación de Beneficios:** Se recomienda probar la herramienta con la funcionalidad de **"Speed Comparison"** que ofrece AWS para determinar si la aceleración mejora las transferencias en tu caso particular.

Con AWS S3 Transfer Acceleration, puedes lograr que la transferencia de archivos sea más rápida y eficiente, mejorando la experiencia de usuarios y optimizando el rendimiento de aplicaciones que dependen de transferencias de datos a nivel global.

**Resumen**

Tomando ventaja del servicio de CDN de AWS podemos cargar nuestra información de forma más rápida, esta característica no se encuentra disponible en buckets que contengan puntos (.) en su nombre.

La transferencia acelerada te será sumamente útil cuando tengas que subir información a tu bucket, pero tú no te encuentres en la misma región donde creaste tu bucket.

**Lecturas recomendadas**

[Aceleracion de transferencia de Amazon S3 - Amazon Simple Storage Service](https://docs.aws.amazon.com/es_es/AmazonS3/latest/dev/transfer-acceleration.html)

[3 accelerate speedtest](https://s3-accelerate-speedtest.s3-accelerate.amazonaws.com/en/accelerate-speed-comparsion.html)

## Eventos en S3

Los **eventos en Amazon S3** te permiten configurar notificaciones automáticas para que se desencadenen acciones cuando ocurren ciertos cambios en tus buckets. Esto es útil para automatizar procesos y flujos de trabajo sin necesidad de intervención manual.

### **Características Principales:**

- **Desencadenar Acciones Automáticas:**  
  Cuando se produce un evento (por ejemplo, la creación, eliminación o modificación de un objeto), S3 puede enviar una notificación a otros servicios de AWS.

- **Destinos de Notificaciones:**  
  Puedes configurar notificaciones para que se envíen a:
  - **AWS Lambda:** Para ejecutar una función en respuesta al evento (ideal para procesamiento de datos en tiempo real).
  - **Amazon SQS (Simple Queue Service):** Para poner en cola los eventos y procesarlos de forma asíncrona.
  - **Amazon SNS (Simple Notification Service):** Para enviar mensajes o alertas a múltiples suscriptores.

- **Configuración Flexible:**  
  Puedes especificar qué eventos deseas monitorear (por ejemplo, `s3:ObjectCreated:*`, `s3:ObjectRemoved:*`) y aplicar filtros basados en prefijos o sufijos en los nombres de archivo.

### **Ejemplo de Configuración:**

1. **Accede a la consola de S3:**  
   Selecciona el bucket para el cual deseas configurar las notificaciones.

2. **Configura las notificaciones:**  
   - Ve a la pestaña **"Properties"** (Propiedades).
   - Busca la sección **"Event notifications"**.
   - Crea una nueva notificación, eligiendo el evento deseado (por ejemplo, "All object create events").
   - Selecciona el destino (Lambda, SQS o SNS).

3. **Guarda la configuración:**  
   Una vez configurada, cada vez que se cumpla el evento, se enviará la notificación al destino configurado.

### **Beneficios:**

- **Automatización:** Permite iniciar procesos automáticos (como análisis de datos, generación de thumbnails, etc.) cuando se suben archivos a S3.
- **Integración:** Facilita la integración entre S3 y otros servicios de AWS.
- **Escalabilidad:** Los eventos se gestionan de forma escalable y se pueden procesar de forma asíncrona.

En resumen, los **eventos en S3** son una herramienta poderosa para automatizar flujos de trabajo y responder a cambios en tus datos de manera rápida y eficiente. ¿Te gustaría profundizar en algún aspecto en particular o necesitas un ejemplo específico de configuración?

**Resumen**

Los eventos nos servirán en los casos donde queremos recibir notificaciones cuando se ejecute determinada acción dentro de un bucket con información importante.

Al momento de crear un evento debemos ponerle un nombre, indicarle la acción que debe notificar, además podemos especificarle la carpeta y el tipo de archivo. Por último, debemos indicarle hacia donde debe mandar la notificación, puede ser hacia:

- SNS Topic.
- SQS Queue.
- Lambda Function.

## Replicación

La replicación en Amazon S3 es una funcionalidad que permite copiar automáticamente objetos de un bucket de origen a otro bucket (ya sea en la misma región o en una región diferente) de forma asíncrona. Esto ayuda a mejorar la disponibilidad de los datos, facilita la recuperación ante desastres y cumple con requisitos de cumplimiento normativo.

### **Características Clave de la Replicación en S3:**

- **Cross-Region Replication (CRR):**  
  Copia objetos de un bucket de S3 a otro en una región diferente, lo que mejora la resiliencia y la latencia para usuarios en distintas geografías.

- **Same-Region Replication (SRR):**  
  Replica objetos entre buckets en la misma región, lo que puede ser útil para la conformidad, la separación de cargas o para crear entornos de copia de seguridad.

- **Configuración basada en reglas:**  
  Puedes definir reglas de replicación para seleccionar qué objetos replicar. Estas reglas pueden incluir filtros basados en prefijos o sufijos para replicar solo ciertos tipos de archivos.

- **Requisitos previos:**  
  Para habilitar la replicación, ambos buckets (origen y destino) deben tener habilitado el **versionado**. Además, es necesario configurar los permisos adecuados mediante políticas de IAM y de bucket.

### **Beneficios:**

- **Alta disponibilidad y durabilidad:**  
  Asegura que los datos estén disponibles incluso en caso de fallas en una región.
  
- **Mejora de la resiliencia:**  
  Permite la recuperación rápida ante desastres, ya que los datos están replicados en múltiples ubicaciones.

- **Cumplimiento normativo:**  
  Facilita cumplir con requisitos de residencia de datos y políticas de respaldo.

### **Ejemplo de Configuración (Resumen):**

1. **Habilitar el versionado** en ambos buckets (origen y destino).
2. **Configurar una regla de replicación** en el bucket de origen, especificando:
   - El bucket de destino.
   - Opcionalmente, filtros de prefijo o sufijo.
   - Las condiciones de replicación (por ejemplo, replicar todos los objetos nuevos o actualizados).
3. **Verificar que los permisos** de IAM y las políticas de bucket permitan la replicación.

Con esto, S3 se encargará de replicar automáticamente los objetos según las reglas definidas, sin necesidad de intervención manual.

¿Necesitas más detalles sobre algún aspecto en particular de la replicación en S3?

**Resumen**

La característica de replicar información se realiza solamente para buckets de una región a otra, no es posible pasar de un bucket de una misma región a otro de la misma.

El proceso de replicación se realiza de forma asíncrona. Es común realizar réplicas para Data Recovery, Auditorías y Compliance.

Al momento de replicar la información podemos indicarle que sean todos los objetos del bucket, los objetos que se encuentren dentro de determinada carpeta o aquellos que tengan cierto tag. Además, podemos replicar objetos encriptados.

**Lecturas recomendadas**

[Replicación entre regiones (CRR) - Amazon Simple Storage Service](https://docs.aws.amazon.com/es_es/AmazonS3/latest/dev/crr.html)