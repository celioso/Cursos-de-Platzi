# Curso de Infraestructura Como Código en AWS

## Infraestructura como Código en AWS: Despliegue Automatizado Seguro

### ¿Qué es IaC?

Infraestructura como Código (IaC) es la práctica de gestionar y aprovisionar recursos de infraestructura mediante archivos de configuración legibles por máquina, en lugar de configurarlos manualmente.

### Beneficios de IaC en AWS

* **Automatización**: El despliegue, actualización y mantenimiento de la infraestructura se realiza automáticamente, evitando errores humanos.
* **Reproducibilidad**: Puedes replicar entornos idénticos en segundos.
* **Versionado**: Control de cambios con sistemas como Git, facilitando auditoría y rollback.
* **Seguridad**: Configuraciones controladas y auditable que minimizan riesgos.

### Herramientas comunes de IaC en AWS

* **AWS CloudFormation**
  Servicio nativo para describir recursos AWS mediante plantillas JSON o YAML.
  Permite crear, actualizar y eliminar pilas completas (stacks) de recursos.

* **AWS CDK (Cloud Development Kit)**
  Framework que permite definir infraestructura con lenguajes de programación como Python, TypeScript o Java, generando plantillas CloudFormation.
  Facilita IaC más expresiva y reutilizable.

* **Terraform (HashiCorp)**
  Herramienta multiplataforma para IaC que soporta AWS y muchos otros proveedores.
  Usa lenguaje declarativo (HCL) y permite manejar infraestructura híbrida.

### Despliegue Automatizado Seguro

Para un despliegue seguro y eficiente usando IaC, sigue estas buenas prácticas:

### 1. Gestión de Accesos

* Usa **IAM Roles** con permisos mínimos necesarios (principio de menor privilegio).
* Automatiza con **roles de servicio** para que las herramientas IaC accedan a AWS de forma segura.

### 2. Versionamiento y Revisión de Código

* Almacena plantillas o scripts en repositorios como Git.
* Usa **Pull Requests** para revisiones y auditorías de cambios antes de aplicar.

### 3. Pruebas y Validaciones

* Realiza validación de sintaxis y pruebas de despliegue en entornos de desarrollo antes de producción.
* Usa herramientas como **cfn-lint** (para CloudFormation) o pruebas unitarias con CDK.

### 4. Automatización con CI/CD

* Integra tu IaC en pipelines CI/CD (GitHub Actions, AWS CodePipeline, Jenkins) para despliegue automático tras validación.

### 5. Monitoreo y Auditoría

* Configura logs y monitoreo para detectar cambios inesperados o fallos.
* Usa AWS CloudTrail para auditar cambios en la infraestructura.

### 6. Manejo de secretos

* Nunca incluyas claves o contraseñas en las plantillas.
* Usa AWS Secrets Manager o AWS Systems Manager Parameter Store para gestionar secretos.

---

## Ejemplo básico con AWS CloudFormation (YAML)

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: "Ejemplo básico de creación de una instancia EC2"
Resources:
  MyEC2Instance:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: t2.micro
      ImageId: ami-0abcdef1234567890
      Tags:
        - Key: Name
          Value: MiInstanciaAutomatizada
```

Con este archivo puedes crear la infraestructura ejecutando:

```bash
aws cloudformation deploy --template-file template.yaml --stack-name MiStack
```

## Despliegue de Infraestructura como Código en la Nube

### ¿Qué es Infraestructura como Código?

Infraestructura como Código (IaC) es una práctica que consiste en definir y gestionar la infraestructura de manera automatizada mediante archivos de configuración, en lugar de hacerlo manualmente. Esto permite:

* Provisionar recursos de manera rápida y reproducible.
* Mantener la infraestructura versionada y auditable.
* Facilitar la colaboración y la integración con procesos de desarrollo (DevOps).

### ¿Por qué usar IaC para desplegar infraestructura en la nube?

* **Automatización:** Se eliminan tareas manuales repetitivas y propensas a errores.
* **Consistencia:** Los entornos (desarrollo, prueba, producción) pueden ser idénticos.
* **Escalabilidad:** Facilita escalar y ajustar recursos bajo demanda.
* **Rapidez:** Despliegue rápido y repetible en múltiples regiones o cuentas.
* **Control de versiones:** Cambios rastreados y reversibles con herramientas tipo Git.
* **Seguridad:** Aplicación de políticas de seguridad estandarizadas y controladas.

### Principales pasos en el despliegue de IaC en la nube

1. **Definir la infraestructura como código**
   Crear archivos de configuración (YAML, JSON, HCL, código fuente) que describan los recursos que quieres provisionar (máquinas virtuales, bases de datos, redes, etc.).

2. **Versionar el código**
   Guardar los archivos en un repositorio (GitHub, GitLab, Bitbucket) para control de versiones, auditoría y colaboración.

3. **Validar la configuración**
   Usar herramientas de validación para verificar sintaxis y buenas prácticas (e.g., `terraform validate`, `cfn-lint`).

4. **Automatizar el despliegue**
   Integrar con pipelines de CI/CD para que la infraestructura se cree o actualice automáticamente tras cambios en el repositorio.

5. **Ejecutar el despliegue**
   Ejecutar comandos para aplicar la infraestructura (ejemplo: `terraform apply`, `aws cloudformation deploy`).

6. **Monitorear y mantener**
   Supervisar el estado de la infraestructura y aplicar actualizaciones mediante cambios en el código y redeployment.

### Herramientas comunes para IaC en la nube

| Herramienta            | Descripción                                                 | Ecosistema             |
| ---------------------- | ----------------------------------------------------------- | ---------------------- |
| **AWS CloudFormation** | Servicio nativo AWS para IaC con JSON/YAML                  | AWS                    |
| **AWS CDK**            | Definición de infraestructura con código (Python, TS, etc.) | AWS                    |
| **Terraform**          | Herramienta open-source para múltiples nubes                | AWS, Azure, GCP, otros |
| **Pulumi**             | IaC con lenguajes de programación modernos                  | AWS, Azure, GCP, etc.  |

### Buenas prácticas para despliegues IaC en la nube

* **Principio de menor privilegio:** Minimiza permisos y roles para mayor seguridad.
* **Separar ambientes:** Mantén archivos/configuraciones separados para dev, test y producción.
* **Revisión de cambios:** Usa pull requests y revisiones para controlar cambios.
* **Automatización completa:** Despliegue y pruebas automáticas para evitar errores.
* **Backup y rollback:** Planifica recuperaciones y versiones anteriores.
* **Documentación:** Mantén documentación actualizada de la infraestructura y procesos.

### Ejemplo sencillo: despliegue con Terraform en AWS

Archivo `main.tf`:

```hcl
provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "bucket" {
  bucket = "mi-bucket-iac-ejemplo"
  acl    = "private"
}
```

Despliegue:

```bash
terraform init
terraform plan
terraform apply
```

#### ¿Qué es desplegar infraestructura en la nube?

Desplegar infraestructura en la nube es un proceso que evoluciona constantemente y trae consigo una serie de herramientas y servicios que facilitan su implementación. Cuando trabajamos con infraestructura en la nube, es importante considerar las herramientas disponibles para su despliegue y los diversos "cloud providers" en los que se puede implementar dicha infraestructura. En este curso, el enfoque está en utilizar Terraform en su versión Cloud, una de las varias herramientas que el mercado ofrece. La variedad de herramientas disponibles permite elegir la más adecuada según las necesidades específicas del proyecto.

#### ¿Cuáles son las ventajas del versionamiento en la infraestructura?

Uno de los aspectos más relevantes de la infraestructura como código es su capacidad de versionamiento. Esta funcionalidad nos permite:

- **Tener un control detallado** sobre los componentes y configuraciones de nuestra infraestructura.
- **Monitorear y rastrear cambios**, incluyendo quién realizó el cambio, qué tipo de cambio fue, y cuándo se implementó. Este nivel de trazabilidad ofrece un control invaluable sobre la infraestructura.
- **Versionar cada actualización o nuevo componente**, facilitando la administración y la evolución de los entornos de manera sistemática. Si inicialmente se tenía solo un servidor y luego se agrega una base de datos o cualquier otro componente, cada etapa puede ser registrada y controlada eficientemente.

#### ¿Cómo la eficiencia impacta en el despliegue?

La eficiencia se manifiesta en múltiples aspectos del despliegue de infraestructura. Al tener una plantilla de código previamente diseñada para la infraestructura deseada, se pueden implementar recursos en diferentes ambientes de manera rápida y estandarizada. Las ventajas incluyen:

- **Despliegue rápido**: Tener predefinidas las configuraciones permite desplegar aplicaciones en minutos.
- **Estandarización**: Contar con normas claras y definidas para la infraestructura antes de la implementación garantiza consistencia.
- **Automatización**: Minimiza errores potenciales y optimiza el tiempo requerido para pasar a un ambiente productivo.

#### ¿Qué implica la reutilización de la infraestructura?

La reutilización de infraestructura consiste en tomar recursos previamente desplegados y emplearlos en otros proyectos de manera eficiente:

- **Ahorro de tiempo**: Al reutilizar componentes ya existentes, el tiempo de configuración y despliegue se reduce considerablemente.
- **Facilidad de implementación**: La capacidad de reutilizar plantillas y componentes facilita la gestión y el crecimiento de proyectos nuevos o existentes.
- **Automatización y optimización**: Al reaprovechar infraestructura ya probada, se fomenta la automatización de procesos, lo que se traduce en una mayor efectividad.

#### ¿Qué es la infraestructura inmutable y cómo beneficia a los proyectos?

El concepto de infraestructura inmutable es crucial en el marco de la infraestructura como código. Significa que, si hay un problema, en lugar de tratar de solucionar el error manualmente, se opta por reinstalar desde cero utilizando el código predefinido:

- **Eficiencia en resolución de problemas**: Al no centrarse en buscar y corregir el error manualmente, se ahorra tiempo valioso.
- **Consistencia y confiabilidad**: Al emplear plantillas y configuraciones ya probadas, se minimizan las probabilidades de errores repetitivos en el servidor o sistema.
- **Rapidez en la recuperación**: Permite recuperar el estado funcional del sistema de manera más rápida y segura.

El camino del aprendizaje y la implementación de estos conceptos es vasto y prometedor. ¡Te animamos a seguir explorando y expandiendo tus conocimientos en infraestructura en la nube!

## Herramientas para Infraestructura como Código Multinube

### 1. Terraform (HashiCorp)

* **Descripción:** La herramienta más popular y madura para IaC multinube.
* **Lenguaje:** Usa HCL (HashiCorp Configuration Language), fácil de aprender.
* **Proveedores soportados:** AWS, Azure, Google Cloud, Alibaba Cloud, Oracle Cloud, VMware, y muchos más.
* **Características destacadas:**

  * Gran ecosistema de providers oficiales y de comunidad.
  * Permite definir infraestructura compleja, incluyendo redes, máquinas, bases de datos, etc.
  * Estado remoto para colaboración y bloqueo de cambios.
  * Modularidad para reutilizar configuraciones.
* **Casos de uso:** Ideal para gestionar infraestructuras híbridas y multinube, automatización de despliegues y migraciones.

### 2. Pulumi

* **Descripción:** Plataforma IaC que permite usar lenguajes de programación convencionales.
* **Lenguajes soportados:** JavaScript, TypeScript, Python, Go, C#.
* **Proveedores soportados:** AWS, Azure, Google Cloud, Kubernetes, Docker, etc.
* **Características destacadas:**

  * Usa lenguajes de programación modernos y sus ecosistemas.
  * Buen soporte para aplicaciones nativas en la nube y contenedores.
  * Control de versiones, testing y reutilización de código avanzado.
* **Casos de uso:** Equipos con desarrolladores que prefieren programar infraestructura con sus lenguajes habituales.

### 3. Ansible

* **Descripción:** Herramienta de automatización y configuración que también soporta aprovisionamiento de infraestructura.
* **Lenguaje:** YAML (Playbooks).
* **Proveedores soportados:** AWS, Azure, Google Cloud, OpenStack, VMware, etc.
* **Características destacadas:**

  * Facilita tanto la gestión de configuración como el despliegue de infra.
  * No requiere agentes en los servidores (usa SSH).
  * Amplio conjunto de módulos para diferentes plataformas.
* **Casos de uso:** Más común en gestión y configuración post-despliegue, pero puede aprovisionar recursos en múltiples nubes.

### 4. Crossplane

* **Descripción:** Proyecto CNCF que extiende Kubernetes para gestionar recursos en múltiples nubes.
* **Lenguaje:** Recursos declarativos en YAML.
* **Proveedores soportados:** AWS, Azure, Google Cloud, Alibaba, etc. (mediante controladores).
* **Características destacadas:**

  * Se ejecuta dentro de Kubernetes, gestionando infra desde el cluster.
  * Permite combinar aplicaciones con infraestructura en un solo entorno.
  * Facilita GitOps y despliegues automáticos.
* **Casos de uso:** Organizaciones que usan Kubernetes como centro de operaciones para toda su infraestructura.

### Comparativa rápida

| Herramienta | Lenguaje       | Multinube | Enfoque principal                            |
| ----------- | -------------- | --------- | -------------------------------------------- |
| Terraform   | HCL            | Sí        | Declarativo, amplio soporte                  |
| Pulumi      | Python, JS, Go | Sí        | Imperativo con lenguajes comunes             |
| Ansible     | YAML           | Sí        | Automatización/configuración                 |
| Crossplane  | YAML (K8s CRD) | Sí        | Infraestructura declarativa sobre Kubernetes |

### ¿Qué herramientas existen para gestionar infraestructura como código?

Hoy en día, gestionar infraestructura como código es esencial para desarrolladores y administradores de sistemas. Este enfoque proporciona eficiencia y consistencia en los despliegues, y diversas herramientas han surgido para facilitar este proceso. En este artículo, exploraremos las características y beneficios de algunas herramientas destacadas que soportan la infraestructura como código en múltiples entornos de nube.

### ¿Qué es Terraform y cuáles son sus ventajas?

Terraform es una potente herramienta que permite realizar despliegues multi-cloud. Posee dos versiones, una Open Source y otra Enterprise, adecuándose a diversas necesidades de los usuarios. Entre sus principales ventajas se incluyen:

- Despliegue en múltiples proveedores de nube desde un único lugar.
- Código abierto, lo que permite modificaciones y personalizaciones para casos específicos.
- Amplia documentación y una comunidad activa que proporciona soporte y extensiones.

Te invitamos a profundizar en el curso de Terraform en Platzi, si deseas aprender más sobre esta herramienta.

### ¿Cómo utiliza Pulumi lenguajes de programación?

Pulumi destaca por su capacidad de aprovechar conocimientos de lenguajes de programación específicos para gestionar despliegues de infraestructura. Algunas características son:

- Despliegue multi-cloud utilizando lenguajes de programación familiares.
- Integración directa con los entornos de desarrollo.
- Acompañada de una comunidad que continuamente mejora y expande sus funcionalidades.

### Serverless Framework: ¿cómo facilita la arquitectura sin servidores?

El Serverless Framework está diseñado para la implementación de arquitecturas serverless, es decir, sin servidores físicos. Esta herramienta:

- Facilita el despliegue de funciones Lambda, bases de datos DynamoDB, almacenamiento S3, entre otros.
- Utiliza CloudFormation para gestionar la infraestructura, ofreciendo una capa de abstracción y simplificación.

Esta herramienta proporciona un marco de trabajo que permite crear infraestructura a través de código de manera directa y eficiente.

### ¿Qué son las SDKs y cómo se integran en la programación?

Los diferentes proveedores de nube ofrecen SDKs (Software Development Kits) que permiten a los desarrolladores gestionar la infraestructura mediante librerías específicas dentro de sus lenguajes de programación favoritos. Un ejemplo relevante es:

- **Boto3 (AWS)**: Librería en Python que facilita la automatización y gestión de recursos en AWS. Es particularmente útil para proyectos que requieran un alto grado de automatización.

### AWS CDK: ¿qué novedad aporta al despliegue de infraestructura?

El AWS Cloud Development Kit (CDK) es una herramienta creada por Amazon que, a diferencia de los SDKs, permite definir la infraestructura directamente en el código fuente. Principales características:

- No requiere librerías externas, todo se define en el código Python.
- Genera CloudFormation templates detrás de escena y gestiona el despliegue.
- Simplifica la creación de arquitecturas al permitir trabajar directamente con un código estructurado.

### AWS SAM: ¿cómo optimiza el desarrollo de aplicaciones serverless?

El AWS Serverless Application Model (SAM) ofrece un enfoque específico para aplicaciones serverless en AWS. Esta herramienta orientada a la implementación de funciones Lambda y otros servicios serverless permite:

- Cambiar y ajustar la definición de recursos para optimizarlos hacia un enfoque serverless.
- Proporciona un marco que reduce la complejidad en la programación y despliegue de aplicaciones serverless.

### Consejos para escoger la herramienta adecuada

La elección de una herramienta para manejar infraestructura como código depende enormemente del caso de uso específico. Algunas recomendaciones al elegir son:

1. **Analizar las necesidades del proyecto**: ¿Cuál es la arquitectura requerida? ¿Se necesita compatibilidad multi-cloud?
2. **Considerar el conocimiento del equipo**: ¿El equipo ya cuenta con conocimientos previos en un lenguaje específico que pueda ser aprovechado?
3. **Evaluar la escalabilidad y futuro del proyecto**: Algunas herramientas ofrecen mejores opciones para grandes despliegues o crecimiento acelerado.

Con esta diversidad de herramientas a tu disposición, la implementación de infraestructura como código se convierte en una tarea manejable y eficiente. Continúa explorando y eligiendo la opción que mejor se adapte a tus necesidades y las de tu equipo para maximizar los beneficios de esta práctica moderna.

### Ventajas y beneficios de usar AWS CloudFormation

Usar **AWS CloudFormation** para gestionar tu infraestructura como código (IaC) en AWS tiene múltiples ventajas, especialmente si trabajas exclusivamente en este ecosistema. Aquí te presento un resumen claro de sus **ventajas y beneficios**:

### ✅ Ventajas de usar AWS CloudFormation

### 1. **Infraestructura como código nativa de AWS**

* Totalmente integrada con todos los servicios de AWS.
* Permite definir recursos (EC2, S3, RDS, Lambda, etc.) como código en JSON o YAML.

### 2. **Automatización completa del ciclo de vida**

* Crea, actualiza y elimina recursos de forma automática y segura.
* Usa plantillas (`templates`) para definir entornos completos y replicables.

### 3. **Gestión de dependencias**

* CloudFormation resuelve automáticamente el orden en que se deben crear los recursos.
* Maneja relaciones como VPCs, subnets, roles IAM, etc., sin intervención manual.

### 4. **Reutilización y modularidad**

* Permite **anidar plantillas** (nested stacks) y usar **módulos reutilizables**.
* Reduce la duplicación de código y mejora la mantenibilidad.

### 5. **Rollback automático**

* Si algo falla durante el despliegue, **revierte automáticamente** los cambios para evitar estados inconsistentes.

### 6. **Seguimiento y auditoría**

* Cambios registrados en **AWS CloudTrail**.
* Puedes ver qué se creó, cuándo, y con qué parámetros.

### 7. **Integración con otras herramientas**

* Compatible con CI/CD (CodePipeline, GitHub Actions, Jenkins, etc.).
* Puede combinarse con **AWS Systems Manager** y **AWS Config** para gobernanza y cumplimiento.

### 8. **Soporte para parámetros y condiciones**

* Puedes personalizar despliegues mediante `Parameters`, `Mappings` y `Conditions`.
* Ideal para tener múltiples entornos (dev, staging, prod) con una sola plantilla.

### 9. **Actualizaciones controladas (Change Sets)**

* Permite revisar los cambios antes de aplicarlos mediante **Change Sets**.
* Ayuda a prevenir errores en producción.

### 10. **Gratuito**

* No tiene costo adicional (solo pagas por los recursos que creas con él).

### 🏆 Beneficios clave para las organizaciones

| Beneficio                    | Descripción                                                     |
| ---------------------------- | --------------------------------------------------------------- |
| **Consistencia**             | Misma infraestructura en todos los entornos.                    |
| **Escalabilidad**            | Despliegue masivo y rápido de recursos.                         |
| **Eficiencia operativa**     | Menos tareas manuales y errores humanos.                        |
| **Seguridad**                | Control de acceso a recursos mediante plantillas seguras.       |
| **Auditoría y cumplimiento** | Trazabilidad de cambios en todo momento.                        |
| **Reducción de costos**      | Automatización evita errores costosos y tiempos de inactividad. |

### Resumen

#### ¿Cuáles son las ventajas de usar CloudFormation en AWS?

Al hablar de Amazon Web Services (AWS), una de las herramientas más poderosas y ventajosas que puedes utilizar es CloudFormation. Esta herramienta permite el despliegue y la gestión de infraestructura y aplicaciones de manera eficiente y segura. A medida que los servicios en la nube se vuelven cada vez más esenciales para el funcionamiento de las organizaciones modernas, entender cómo usar CloudFormation te brindará una ventaja competitiva.

#### ¿Cómo funciona el flujo de despliegue en CloudFormation?

CloudFormation utiliza plantillas YAML o JSON para definir la infraestructura como código. El flujo básico consiste en verificar el código y realizar la fase de despliegue. Esta integración profunda con AWS permite una implementación fluida de la infraestructura. Además, existen múltiples servicios relacionados que facilitan la gestión y optimización de tus recursos.

#### ¿Cuál es el soporte técnico que ofrece AWS para CloudFormation?

Una de las características más destacadas de CloudFormation es el soporte que provee AWS. Si tienes un contrato de soporte con AWS y experimentas problemas con el despliegue, puedes abrir un caso de soporte. Un equipo especializado te asistirá para revisar y corregir el código, asegurándose de que los despliegues se realicen correctamente. Esta capacidad de soporte es fundamental para mantener la operatividad sin interrupciones.

#### ¿Por qué es importante la integración nativa de CloudFormation con AWS?

CloudFormation, al ser un servicio nativo de AWS, tiene una integración total con los demás servicios de la plataforma. Esto significa que puedes aprovechar las mejores prácticas de seguridad, escalabilidad y operatividad en tus despliegues. La funcionalidad de Designer de CloudFormation, por ejemplo, permite la creación de infraestructura de forma visual, asegurando que las configuraciones sean precisas y alineadas con tus necesidades.

#### ¿Qué beneficios ofrece CloudFormation en términos de escalabilidad y seguridad?

- **Escalabilidad**: Permite desplegar desde un solo servidor hasta cientos en diferentes cuentas de manera simultánea. Esto facilita la gestión de diferentes ambientes de trabajo sin complicaciones.
- **Seguridad**: Integra múltiples servicios de seguridad para cifrar llaves de conexión y gestionar bases de datos. De este modo, puedes aplicar las mejores prácticas de seguridad de AWS en tus recursos.

#### ¿Por qué es CloudFormation una herramienta transversal y de uso extendido?

CloudFormation es apto para cualquier empresa, independientemente del sector o industria. Es una herramienta transversal que ha cobrado importancia por su capacidad de transformar prácticas de desarrollo de código en despliegues de infraestructura. Empresas reconocidas como el FC Barcelona, Expedia y CoinBase, entre otras, utilizan este servicio para gestionar su infraestructura de manera eficaz.

Usar CloudFormation no solo moderniza tus procesos, sino que también garantiza una infraestructura robusta y ajustada a las demandas dinámicas del entorno tecnológico actual. Con estas ventajas, el aprendizaje y dominio de CloudFormation te posicionará en un lugar privilegiado en la gestión de servicios en la nube.

## Uso de la Consola de CloudFormation para Despliegues de Infraestructura

Usar la **Consola de CloudFormation** en AWS para desplegar infraestructura como código (IaC) permite automatizar y gestionar recursos de manera segura, escalable y reproducible. A continuación, te explico cómo se utiliza y cuáles son sus ventajas clave.

### 🧭 ¿Qué es la Consola de CloudFormation?

La **Consola de CloudFormation** es la interfaz gráfica web de AWS para crear, visualizar, administrar y eliminar stacks (conjuntos de recursos) definidos en plantillas YAML o JSON. Permite a los usuarios desplegar infraestructura sin necesidad de interactuar con la CLI o APIs directamente.

### 🚀 Pasos para desplegar infraestructura con la Consola de CloudFormation

#### 1. **Acceder a la consola**

* Ve a: [https://console.aws.amazon.com/cloudformation](https://console.aws.amazon.com/cloudformation)
* Selecciona tu región preferida.

#### 2. **Crear un stack**

* Haz clic en **"Create stack"** > **"With new resources (standard)"**.
* Elige una fuente:

  * Subir archivo local (`.yaml` o `.json`).
  * Ingresar una URL de plantilla en S3.
  * Escribir manualmente la plantilla.

#### 3. **Configurar detalles del stack**

* Asigna un **nombre al stack**.
* Introduce los **valores de parámetros** (si la plantilla los requiere).

#### 4. **Opciones avanzadas (opcional)**

* Etiquetas para organización.
* Roles de IAM que CloudFormation usará.
* Configuraciones de stack policies y protección contra eliminación.

#### 5. **Revisión y creación**

* Revisa el resumen de configuración.
* Marca la casilla para confirmar que CloudFormation creará recursos con posibles costos.
* Haz clic en **"Create stack"**.

### 🔍 Seguimiento y gestión del stack

* Puedes monitorear el progreso en la pestaña **"Events"**.
* Ver recursos creados en **"Resources"**.
* Consultar salidas (`Outputs`) que contienen información útil como URLs, ARNs, etc.
* Actualizar el stack desde la consola si hay cambios en la plantilla.

### ✅ Ventajas de usar la Consola de CloudFormation

| Beneficio                      | Descripción                                                        |
| ------------------------------ | ------------------------------------------------------------------ |
| 🎛️ Interfaz amigable          | Ideal para usuarios nuevos en IaC que prefieren no usar la CLI.    |
| 🔐 Seguridad controlada        | Integración con IAM para control de acceso granular.               |
| 📊 Visualización clara         | Visualiza dependencias y relaciones entre recursos en un diagrama. |
| 🕒 Historial y seguimiento     | Registra eventos, fallos y cambios de estado del stack.            |
| 🔁 Reutilización de plantillas | Permite usar la misma plantilla en múltiples entornos o regiones.  |
| 🧪 Validación automática       | Detecta errores de sintaxis antes del despliegue.                  |

### 🧰 Buenas prácticas

* Usa plantillas validadas con `cfn-lint`.
* Habilita **Stack termination protection** para evitar eliminaciones accidentales.
* Utiliza parámetros y mappings para hacer tus plantillas reutilizables.
* Combina con **S3** y **CodePipeline** para despliegues automatizados desde repositorios.

### Resumen

#### ¿Qué es la consola de CloudFormation y cómo se accede?

La consola de CloudFormation es una herramienta esencial para quienes trabajan con infraestructura como código en AWS. Proporciona un entorno visual e interactivo para gestionar recursos de AWS a través de plantillas de infraestructura declarativas. Comenzar a familiarizarse con esta consola es el primer paso para aprovechar todas sus funcionalidades.

Para acceder a la consola de CloudFormation, inicia sesión en AWS y busca el servicio "CloudFormation" en la barra de búsqueda. Dentro de la consola, observarás varias secciones, cada una con una funcionalidad diferente.

#### ¿Cuál es la estructura de la consola de CloudFormation?

#### Sección de stacks

La sección de stacks es donde se crean y gestionan las colecciones de recursos de AWS. Al seleccionar "crear stack", la consola te guiará a través de una serie de pasos para definir y desplegar estos recursos.

#### Stacks sets

Los "stack sets" son muy útiles para despliegues multi-cuenta de infraestructura. Permiten estandarizar configuraciones en diferentes ambientes y ahorrar tiempo en implementaciones masivas.

#### Variables exportadas y comunicación

En la sección de "exports", encontrarás variables que permiten la comunicación entre diferentes stacks. Esta es una funcionalidad crítica para estructurar proyectos complejos, donde los recursos distribuidos necesitan interactuar entre sí.

#### ¿Qué es el Designer y cómo se utiliza?

El Designer de CloudFormation es una herramienta gráfica dentro de la consola que permite crear plantillas visualmente. Representa recursos como elementos gráficos que se pueden arrastrar y soltar, generando automáticamente una plantilla JSON al momento de guardar.

#### Ventajas del Designer

- **Visualización intuitiva**: Ideal para quienes prefieren interactuar gráficamente.
- **Transformación directa**: Los diseños se convierten en plantillas JSON listas para ser desplegadas en AWS.

#### ¿Cómo monitorear y gestionar despliegues?

#### Estado de los templates

Dentro de la consola principal de CloudFormation, los templates desplegados se categorizan según su estado:

- Activos
- Completados
- Fallidos
- Eliminados
- En progreso

Esto te permite realizar un seguimiento fácil y eficiente de tus deployments.

#### Detalle y solución de problemas

En cada template, puedes profundizar en los detalles para verificar qué recursos se han desplegado o identificar errores. Esta visibilidad es crucial para la resolución de problemas, ya que te permite:

- Localizar en qué parte o cuenta falló un stack.
- Entender las razones detrás de cualquier fallo.
- Corregir y volver a desplegar con rapidez.

Con una comprensión clara de estas secciones y herramientas, estarás completamente equipado para manejar la infraestructura de AWS de manera eficiente. La familiarización continua con esta consola te brindará una ventaja significativa a medida que avances en el curso y desarrolles competencias en infraestructura como código. ¡Ánimo, sigue explorando y aprendiendo!

### Resumen

#### ¿Qué es un Temple en CloudFormation?

En el mundo de AWS, las plantillas de CloudFormation son el alma de la infraestructura como código. Estas plantillas ofrecen la posibilidad de definir y aprovisionar recursos de AWS de una manera organizada y sistemática, en la que los componentes clave están claramente establecidos.

#### ¿Cuál es el propósito de la versión en un Temple?

Los Temples, o plantillas, tienen una versión específica. Si no se define explícitamente, AWS lo hará automáticamente, utilizando la versión 2010-09-09. Este campo, aunque opcional, es importante para garantizar que la plantilla pueda aprovechar todas las funcionalidades más recientes de CloudFormation.

`AWSTemplateFormatVersion: '2010-09-09'`

#### ¿Para qué sirve la descripción en un Temple?

La descripción es otro campo opcional en las plantillas de CloudFormation. Su principal utilidad es permitirte identificar qué estás desplegando, funcionando como una suerte de metadata personalizada. Es altamente recomendada como una práctica para mejorar la legibilidad y comprensión del Temple.

`Description: 'Esta es mi primera lambda en CloudFormation'`

#### ¿Por qué es relevante la Metadata en un Temple?

La metadata va un paso más allá en complejidad y permite definir registros específicos como Transformaciones, Init, Interfaces y Diseño. Aunque es completamente opcional, en proyectos más avanzados puede ser crucial para lograr ciertos comportamientos deseados. Sin embargo, no la usaremos en nuestro ejemplo.

#### ¿Qué son los Parámetros en un Temple?

Los parámetros son uno de los elementos más vitales en tu plantilla. Son los datos que necesitas pasarle al Temple para personalizar los recursos que vas a desplegar. Te permiten reutilizar una misma plantilla para diferentes configuraciones.

- **Ejemplo de Parámetro en Lambda:**

Para determinar el valor de runtime en una función Lambda, se puede especificar el lenguaje como un parámetro, adaptable según las necesidades.

```yaml
Parameters:
  Runtime:
    Type: String
    Default: 'Python3.8'
    AllowedValues:
      - 'Python3.6'
      - 'Python3.7'
      - 'Python3.8'
      - 'Java8'
```
#### ¿Cómo funcionan los mappings en un Temple?

Los mappings funcionan como arreglos de llave valor que especifican valores específicos para diferentes configuraciones. Se utilizan comúnmente para definir valores diferencias por región.

- **Ejemplo de Mapping:**

Imagina que quieres desplegar un servidor en múltiples regiones como Frankfurt, Sao Paulo, y Virginia. Puedes definir un mapping para cada región con su correspondiente ID de imagen.

```yaml
Mappings:
  RegionMap:
    us-east-1:
      "AMIID": "ami-0ff8a91507f77f867"
    eu-west-1:
      "AMIID": "ami-047bb4163c506cd98"
```

#### Recomendaciones prácticas

1. **Utilizar Descripciones**: Aunque opcional, las descripciones clarifican el propósito del código y mejoran la mantenibilidad.

2. **Implementar Parámetros**: Facilitan la flexibilidad y reutilización de tus plantillas a lo largo de distintas implementaciones.

3. **Definir Mappings**: Estos aceleran la implementación en múltiples regiones, ajustando automáticamente configuraciones para cada caso.

Al emplear estas técnicas y comprender la esencia de cada uno de estos componentes, podrás crear plantillas de CloudFormation eficaces y versátiles, listas para escalar y adaptarse a tus necesidades de infraestructura. ¡Anímate a explorar más sobre CloudFormation y transforma tu manera de desplegar y gestionar recursos en la nube!

## Creación de Tablas en DynamoDB desde Cero con AWS

Crear tablas en **Amazon DynamoDB** desde cero es un paso clave para construir aplicaciones sin servidor (serverless) o altamente escalables. Puedes hacerlo mediante la **Consola de AWS**, **AWS CLI**, **CloudFormation**, o usando SDKs como Python (Boto3), Node.js, etc.

### 🧩 ¿Qué es DynamoDB?

Amazon DynamoDB es un servicio de base de datos NoSQL completamente gestionado que proporciona almacenamiento rápido y flexible con escalado automático, baja latencia y alta disponibilidad.

### 🔧 Creación de Tablas DynamoDB desde Cero

### Opción 1: 📊 Usando la Consola de AWS

1. Ve a la [Consola de DynamoDB](https://console.aws.amazon.com/dynamodb).
2. Haz clic en **"Create Table"**.
3. Llena los campos requeridos:

   * **Table name**: Por ejemplo, `Usuarios`
   * **Partition key (clave primaria)**: Por ejemplo, `UserId` (tipo `String`)
   * *(Opcional)* Agrega una **sort key** si necesitas una clave compuesta.
4. Opcionalmente configura:

   * Capacidad: **On-demand** (automático) o **provisioned** (manual)
   * **Encryption**, **streams**, **TTL**, **secondary indexes**
5. Haz clic en **"Create Table"**

### Opción 2: 🖥️ Usando AWS CLI

```bash
aws dynamodb create-table \
  --table-name Usuarios \
  --attribute-definitions AttributeName=UserId,AttributeType=S \
  --key-schema AttributeName=UserId,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

🔹 Esto crea una tabla con:

* Clave primaria `UserId` (tipo String)
* Modo de facturación bajo demanda

### Opción 3: 🧬 Con AWS CloudFormation (YAML)

```yaml
Resources:
  UsuariosTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: Usuarios
      AttributeDefinitions:
        - AttributeName: UserId
          AttributeType: S
      KeySchema:
        - AttributeName: UserId
          KeyType: HASH
      BillingMode: PAY_PER_REQUEST
```

🔹 Puedes incluir este recurso en una plantilla de CloudFormation para automatizar la infraestructura.

### Opción 4: 🐍 Con Python (Boto3)

```python
import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

table = dynamodb.create_table(
    TableName='Usuarios',
    KeySchema=[
        {'AttributeName': 'UserId', 'KeyType': 'HASH'},
    ],
    AttributeDefinitions=[
        {'AttributeName': 'UserId', 'AttributeType': 'S'},
    ],
    BillingMode='PAY_PER_REQUEST'
)

print("Creando tabla. Esperando...")
table.meta.client.get_waiter('table_exists').wait(TableName='Usuarios')
print("Tabla creada con éxito")
```

### ✅ Buenas Prácticas

* Usa **PAY\_PER\_REQUEST** si no tienes una carga constante o si estás empezando.
* Añade **Global Secondary Indexes (GSI)** si necesitas consultas por otros campos.
* Activa **Streams** si necesitas activar eventos Lambda desde operaciones CRUD.
* Usa **IAM policies** para controlar el acceso a la tabla.

### Resumen

#### ¿Cómo crear una tabla en AWS DynamoDB usando un template?

La creación de una tabla en AWS DynamoDB puede parecer un desafío, pero con las herramientas y conocimientos adecuados, se vuelve una tarea manejable y emocionante. A continuación, se presenta una guía básica para crear una tabla desde cero utilizando un template en formato YAML, siguiendo la documentación oficial de AWS.

#### ¿Cómo comenzar con la documentación de AWS y crear un archivo YAML?

Para empezar desde cero, es fundamental dirigirse a la documentación oficial de AWS DynamoDB para obtener un template base que nos dirija en el proceso. Aquí está el paso a paso:

1. Busca "DynamoDB" en Google y localiza la documentación de AWS para DynamoDB.
2. Encuentra ejemplos sencillos en formato JSON y YAML. Para este caso, utilizaremos el ejemplo en YAML.
3. Copia el template completamente vacío proporcionado por AWS a tu editor de texto favorito.
4. Guarda el archivo con un nombre adecuado, por ejemplo, `miDynamoDB.yaml`.

`AWSTemplateFormatVersion: "2010-09-09"`

#### ¿Cuáles son los elementos clave de un template?

Al crear un template, es crucial asegurar que contiene los componentes esenciales, como parámetros y recursos:

- **Versión del formato**: Especifica la versión del template de AWS que estás utilizando, p. ej., 2010-09-09.

`AWSTemplateFormatVersion: "2010-09-09"`

- **Recursos**: Es la única propiedad completamente obligatoria, que en este caso será una tabla de DynamoDB.

```yaml
Resources:
  MyDynamoTable:
    Type: "AWS::DynamoDB::Table"
```

#### ¿Cómo definir atributos y llaves primarias?

La definición de atributos es crucial, ya que determina la estructura de tu base de datos.

- **Atributo Definición**: Aunque no obligatorio, es una buena práctica definirlo. Aquí, se especifica la llave primaria para la base de datos.

```yaml
AttributeDefinitions:
  - AttributeName: Gender
    AttributeType: S
```

- **KeySchema**: Esta sección especifica cómo se construirá la llave primaria.

```yaml
KeySchema:
  - AttributeName: Gender
    KeyType: HASH
```
#### ¿Cómo configurar las lecturas y escrituras pagadas?

Dependiendo de tu carga, puedes configurar DynamoDB para que pague por solicitudes o para tener una capacidad preestablecida.

- **BillingMode**: Aquí establecemos cómo se realizarán los cargos, utilizando "PAY_PER_REQUEST" para práctica flexibilidad.

`BillingMode: PAY_PER_REQUEST`

#### ¿Qué se debe saber sobre la encriptación y el exportado de nombres?

La seguridad es una prioridad, y AWS permite activar la encriptación para los datos en reposo.

- **Encriptación**: Active con la configuración SSESpecification.

```yaml
SSESpecification:
  SSEEnabled: true
```

Además, puedes exportar valores, como el nombre de la tabla, para su uso posterior en otras partes de tu infraestructura:

- **Outputs**: Exporta el nombre de DynamoDB para facilitar su referencia.

```yaml
Outputs:
  TableName:
    Value: !Ref MyDynamoDBTable
    Export:
      Name: MyDynamoTableName
```

Con estos pasos, habrás establecido un template básico que puedes desplegar para crear una tabla en DynamoDB con AWS CloudFormation. Recuerda siempre revisar la documentación y mantenerte actualizado sobre las mejores prácticas de AWS para una implementación eficiente. Mantente avanzado en tu aprendizaje revisando más cursos sobre bases de datos en AWS para ampliar tus conocimientos y habilidades.

**Recursos**

[AWS::DynamoDB::Table](https://docs.aws.amazon.com/es_es/AWSCloudFormation/latest/UserGuide/aws-resource-dynamodb-table.html)

## Creación de Stack en AWS paso a paso

Aquí tienes una **guía paso a paso** para la **creación de un Stack en AWS** usando **CloudFormation**, que te permite desplegar infraestructura como código de manera automática y segura.

### 🚀 ¿Qué es un Stack en CloudFormation?

Un **Stack** es una colección de recursos de AWS que se crean, actualizan o eliminan como una sola unidad utilizando una **plantilla (template)** de CloudFormation escrita en YAML o JSON.

### 🛠️ Paso a Paso para Crear un Stack en AWS

### 🟢 1. Preparar la Plantilla (YAML o JSON)

Ejemplo simple en YAML para crear una tabla DynamoDB:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Crear una tabla DynamoDB básica

Resources:
  MiTablaDynamoDB:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: MiTabla
      AttributeDefinitions:
        - AttributeName: Id
          AttributeType: S
      KeySchema:
        - AttributeName: Id
          KeyType: HASH
      BillingMode: PAY_PER_REQUEST
```

Guarda esto como `template.yaml` (o súbelo directamente si usas la consola).

### 🟡 2. Accede a la Consola de AWS

1. Ve a: [https://console.aws.amazon.com/cloudformation](https://console.aws.amazon.com/cloudformation)
2. Haz clic en **"Create stack"** > **"With new resources (standard)"**

### 🟠 3. Cargar la Plantilla

* **Plantilla local**: Carga el archivo `.yaml` o `.json` desde tu computadora.
* **S3**: Si la plantilla está alojada en un bucket de S3, pega su URL.
* **Ejemplo rápido**: También puedes usar plantillas de muestra de AWS.

Haz clic en **"Next"**.

### 🔵 4. Configurar el Stack

1. **Stack name**: Elige un nombre, por ejemplo, `MiPrimerStack`
2. Si tu plantilla tiene **parámetros**, aquí puedes asignar valores.

Haz clic en **"Next"**.

### 🟣 5. Opciones Avanzadas (opcional)

Aquí puedes:

* Agregar etiquetas
* Crear roles de IAM específicos
* Configurar políticas de stack
* Activar notificaciones o protecciones contra eliminación

Haz clic en **"Next"**.

### 🔴 6. Revisión y Creación

1. Revisa todos los detalles
2. Marca la casilla de **“I acknowledge that AWS CloudFormation might create IAM resources”** si aplica
3. Haz clic en **"Create stack"**

### 📈 7. Monitorea la Creación del Stack

* Ve a la pestaña **"Events"** del Stack para ver el progreso.
* En unos segundos/minutos verás el estado como: ✅ `CREATE_COMPLETE`.

### 📦 8. Accede a los Recursos Creados

Desde la pestaña **"Resources"** del Stack puedes:

* Ver los recursos creados
* Acceder directamente a ellos en sus respectivos servicios (como DynamoDB, S3, Lambda, etc.)

### 🧹 9. (Opcional) Eliminar el Stack

Cuando ya no lo necesites, puedes seleccionar el Stack y hacer clic en **"Delete"** para borrar todos los recursos relacionados automáticamente.

### ✅ Consejos Finales

* Usa plantillas reutilizables y controladas con Git.
* Aprovecha los **outputs** para compartir valores generados (como ARNs, URLs, etc.).
* Integra con herramientas como **CI/CD**, **SAM**, o **Serverless Framework** para flujos más avanzados.

### Resumen

#### ¿Cómo crear y desplegar un template en AWS utilizando AWS CloudFormation?

AWS CloudFormation es una herramienta poderosa que automatiza la implementación de recursos en la nube de AWS. En este contexto, aprenderás a crear y desplegar un template desde cero usando CloudFormation, con la finalidad de gestionar tus recursos de manera eficiente. Este proceso te permitirá, con el conocimiento adecuado, construir cualquier tipo de recurso en AWS. ¡Síguenos para descubrir cada paso con precisión!

#### ¿Cómo comenzar el proceso de despliegue en AWS CloudFormation?

Primero, es esencial tener un archivo template que contenga la información necesaria para configurar tus recursos. En este caso, el archivo may-day-in-amo es el punto de partida.

1. Acceder a AWS: Dirígete a la consola de AWS e inicia sesión.
2. Navegar a CloudFormation: En el menú superior izquierdo, selecciona "CloudFormation".
3. Crear stack: Haz clic en "Crear stack" y tendrás la opción de cargar tu template.

# Ejemplo de configuración de un template básico en YAML 
AWSTemplateFormatVersion: '2010-09-09'
Resources:
  # Definiciones de recursos

#### ¿Qué hacer cuando aparece un error?

Al cargar tu template en AWS CloudFormation, es posible que encuentres errores. Por ejemplo, propiedades inválidas que bloqueen el proceso de carga. La solución es revisar la documentación de AWS y corregir las propiedades específicas.

- **Identificar errore**s: AWS proporciona mensajes de error claros. Identifica la propiedad afectada, como la propiedad "Tags" o "StackPolicy".
- **Verificar y corregir**: Consulta la documentación oficial de AWS para entender la propiedad y realizar los cambios necesarios.

#### ¿Cómo continuar después de corregir los errores?

Después de realizar las correcciones necesarias en tu template:

1. **Recargar el template**: Carga el archivo corregido may-day-in-amo nuevamente.
2. **Configurar el stack**: Especifica detalles como el nombre del stack. Ejemplo: "Mi primer TAC Platzi".
3. **Revisión y creación**: Revisa las configuraciones y haz clic en "Crear stack". CloudFormation iniciará el proceso de creación.

### ¿Cómo verificar que el recurso se ha creado correctamente?

Realizar verificaciones es fundamental para asegurarse de que todo está funcionando correctamente.

1. **Supervisar el estado**: En CloudFormation, verifica el estado del stack; debería mostrar "CREATE_COMPLETE".
2. **Consultar DynamoDB**: Si implementaste una tabla en DynamoDB, abre la consola de DynamoDB y verifica que el recurso se haya creado.

#### ¿Qué hacer si necesitas eliminar el stack?

Eliminar un stack es un proceso delicado que requiere confirmación para evitar la pérdida de recursos importantes.

1. **Seleccionar y eliminar el stack**: En la consola de CloudFormation, selecciona el stack y haz clic en "Eliminar".
2. **Confirmación**: AWS solicita confirmar la eliminación para evitar errores.
3. **Monitorear el proceso de eliminación**: El estado cambiará a "DELETE_IN_PROGRESS". Una vez complete, verifica en DynamoDB que el recurso ya no exista.

AWS CloudFormation no solo facilita la creación de recursos complejos de forma automática, sino que optimiza el manejo mediante la reutilización eficiente de templates. Este conocimiento te empodera para avanzar en tus proyectos en la nube. ¡Continúa explorando y aprendiendo del vasto universo de AWS!


## Creación y Conexión de Funciones Lambda y Recursos en AWS

La **creación y conexión de funciones Lambda con otros recursos en AWS** es uno de los pilares de una arquitectura serverless. A continuación te explico paso a paso cómo hacerlo utilizando buenas prácticas y herramientas como CloudFormation:

### ✅ 1. **Crear una Función Lambda**

Puedes hacerlo desde la consola o usando IaC como CloudFormation.

### Ejemplo básico en CloudFormation:

```yaml
Resources:
  MiFuncionLambda:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: MiFuncionLambda
      Runtime: python3.12
      Handler: index.handler
      Code:
        ZipFile: |
          def handler(event, context):
              return {"statusCode": 200, "body": "Hola desde Lambda"}
      Role: !GetAtt LambdaExecutionRole.Arn
```

### ✅ 2. **Crear una Role IAM para Lambda**

Esta role da permisos mínimos necesarios.

```yaml
  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

### ✅ 3. **Conectar Lambda con otros servicios**

Lambda se puede conectar a recursos como:

### 📌 DynamoDB

```yaml
  PermisoDynamo:
    Type: AWS::IAM::Policy
    Properties:
      PolicyName: LambdaDynamoAccess
      Roles: [!Ref LambdaExecutionRole]
      PolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Action:
              - dynamodb:GetItem
              - dynamodb:PutItem
            Resource: "*"
```

### 📌 API Gateway

```yaml
  ApiGateway:
    Type: AWS::ApiGateway::RestApi
    Properties:
      Name: MiAPI

  LambdaPermission:
    Type: AWS::Lambda::Permission
    Properties:
      Action: lambda:InvokeFunction
      FunctionName: !GetAtt MiFuncionLambda.Arn
      Principal: apigateway.amazonaws.com
```

### 📌 EventBridge, S3, SQS, SNS, etc.

Se configuran con triggers o eventos automáticos desde esos servicios.

### ✅ 4. **Desplegar usando AWS Console, CLI o CloudFormation**

Ejemplo con AWS CLI:

```bash
aws cloudformation deploy \
  --template-file plantilla.yaml \
  --stack-name MiStackLambda \
  --capabilities CAPABILITY_NAMED_IAM
```

### ✅ Recomendaciones:

* Usa variables de entorno en Lambda (`Environment`) para URLs o configuraciones externas.
* Usa CloudWatch Logs para depurar y monitorear.
* Usa versiones y alias para gestionar despliegues.

### Resumen

#### ¿Qué son las condiciones en la creación de plantillas CloudFormation?

Al crear plantillas en AWS CloudFormation, puedes usar condiciones opcionales que deciden si se crea un recurso o se asigna una variable a un recurso. Por ejemplo, un volumen solo se crea si ya están desplegados los recursos de producción necesarios. Esta condicionalidad de recursos permite modular y optimizar la gestión de infraestructura.

- **Opcionalidad**: Estas condiciones son completamente opcionales. Pueden usarse o no según las necesidades específicas de la infraestructura.
- **Funcionalidad**: Ayudan a asegurar que ciertos componentes solo se crean cuando otros ya están presentes, evitando configuraciones incompletas o inválidas.

#### ¿Cómo se usa 'Transform' en aplicaciones serverless?
Dentro del contexto de AWS, 'Transform' es una función que se utiliza al crear aplicaciones completamente serverless basadas en AWS SAM (Serverless Application Model).

- **Función principal**: Define el template como serverless para permitir la creación eficiente de estos recursos.
- **Importancia**: Es fundamental cuando se trabaja con aplicaciones serverless, pues asegura que los recursos se creen de manera correcta y óptima bajo este modelo.

#### ¿Cuáles componentes son obligatorios en un template de CloudFormation?

Dentro de un template de AWS CloudFormation, `Resources` es el campo más importante y obligatorio.

- **Razón de obligatoriedad**: Es donde se especifica qué recursos se van a crear. Por ejemplo, al configurar una función Lambda, bases de datos o un bucket, todos deben estar enlistados en esta sección.
- **Elemento central**: Sin este campo, no sería posible definir ni desplegar los recursos que componen tu infraestructura.

### ¿Cómo conectar una función Lambda a una base de datos con outputs?

Al trabajar con funciones Lambda, a menudo es necesario conectarlas a otros recursos como bases de datos DynamoDB.

- **Uso de Outputs**: Mediante los Outputs, se exportan propiedades del recurso creado. Un ejemplo sería exportar el ARN de una función Lambda.
- **Interconexión**: Al crear una base de datos Dynamo, puedes exportar el nombre de la tabla y luego configurar la función Lambda para que use este valor como variable de entorno.

Aquí tienes un ejemplo de cómo se exporta la URL (ARN) de una función Lambda para su posterior uso:

```yaml
Outputs:
  LambdaFunctionArn:
    Description: "ARN de la función Lambda"
    Value: !GetAtt MyLambdaFunction.Arn
    Export:
      Name: !Sub "${AWS::StackName}-LambdaFunctionARN"
```

#### ¿Qué son y cómo se usan los outputs en plantillas CloudFormation?

Los Outputs son cruciales para interconectar diferentes recursos. Pueden ser utilizados para compartir información entre distintos stacks o recursos.

- **Función principal**: Permiten exportar datos significativos de los recursos creados para ser utilizados por otros servicios o recursos.
- **Aplicación práctica**: Si tienes dos recursos, A y B, y deseas conectar B tomando datos de A, podrías usar los Outputs para exportar información relevante de A, que B necesitará como entrada.

La utilización correcta de condiciones, transformaciones y outputs te permitirá diseñar arquitecturas más eficientes y robustas en AWS. La metodología compartida te capacita para establecer ecosistemas de múltiples recursos bien integrados, lo cual es crucial en la administración moderna de IT. ¡Continúa explorando y aplicando estos conceptos en tus proyectos futuros!

## Gestión de Stacks en AWS CloudFormation

### Resumen

#### ¿Qué es un Stack en AWS CloudFormation?

En el contexto de AWS CloudFormation, un Stack es esencialmente un conjunto de recursos que se gestionan de manera unitaria. Esto significa que tú, como desarrollador o ingeniero de devops, puedes gestionar múltiples recursos como una sola entidad. Considera un escenario donde has desplegado una base de datos, una función Lambda, y un bucket de S3: todos estos recursos se agrupan en un único Stack.

CloudFormation asegura que todos estos recursos se creen al mismo tiempo. Si falla la creación de un recurso, por ejemplo, el bucket de S3, el sistema eliminará automáticamente los otros recursos, como la función Lambda y la base de datos, garantizando que no queden recursos a medio configurar.

#### ¿Cómo gestiona los errores AWS CloudFormation?

Un aspecto clave de AWS CloudFormation es su mecanismo automático de rollback. En caso de un fallo en la creación de cualquiera de los recursos dentro del Stack, el sistema eliminará los recursos ya creados, evitando configuraciones parciales. Esta funcionalidad asegura que si algo sale mal, no quedes con una infraestructura a medias que podría causar problemas mayores.

Sin embargo, AWS CloudFormation ofrece flexibilidad al permitir, mediante el "troubleshooting" adecuado, detenerse si un recurso falla. En este escenario, puedes acceder a la consola para identificar y corregir el error, antes de decidir si eliminar los recursos. Usualmente, el comportamiento estándar es remover todos los recursos si uno solo falla al crearse.

#### ¿Qué sucede al borrar un Stack?

Eliminar un Stack es un proceso crítico, ya que implica la eliminación de todos los recursos asociados. Esto significa que, si no tienes cuidado, podrías borrar toda la infraestructura o aplicación en la que estás trabajando. Por lo tanto, gestionar múltiples Stacks requiere atención meticulosa. Siempre verifica y confirma antes de borrar un Stack para asegurarte de que no afectará funcionalidades críticas de la aplicación.

#### ¿Qué es un Drift en AWS CloudFormation?

En AWS CloudFormation, el "Drift" se refiere a la desviación entre la configuración original que fue desplegada y el estado actual en la consola. Por ejemplo, si despliegas una función Lambda y una base de datos pero luego cambias manualmente las configuraciones del bucket S3, esta acción genera un Drift. Estos desajustes no son una buena práctica ya que podrían ocasionar desincronizaciones y problemas en futuras actualizaciones.

Para abordar estos desajustes, los Drifts permiten identificar y corregir estas desviaciones para volver al estado original. Es esencial que todas las actualizaciones y cambios se realicen a través de CloudFormation para mantener una administración centralizada y evitar posibles conflictos.

#### ¿Cómo desplegar un Stack utilizando plantillas?

El despliegue de un Stack en AWS CloudFormation se realiza usualmente mediante plantillas (templates), que pueden crearse en formato JSON o YAML. Estas plantillas describen la infraestructura, permitiendo cargarla a CloudFormation de dos formas:

1. **Carga directa a S3**: Puedes almacenar la plantilla en un bucket de S3 y proporcionar la ruta a CloudFormation.
2. **Carga directa a CloudFormation**: Alternativamente, puedes subir la plantilla directamente a CloudFormation.

Una vez cargada, CloudFormation realiza una validación sobre la sintaxis y los recursos definidos antes de proceder con el despliegue. Esto se lleva a cabo para asegurar que tanto en formato JSON como YAML, las estructuras y definiciones sean correctas y estén bien configuradas.

No te preocupes si los conceptos parecen abstractos ahora; en los laboratorios prácticos que iremos desarrollando, aplicarás estos conocimientos para reforzarlos en la práctica. ¡Te animamos a continuar aprendiendo y a no desistir! La práctica te ayudará a profundizar en estos conceptos y a manejarlos con soltura.

## Creación de Funciones Lambda y Recursos AWS con Stacks

La **creación de funciones Lambda y recursos AWS usando Stacks de CloudFormation** permite automatizar la infraestructura sin errores manuales, logrando integraciones limpias y seguras. A continuación, te explico cómo puedes hacerlo paso a paso.

### 🧱 ¿Qué es un Stack con Lambda?

Un **Stack de CloudFormation** puede incluir una función Lambda, su rol de ejecución, triggers como API Gateway o S3, y otros recursos como DynamoDB. Todo definido en una sola plantilla YAML.

### ⚙️ Ejemplo de plantilla (`template.yaml`)

Aquí tienes un ejemplo básico que:

* Crea una función Lambda en Node.js.
* Crea un rol IAM necesario.
* Crea una tabla DynamoDB.
* Otorga permisos a la Lambda para interactuar con la tabla.

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Stack para Lambda + DynamoDB

Parameters:
  LambdaFunctionName:
    Type: String
    Default: MiFuncionLambda

Resources:

  MiDynamoTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: MiTabla
      AttributeDefinitions:
        - AttributeName: id
          AttributeType: S
      KeySchema:
        - AttributeName: id
          KeyType: HASH
      BillingMode: PAY_PER_REQUEST

  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: LambdaDynamoRole
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      Policies:
        - PolicyName: LambdaDynamoPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - dynamodb:PutItem
                  - dynamodb:GetItem
                  - dynamodb:Scan
                Resource: !GetAtt MiDynamoTable.Arn
              - Effect: Allow
                Action:
                  - logs:CreateLogGroup
                  - logs:CreateLogStream
                  - logs:PutLogEvents
                Resource: "*"

  MiFuncionLambda:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Ref LambdaFunctionName
      Handler: index.handler
      Role: !GetAtt LambdaExecutionRole.Arn
      Runtime: nodejs20.x
      Timeout: 10
      Code:
        ZipFile: |
          exports.handler = async (event) => {
              console.log("Evento recibido:", event);
              return {
                  statusCode: 200,
                  body: JSON.stringify({ mensaje: "Hola desde Lambda!" })
              };
          };

Outputs:
  LambdaARN:
    Description: ARN de la función Lambda
    Value: !GetAtt MiFuncionLambda.Arn
```

### 🚀 Despliegue del stack

Desde la terminal usando AWS CLI:

```bash
aws cloudformation deploy \
  --template-file template.yaml \
  --stack-name StackLambdaDynamo \
  --capabilities CAPABILITY_NAMED_IAM
```

### ✅ Buenas prácticas

* Usa **Parameters** para personalizar nombres y configuraciones.
* Usa **Outputs** para exportar ARN o nombres de recursos.
* Empaqueta código real en ZIP o usa S3 para funciones Lambda más complejas.
* Puedes añadir **triggers** como API Gateway, S3 o eventos de DynamoDB.

### Resumen

#### ¿Cómo se componen los recursos en un Stack?

En este artículo vamos a desentrañar cómo se organizan y trabajan los diversos componentes de un Stack en el contexto de desarrollo en AWS. Es fundamental entender el papel que juega cada recurso, desde bases de datos hasta funciones Lambda, y cómo se integran para formar un sistema robusto y eficiente. Esto no solo te prepara para manejar stacks en AWS sino que amplía tus capacidades de desarrollo en la nube.

#### ¿Qué es una interfaz de metadatos?

Al implementar un stack, uno de los primeros elementos que encontramos es la interfaz de metadatos. Esta interfaz se utiliza para organizar parámetros críticos que se ingresarán posteriormente en la configuración de los recursos.

- **Propiedades de los parámetros**: Para una organización eficiente, se recomienda distinguir entre los parámetros de DynamoDB y las funciones Lambda.
- **Ejemplos de parámetros**: Nombre de la tabla, clave primaria, y nombre de la función Lambda.
- **Propósito**: Simplifica la gestión y evita errores, ya que muestra solo los elementos permitidos mediante listas desplegables.

#### ¿Qué recursos conforman el Stack?

Un Stack eficaz reúne múltiples recursos y aquí te mostramos algunos de los más comunes y necesarios:

1. **DynamoDB:** Base de datos altamente flexible que funciona con los parámetros definidos para su correcta creación.
2. **API Gateway**: Herramienta que se encarga de manejar las solicitudes hechas hacia las bases de datos.
3. **Lambda Function**: Función que ofrece computación sin servidor permitiendo ejecutar código en respuesta a eventos.

Cada uno de estos recursos se configura con parámetros específicos que se han cargado previamente, garantizando así una construcción robusta del servicio.

#### ¿Cómo se manejan las políticas y los roles?

Las políticas y roles son esenciales dentro de un stack para definir permisos y darle seguridad a cada recurso.

- **Rol de Lambda**: Contiene políticas que permiten a la función Lambda interactuar con diferentes servicios.
- **Políticas asociadas**:
 - Permisos para S3 al extraer el código.
 - Permisos para acceder a logs en CloudWatch.
 - Permisos para consultar DynamoDB.
 
La adecuada asignación de roles y políticas asegura que la función Lambda puede funcionar sin fricciones dentro de AWS.

#### Detalles adicionales sobre políticas y permisos

Además de los roles y las políticas básicas, es esencial comprender cómo se manejan los siguientes aspectos:

- **Permisos de ejecución**: Definidos para que un servicio (por ejemplo, API Gateway) pueda activar una función Lambda.
- **Restricciones específicas**: Se determinan de acuerdo con los servicios que la función Lambda necesitará consultar o registrar eventos.

Cada recurso dentro de un stack tiene configuraciones específicas que deben ser tenidas en cuenta para asegurar una operación segura y eficiente de la infraestructura.

#### ¿Por qué son importantes los stacks anidados?

Entender los stacks anidados es esencial para proyectos de mayor escala y complejidad. Estos permiten dividir un stack grande en componentes más pequeños y manejables, facilitando el mantenimiento, la actualización y la reutilización de ciertos componentes.

Encamínate en el aprendizaje continuo y descubre cómo estas herramientas y estructuras pueden simplificar tus proyectos en la nube. ¡La aventura de la infraestructura en AWS apenas comienza!

**Lecturas recomendadas**

[cloudformation/master.yml at composition-non-nested-stacks · czam01/cloudformation · GitHub](https://github.com/czam01/cloudformation/blob/composition-non-nested-stacks/master.yml)

## Despliegue Multi Cuenta con AWS Stack Sets

El **despliegue multi cuenta con AWS StackSets** permite implementar automáticamente plantillas de CloudFormation (stacks) en múltiples cuentas y regiones de AWS desde una cuenta administradora. Es ideal para organizaciones con entornos distribuidos (por ejemplo, dev, test, prod) que comparten infraestructura base como redes, roles IAM, o funciones Lambda.

### 🧰 ¿Qué es AWS StackSets?

Un **StackSet** es un conjunto de instrucciones de infraestructura (una plantilla de CloudFormation) que se puede desplegar y gestionar en múltiples **cuentas AWS** y **regiones** a la vez.

### 📦 Casos de uso comunes

* Configuración uniforme de **CloudTrail** o **AWS Config** en todas las cuentas.
* Despliegue de roles IAM o políticas estándar.
* Infraestructura compartida (como buckets S3, SNS topics o tablas DynamoDB).

### 🧱 Arquitectura: StackSet con cuenta organizacional

1. **Cuenta Administradora (Management Account):** Crea y administra el StackSet.
2. **Cuentas Objetivo (Target Accounts):** Reciben los stacks.
3. **Organización AWS Organizations:** Simplifica permisos usando la opción de “**self-managed permissions**” o “**service-managed permissions**”.

### 🔐 Requisitos previos

### A. Organización habilitada con AWS Organizations

```bash
aws organizations enable-aws-service-access \
  --service-principal cloudformation.stacksets.amazonaws.com
```

### B. Crear una plantilla base

Ejemplo: Plantilla para crear un bucket S3 en cada cuenta

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Crear un bucket S3 estándar

Resources:
  StandardS3Bucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub 'standard-bucket-${AWS::AccountId}'
      VersioningConfiguration:
        Status: Enabled
```

### 🚀 Creación del StackSet

Puedes hacerlo desde:

### A. Consola de AWS:

1. Ve a **CloudFormation > StackSets**
2. Clic en **Create StackSet**
3. Proporciona la plantilla (S3 URL o YAML directamente)
4. Elige si usas **permisos administrados por servicio (recomendado)** o self-managed
5. Selecciona unidades organizacionales (OUs) o IDs de cuentas
6. Elige regiones

### B. AWS CLI:

```bash
aws cloudformation create-stack-set \
  --stack-set-name MiStackMultiCuenta \
  --template-body file://plantilla.yaml \
  --permission-model SERVICE_MANAGED \
  --capabilities CAPABILITY_NAMED_IAM
```

Luego:

```bash
aws cloudformation create-stack-instances \
  --stack-set-name MiStackMultiCuenta \
  --deployment-targets OrganizationalUnitIds=ou-abc1-xyz123 \
  --regions us-east-1 us-west-2
```

### 🛠️ Monitoreo y Actualización

* Puedes ver el estado de cada stack por cuenta y región.
* Para hacer actualizaciones, solo cambias la plantilla en el StackSet y se actualizan automáticamente todas las cuentas.

### ✅ Ventajas

* Uniformidad: Infraestructura estándar en todo el entorno.
* Seguridad: Administración centralizada de permisos.
* Escalabilidad: Agrega nuevas cuentas sin reprocesar todo.
* Automatización: Ideal para estructuras CI/CD multi cuenta.

### Resumen

#### ¿Cómo se componen los recursos en un Stack?

En este artículo vamos a desentrañar cómo se organizan y trabajan los diversos componentes de un Stack en el contexto de desarrollo en AWS. Es fundamental entender el papel que juega cada recurso, desde bases de datos hasta funciones Lambda, y cómo se integran para formar un sistema robusto y eficiente. Esto no solo te prepara para manejar stacks en AWS sino que amplía tus capacidades de desarrollo en la nube.

#### ¿Qué es una interfaz de metadatos?

Al implementar un stack, uno de los primeros elementos que encontramos es la interfaz de metadatos. Esta interfaz se utiliza para organizar parámetros críticos que se ingresarán posteriormente en la configuración de los recursos.

- **Propiedades de los parámetros**: Para una organización eficiente, se recomienda distinguir entre los parámetros de DynamoDB y las funciones Lambda.
- **Ejemplos de parámetros**: Nombre de la tabla, clave primaria, y nombre de la función Lambda.
- **Propósito**: Simplifica la gestión y evita errores, ya que muestra solo los elementos permitidos mediante listas desplegables.

#### ¿Qué recursos conforman el Stack?

Un Stack eficaz reúne múltiples recursos y aquí te mostramos algunos de los más comunes y necesarios:

1. **DynamoDB:** Base de datos altamente flexible que funciona con los parámetros definidos para su correcta creación.
2. **API Gateway**: Herramienta que se encarga de manejar las solicitudes hechas hacia las bases de datos.
3. **Lambda Function**: Función que ofrece computación sin servidor permitiendo ejecutar código en respuesta a eventos.

Cada uno de estos recursos se configura con parámetros específicos que se han cargado previamente, garantizando así una construcción robusta del servicio.

#### ¿Cómo se manejan las políticas y los roles?

Las políticas y roles son esenciales dentro de un stack para definir permisos y darle seguridad a cada recurso.

- **Rol de Lambda**: Contiene políticas que permiten a la función Lambda interactuar con diferentes servicios.
- **Políticas asociadas**:
 - Permisos para S3 al extraer el código.
 - Permisos para acceder a logs en CloudWatch.
 - Permisos para consultar DynamoDB.
 
La adecuada asignación de roles y políticas asegura que la función Lambda puede funcionar sin fricciones dentro de AWS.

#### Detalles adicionales sobre políticas y permisos

Además de los roles y las políticas básicas, es esencial comprender cómo se manejan los siguientes aspectos:

- **Permisos de ejecución**: Definidos para que un servicio (por ejemplo, API Gateway) pueda activar una función Lambda.
- **Restricciones específicas**: Se determinan de acuerdo con los servicios que la función Lambda necesitará consultar o registrar eventos.

Cada recurso dentro de un stack tiene configuraciones específicas que deben ser tenidas en cuenta para asegurar una operación segura y eficiente de la infraestructura.

#### ¿Por qué son importantes los stacks anidados?

Entender los stacks anidados es esencial para proyectos de mayor escala y complejidad. Estos permiten dividir un stack grande en componentes más pequeños y manejables, facilitando el mantenimiento, la actualización y la reutilización de ciertos componentes.

Encamínate en el aprendizaje continuo y descubre cómo estas herramientas y estructuras pueden simplificar tus proyectos en la nube. ¡La aventura de la infraestructura en AWS apenas comienza!

## Despliegue Multi Cuenta con AWS CloudFormation y DynamoDB

El **despliegue multi cuenta con AWS CloudFormation y DynamoDB** te permite provisionar tablas DynamoDB en múltiples cuentas y regiones de AWS de forma centralizada y automatizada. Esto es útil en organizaciones que gestionan varios entornos (desarrollo, QA, producción) o tienen estructuras de cuentas distribuidas.

### ✅ ¿Qué necesitas?

### 1. **Una plantilla CloudFormation (YAML o JSON)**

Define la tabla DynamoDB con sus atributos, claves, modo de facturación, etc.

### 2. **StackSets en AWS CloudFormation**

Permite desplegar esta plantilla en múltiples cuentas y/o regiones.

### 📘 Ejemplo de plantilla CloudFormation para DynamoDB

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Crear tabla DynamoDB multi cuenta

Parameters:
  TableName:
    Type: String
  PartitionKey:
    Type: String
    Default: id

Resources:
  DynamoTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Ref TableName
      AttributeDefinitions:
        - AttributeName: !Ref PartitionKey
          AttributeType: S
      KeySchema:
        - AttributeName: !Ref PartitionKey
          KeyType: HASH
      BillingMode: PAY_PER_REQUEST
      SSESpecification:
        SSEEnabled: true

Outputs:
  DynamoTableName:
    Value: !Ref DynamoTable
```

### 🛠️ ¿Cómo desplegarlo con StackSets?

### A. Desde la **Consola**:

1. Ir a **CloudFormation > StackSets**
2. Click en **Create StackSet**
3. Subir tu plantilla YAML
4. Usa **Service-managed permissions** si tienes AWS Organizations
5. Selecciona las cuentas o unidades organizacionales (OU)
6. Elige las regiones (ej. `us-east-1`, `us-west-2`)
7. Ingresa los valores de parámetros (`TableName`, `PartitionKey`)

### B. Desde la **CLI**:

```bash
aws cloudformation create-stack-set \
  --stack-set-name DynamoMultiCuenta \
  --template-body file://dynamo.yaml \
  --parameters ParameterKey=TableName,ParameterValue=MiTabla \
  --permission-model SERVICE_MANAGED \
  --capabilities CAPABILITY_NAMED_IAM
```

Luego:

```bash
aws cloudformation create-stack-instances \
  --stack-set-name DynamoMultiCuenta \
  --deployment-targets OrganizationalUnitIds=ou-abc1-xyz123 \
  --regions us-east-1 us-west-2
```

### 🔐 Permisos requeridos

### Cuenta administradora (StackSet):

* Permisos para crear recursos en CloudFormation.
* Permisos para acceder a cuentas miembro.

### Cuentas destino:

* Se requiere confianza si se usa el modo **self-managed** (se configuran manualmente los roles de ejecución).

### 🚀 Beneficios del despliegue multi cuenta

* **Estandarización**: misma configuración de DynamoDB en todas las cuentas.
* **Escalabilidad**: añade nuevas cuentas fácilmente.
* **Seguridad**: configuración de cifrado y control de acceso unificado.
* **Automatización**: despliegue en múltiples regiones y cuentas desde una sola acción.

### Resumen

#### ¿Cómo hacer un despliegue multicuentas con Stax Edit?

Los entornos de infraestructura para aplicaciones grandes requieren un manejo especializado de la seguridad y la administración de múltiples cuentas. En este artículo, vamos a detallar cómo realizar un despliegue multicuentas utilizando Stax Edit, dentro de una infraestructura de Amazon Web Services (AWS). Este proceso es especialmente relevante para empresas grandes que manejan aplicaciones complejas y necesitan un alto nivel de seguridad.

#### ¿Qué considerar antes de comenzar?

Antes de iniciar con el despliegue multicuentas, es esencial tener un conjunto de cuentas bien estructuradas e interconectadas. Aquí están los pasos preliminares:

1. **Estructurar las cuentas**: Define las cuentas necesarias, como servicios compartidos, desarrollo (Dev), testing (QA), preproducción y producción.
2. **Configurar permisos y roles**: Asegúrate de que las cuentas están correctamente configuradas con roles específicos para administración y permisos de ejecución.
3. **Entrega de plantilla**s: Decide si utilizarás plantillas desde un Amazon S3 o si cargarás una plantilla personalizada desde tu equipo.

#### ¿Qué es AWS Landing Zone?

AWS Landing Zone es un servicio que permite agrupar diferentes cuentas y ofrecer un acceso común utilizando Active Directory. Esto facilita la administración centralizada de diversas cuentas, lo que es clave para el éxito del despliegue multicuentas.

#### ¿Cuáles son los pasos para crear un nuevo stack?

Para crear un nuevo stack y realizar el despliegue, sigue las siguientes instrucciones:

- **Cargar la plantilla**: Ve a la consola de administración de tu cuenta de servicios compartidos y selecciona la plantilla desde donde quieras cargarla.

```yaml
# Ejemplo de YAML para DynamoDB
Resources:
  MyDynamoDBTable:
    Type: "AWS::DynamoDB::Table"
    Properties:
      TableName: "Platzi"
      AttributeDefinitions:
        - AttributeName: "ID"
          AttributeType: "S"
      KeySchema:
        - AttributeName: "ID"
          KeyType: "HASH"
```

- **Especificar roles**: Determina los roles de administración y ejecución necesarios para el despliegue.

#### ¿Cómo definir las cuentas y regiones para el despliegue?

Debes especificar las cuentas y la región en la cual se encuentra tu infraestructura. Para esto necesitas:

- Identificar tus cuentas con su número único.
- Elegir una región compatible (por ejemplo, Virginia para el caso de Estados Unidos).

#### ¿Qué configuración adicional se debe tener en cuenta?

Al realizar el despliegue, es crucial establecer:

-** Cantidad de cuentas concurrentes**: Define cuántas cuentas se desplegarán al mismo tiempo.
- **Condiciones de fallo**: Configura si el despliegue se detendrá si una cuenta falla.

#### ¿Cómo se verifica el estado del despliegue?

Una vez iniciado el despliegue, se deben realizar revisiones:

- **Estado en la consola**: Accede a la consola de AWS y verifica el estado del despliegue para cada cuenta involucrada.
- **Revisión de la tabla DynamoDB**: Confirma que la tabla se ha creado correctamente en cada cuenta.

#### ¿Cuáles son los beneficios de usar Stax Edit para despliegues multicuentas?

El despliegue multicuentas con Stax Edit ofrece varias ventajas:

- **Centralización**: Permite centralizar el control de recursos en múltiples cuentas, optimizando la administración.
- **Seguridad**: Mejora la seguridad al separar los ambientes y asignar roles específicos.
- **Escalabilidad**: Facilita el crecimiento de la infraestructura, permitiendo agregar recursos y cuentas con flexibilidad.

Con estos pasos, podrás gestionar despliegues de aplicaciones de gran escala en entornos seguros y organizados, lo que es fundamental para el éxito continuo de las aplicaciones empresariales. ¡Adelante, y sigue aprendiendo para mejorar tus habilidades en la gestión de infraestructuras en la nube!

## Uso de Nested Stacks para superar límites en CloudFormation

El **uso de Nested Stacks (stacks anidados)** en AWS CloudFormation es una práctica avanzada que te permite organizar, reutilizar y **superar los límites de tamaño y complejidad** en plantillas de infraestructura. Aquí te explico cómo funcionan, sus ventajas y cuándo usarlos:

### ✅ ¿Qué son los Nested Stacks?

Un **Nested Stack** es un stack que se define como un recurso dentro de otra plantilla principal. Permite dividir una plantilla grande en archivos más pequeños y reutilizables. Cada substack es una plantilla propia que se despliega como parte del stack principal.

### 🎯 ¿Por qué usar Nested Stacks?

### 1. **Superar límites de CloudFormation**

CloudFormation impone los siguientes límites (por defecto):

* Tamaño máximo de plantilla: **1 MB**
* Número máximo de recursos por stack: **500**

Con Nested Stacks puedes **dividir la lógica en varios archivos**, cada uno con su propio conjunto de recursos.

### 2. **Reutilización de plantillas**

Puedes crear componentes comunes como VPCs, buckets S3, roles IAM, etc., y reutilizarlos en múltiples stacks.

### 3. **Modularidad y mantenibilidad**

Organizas tu infraestructura en módulos lógicos:

* `network.yaml`
* `database.yaml`
* `compute.yaml`
* `app.yaml`

Esto mejora la legibilidad y facilita el mantenimiento.

### 🧱 Ejemplo de uso básico de Nested Stacks

### 📁 Estructura de archivos

```
main-stack.yaml
network-stack.yaml
compute-stack.yaml
```

### `main-stack.yaml`

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Resources:
  NetworkStack:
    Type: AWS::CloudFormation::Stack
    Properties:
      TemplateURL: https://s3.amazonaws.com/mybucket/network-stack.yaml

  ComputeStack:
    Type: AWS::CloudFormation::Stack
    Properties:
      TemplateURL: https://s3.amazonaws.com/mybucket/compute-stack.yaml
```

### 🔗 Cómo usar Nested Stacks

1. **Sube las plantillas hijas a un bucket S3** accesible por CloudFormation.
2. Usa `AWS::CloudFormation::Stack` en la plantilla principal.
3. Define la propiedad `TemplateURL` apuntando al archivo en S3.
4. Pasa parámetros si es necesario usando `Parameters`.

### 📌 Buenas prácticas

* Usa nombres descriptivos para los stacks y sus parámetros.
* Centraliza templates reutilizables en un bucket S3 versionado.
* Valida individualmente cada substack antes de integrarlo.
* Documenta dependencias y relaciones entre stacks.

### 🚧 Consideraciones

* Nested Stacks **cuentan como recursos**, así que ten en cuenta el límite de 500 recursos por stack.
* Si un Nested Stack falla, **todo el stack principal puede revertirse**.
* Requiere buena gestión de versiones y cambios para evitar problemas de dependencias.

### Resumen

#### ¿Qué son los nested stacks y por qué son necesarios?

Los nested stacks son una importante solución para superar las limitaciones en el uso de AWS Cloud Formation, especialmente cuando debemos administrar una gran cantidad de recursos en un solo stack. La necesidad de utilizarlos surge debido al límite que existe en la cantidad de elementos que podemos manejar: 100 mappings y 200 recursos por stack. Cuando superamos estas cifras, necesitamos un enfoque diferente y es ahí donde entran en juego los nested stacks.

### ¿Cómo funcionan los nested stacks?

Imagina un stack maestro que sirve como un controlador principal que se comunica con varios stacks más pequeños, cada uno manejando menos de 200 recursos. Estos stacks pequeños pueden tener muchos más mappings y ayudas específicas para cada contexto. Esto no solo nos permite superar los límites impuestos por AWS, sino que también organiza y segmenta los recursos de manera efectiva.

#### Ventajas de utilizar nested stacks

Utilizar nested stacks no solo ayuda a superar limitaciones numéricas:

- **Orden y organización**: Al dividir recursos en stacks separados, cada uno tiene su propósito y contexto claro, facilitando el entendimiento y manejo de los recursos.
- **Facilidad de uso**: Con stacks más pequeños, las operaciones de troubleshooting (resolver problemas) se vuelven más simples y rápidas.
- **Interacción de recursos**: A través de los outputs, podemos comunicar stacks entre sí, logrando que los recursos interactúen de manera eficiente.

#### ¿Cómo los nested stacks benefician proyectos del día a día?

Cuando gestionamos proyectos complejos que incluyen, por ejemplo, un API Gateway, una función Lambda, un repositorio de código y un DynamoDB, los nested stacks nos permiten desplegar estos recursos de manera eficiente y organizada:

En un escenario sin nested stacks, todos los recursos se despliegan desde una única plantilla, complicando los cambios y la reutilización de recursos. Pero con nested stacks, un stack maestro controla la creación y gestión de stacks individuales para cada componente como Lambda o DynamoDB. Esto permite replicar, modificar y reutilizar componentes fácilmente sin complicaciones.

#### Escenario práctico: Organización de recursos

Un caso práctico es el siguiente: Imagina un proyecto que necesita desplegar recursos alojados en S3. Cada recurso puede manejarse de forma directa mediante su stack, lo que permite una gestión granular y evita sobrecarga en el stack principal. A través de direcciones en la AWS S3, los nested stacks se encargan de manejar cada componente de forma segregada.

#### Implementación de un stack maestro con nested stacks

Un stack maestro típicamente incluye una imagen de cómo debe estructurarse el proyecto, señalando:

- La API y su diseño dentro del stack.
- La función Lambda y sus dependencias.
- Integración de DynamoDB como base de datos.
- Uso de un bucket en S3 como almacenamiento central.

Este enfoque promueve la reutilización de recursos y una implementación flexible, reducida en complejidad.

#### Conclusión: Uso recomendado de nested stacks

Utilizar nested stacks se convierte en una práctica esencial para cualquier proyecto con dimensionamiento considerable. No solo optimiza la gestión de recursos y su reutilización, sino que también contribuye a un diseño claro y ordenado que facilitan el mantenimiento y la escalabilidad. Para explorar más sobre este enfoque, el repositorio del curso ofrece ejemplos donde podrás reemplazar URLs y ajustar stacks a tus necesidades, maximizando el potencial del uso de AWS Cloud Formation en tus proyectos.

## Despliegue de Recursos AWS con CloudFormation para Votan Help

Aquí tienes una guía práctica para el **Despliegue de Recursos AWS con CloudFormation para Votan Help**, una aplicación (hipotética o real) que puede incluir APIs, bases de datos y lógica de negocio en Lambda. El enfoque se basa en **Infraestructura como Código (IaC)** utilizando AWS CloudFormation.

### ✅ Objetivo

Desplegar de forma automatizada los recursos de AWS necesarios para **Votan Help**, incluyendo:

* API Gateway (REST o HTTP)
* AWS Lambda Functions
* DynamoDB (para almacenamiento)
* Roles IAM (permisos)
* Outputs exportables

### 📦 Estructura de Recursos para Votan Help

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Recursos AWS para Votan Help

Parameters:
  TableName:
    Type: String
    Default: VotanHelpTable

Resources:

  ### DynamoDB Table ###
  VotanHelpTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Ref TableName
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: id
          AttributeType: S
      KeySchema:
        - AttributeName: id
          KeyType: HASH
      SSESpecification:
        SSEEnabled: true

  ### IAM Role for Lambda ###
  VotanHelpLambdaRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: votan-help-lambda-role
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      Policies:
        - PolicyName: votan-help-dynamodb-access
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action:
                  - dynamodb:*
                Resource: "*"
              - Effect: Allow
                Action:
                  - logs:*
                Resource: "*"

  ### Lambda Function ###
  VotanHelpFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: votan-help-handler
      Handler: index.handler
      Role: !GetAtt VotanHelpLambdaRole.Arn
      Runtime: nodejs18.x
      Timeout: 10
      Code:
        ZipFile: |
          exports.handler = async (event) => {
            return {
              statusCode: 200,
              body: JSON.stringify({ message: "Votan Help is working!" })
            };
          };

  ### API Gateway HTTP API ###
  VotanHelpApi:
    Type: AWS::ApiGatewayV2::Api
    Properties:
      Name: votan-help-api
      ProtocolType: HTTP
      Target: !GetAtt VotanHelpFunction.Arn

  ### API Integration Permission ###
  LambdaPermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref VotanHelpFunction
      Action: lambda:InvokeFunction
      Principal: apigateway.amazonaws.com
      SourceArn: !Sub arn:aws:execute-api:${AWS::Region}:${AWS::AccountId}:${VotanHelpApi}/*

Outputs:
  ApiEndpoint:
    Value: !Sub "https://${VotanHelpApi}.execute-api.${AWS::Region}.amazonaws.com"
    Export:
      Name: VotanHelpAPIEndpoint
  TableName:
    Value: !Ref VotanHelpTable
    Export:
      Name: VotanHelpDynamoDB
```

### 🚀 Despliegue

1. Guarda el archivo como `votan-help-stack.yaml`
2. Súbelo a CloudFormation vía consola o CLI:

```bash
aws cloudformation create-stack \
  --stack-name votan-help-stack \
  --template-body file://votan-help-stack.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

### 🛠️ Personalización

Puedes añadir más funcionalidades como:

* Autenticación con Cognito
* API Gateway con rutas más complejas
* Otros entornos (dev, prod) mediante parámetros

### Resumen

#### ¿Cómo desplegar recursos en AWS con un Stack?

Aprovechar las capacidades de Amazon Web Services para desplegar aplicaciones y recursos es esencial para cualquier desarrollador. En este laboratorio, centrándonos en el despliegue de Votan Help, aprenderás a usar un Stack para crear y configurar recursos como API Gateway, Lambda y DynamoDB de manera eficiente. La estructura y coordinación de estos elementos es crucial para un proyecto exitoso, y aquí te mostramos cómo lograrlo.

#### ¿Cómo clonar el repositorio de código?

Para empezar, es importante contar con el código fuente correcto. Dirígete a tu repositorio de código y clónalo siguiendo estos pasos:

1. Copia la URL del repositorio.
2. Abre tu terminal y utiliza el comando git clone seguido de la URL copiada.
3. Verifica que la clonación haya sido exitosa revisando la estructura de carpetas del repositorio en tu sistema local.

#### ¿Cómo preparar el entorno de AWS S3 para el proyecto?

Amazon S3 es un servicio de almacenamiento de objetos esencial donde se guarda el código y los archivos necesarios para Lambda. Aquí está cómo configurarlo:

1. Accede a la consola de Amazon S3 y crea un bucket si no lo tienes ya. Solo necesitas especificar el nombre.
2. Una vez creado el bucket, carga el código comprimido de la función Lambda en formato `.zip` o `.pkg`. Este archivo será clave para desplegar la función Lambda.

#### ¿Cómo cargar y configurar el Stack en AWS CloudFormation?

CloudFormation se utiliza para desplegar y manejar múltiples recursos en AWS con un solo template. A continuación, te explicamos cómo hacerlo:

- Copia la URL completa del archivo de template master desde el repositorio de código en S3.
- Accede a la consola de AWS CloudFormation y selecciona "Crear Stack".
- Proporciona la URL copiada como el origen de tu template.
- Completa los campos requeridos, como el nombre del Stack (por ejemplo, `VotaNextMaster`) y detalles específicos del proyecto como el nombre de la tabla de DynamoDB, llave principal, nombre de la función Lambda y bucket S3 donde se almacena el código.

#### ¿Qué aspectos son vitales al crear un Stack?

Durante el proceso de creación del Stack, hay varios aspectos importantes a considerar para evitar errores. Estos incluyen:

- Asegúrate de seleccionar los permisos IaaM necesarios, especialmente al crear roles y políticas.
- Revisa cada paso antes de continuar para asegurar que los parámetros y configuraciones son correctos.
- Controla el proceso de despliegue, verificando que cada recurso se crea con éxito a través del estatus de CloudFormation.

#### ¿Cómo verificar que los recursos han sido creados correctamente?

Tras la creación del Stack, es esencial asegurarse de que cada recurso funciona correctamente:

- **Lambd**a: Confirma que la función Lambda está desplegada y configurada para ejecutarse con el runtime adecuado (e.g., Python 3.7). Verifica las políticas y permisos.

- **DynamoDB**: Revisa que la tabla DynamoDB existe y tiene las claves correctas configuradas. Es importante tener lo principal, como cédula de ciudadanía, preparado para consultas.

- **API Gateway**: Verifica la creación de los endpoints en API Gateway, asegurándote de que están correctamente integrados con la función Lambda.

Este enfoque detallado te ofrece la claridad y seguridad necesarias para implementar proyectos en AWS, habilitándote para llevar tus aplicaciones al siguiente nivel con confianza. Investiga más sobre cada componente y sigue practicando para dominar esta herramienta indispensable en el desarrollo en la nube. ¡Adelante, sigue aprendiendo y potenciando tus proyectos!

## Verificación y Pruebas de Recursos en AWS Lambda y API Gateway

Para realizar la **verificación y pruebas de recursos en AWS Lambda y API Gateway**, debes seguir una serie de pasos que aseguran que tanto tus funciones Lambda como tus endpoints de API estén funcionando correctamente. A continuación, te presento una guía clara y estructurada:

### ✅ 1. **Verifica la función Lambda**

#### A. Desde la consola de AWS

1. Ve a la consola de **AWS Lambda**.
2. Selecciona la función que deseas probar.
3. Haz clic en **"Test"**.
4. Crea un evento de prueba con un payload de ejemplo (JSON).
5. Haz clic en **"Invoke"** o **"Test"**.
6. Revisa los resultados:

   * **Response body**: La respuesta de tu función.
   * **Logs**: Verifica errores en el tab de logs.

#### B. Desde la CLI

```bash
aws lambda invoke \
  --function-name NOMBRE_FUNCION \
  --payload '{"key":"value"}' \
  response.json
cat response.json
```

### ✅ 2. **Verifica configuración del trigger de API Gateway**

Si tu Lambda está conectada a una API Gateway:

#### A. En la consola de API Gateway:

1. Ve a **Amazon API Gateway**.
2. Elige tu API (REST o HTTP).
3. Verifica:

   * Métodos definidos (GET, POST, etc.).
   * Integración con Lambda (check en “Integration Request”).
   * Si hay un stage desplegado (`prod`, `dev`, etc.).

#### B. Asegúrate de haber desplegado la API

En el caso de APIs REST:

* Selecciona **“Actions” → “Deploy API”**
* Elige o crea un stage (`prod`, `test`, etc.)

### ✅ 3. **Prueba el endpoint de API Gateway**

#### A. Desde el navegador (GET)

```
https://<api-id>.execute-api.<region>.amazonaws.com/<stage>/<resource>
```

#### B. Con `curl` (POST o GET)

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"key":"value"}' \
  https://<api-id>.execute-api.<region>.amazonaws.com/<stage>/<resource>
```

#### C. Desde Postman

1. Crea una nueva solicitud (GET/POST).
2. Pega la URL del endpoint.
3. En “Body”, selecciona `raw` → `JSON`, y agrega tu payload.

### ✅ 4. **Consulta logs en CloudWatch**

Si algo no funciona:

1. Ve a **CloudWatch → Logs → Log groups**.
2. Busca el grupo `/aws/lambda/tu-funcion`.
3. Revisa las últimas invocaciones para identificar errores.

### 🛠️ Consejos útiles

* Verifica los permisos de la Lambda (`IAM Role`) para asegurarte de que puede ser invocada desde API Gateway.
* Usa `aws apigateway get-rest-apis` y `get-stages` para verificar despliegues vía CLI.
* Usa `aws logs tail` para ver logs en tiempo real:

  ```bash
  aws logs tail /aws/lambda/mi-funcion --follow
  ```

### Resumen

#### ¿Cómo verificar el funcionamiento de los recursos en AWS?

La comprensión y verificación del funcionamiento de los recursos en AWS es crucial para garantizar una implementación efectiva y obtener resultados óptimos. A menudo, este proceso puede parecer intimidante al principio, pero con un enfoque estructurado y el uso de herramientas adecuadas, cualquier desarrollador puede manejarlo con confianza. En este artículo, te guiaré sobre cómo verificar los recursos de Bota net mediante AWS, utilizando varios servicios como DynamoDB y AWS Lambda.

#### ¿Cómo gestionar y verificar datos en DynamoDB?

Primero, para empezar la verificación, nuestra tarea inicial es acudir a la base de datos DynamoDB. Una vez dentro, es fundamental identificar las tablas creadas, en nuestro caso, la tabla 'Platzi'. Tras esto, accedemos a la sección 'Items' para agregar y gestionar datos, con pasos sencillos pero poderosos que permiten mantener un control total de la información.

- **Agregar elementos**: Ingresar a la tabla y, en la sección de 'Items', crear un nuevo registro.
- **Campos a considerar**: Se manejan campos como cédula de ciudadanía, nombre, dirección y barrio. Por ejemplo, puede utilizar nombres como Carlos Zambrano y direcciones como Calle One, Two, Three.
- **Guardar cambios**: Finalmente, presionar el botón Save para registrar todos los datos nuevos.

#### ¿Cómo hacer pruebas con AWS Lambda?

Después de haber registrado los datos en DynamoDB, el siguiente paso es verificar la funcionalidad en AWS Lambda. Este proceso asegura que las funciones Lambda puedan acceder y recuperar datos de DynamoDB de manera eficiente.

- **Crear un evento de prueba**: Asignar un nombre al evento, como "Mi Prueba", y enviar un JSON con la información necesaria, en este caso, el número de cédula.
- **Probar la función Lambda**: Ejecutar el evento de prueba y verificar la respuesta. Lambda debería devolver la información completa del registro consultado desde DynamoDB.

#### ¿Cómo integrar con API Gateway?

La integración con API Gateway es esencial para ampliar el acceso al mundo exterior. A través de API Gateway puedes exponer tus funciones Lambda y hacerlas accesibles desde aplicaciones externas o clientes web.

- **Configurar el API Gateway**: Verificar que tenga acceso a la función Lambda configurada.
- **Hacer una solicitud de prueba**: Utilizar herramientas como cURL o Postman para enviar solicitudes hacia la API Gateway. Por ejemplo, un cURL puede enviarse así:

`curl -X POST -H "Content-Type: application/json" -d '{"cédula":"111"}' [API_URL]` en linux

Esta solicitud comprobará que la comunicación entre API Gateway y Lambda sea efectiva y el sistema devuelva los datos esperados correctamente.

#### ¿Qué hacer si quieres profundizar más?

Para aquellos que deseen expandir sus habilidades, es recomendable sumergirse en cursos especializados sobre bases de datos en AWS. Explorar las capacidades de AWS te otorgará una perspectiva más amplia y control sobre la arquitectura de tu aplicación.

Además, anexo a este aprendizaje técnico, recuerda siempre analizar factores como el tiempo de ejecución y la eficiencia de los recursos, ya que AWS cobra en función de la duración y el manejo de la memoria durante la ejecución de las funciones Lambda. ¡Continúa aprendiendo y experimentando para optimizar continuamente tu infraestructura en la nube!

## Despliegue de Recursos con Stacks Anidados en AWS

El **despliegue de recursos con Stacks Anidados (Nested Stacks)** en AWS CloudFormation es una técnica poderosa para organizar y reutilizar plantillas de infraestructura como código (IaC). A continuación te explico en qué consisten, cómo se usan y los beneficios clave:

### 🧩 ¿Qué son los Stacks Anidados en CloudFormation?

Los **Nested Stacks** son stacks definidos dentro de otro stack principal (parent stack) usando el recurso `AWS::CloudFormation::Stack`. Permiten **modularizar** plantillas grandes o complejas dividiéndolas en partes reutilizables.

### ✅ Ventajas de usar Stacks Anidados

* 🔁 **Reutilización**: Puedes usar la misma plantilla en diferentes entornos.
* 🧼 **Organización**: Mantienes tu infraestructura modular y más legible.
* 🔍 **Mantenimiento**: Actualizaciones más simples al modificar solo un stack hijo.
* 📏 **Límites**: Ayudan a superar límites de longitud en plantillas (por línea y tamaño total).

### 📁 Estructura básica de un Nested Stack

### 🧾 Archivo principal (`main-template.yaml`)

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Resources:
  MiStackDynamoDB:
    Type: AWS::CloudFormation::Stack
    Properties:
      TemplateURL: https://bucket-s3.s3.amazonaws.com/dynamo-template.yaml
      Parameters:
        NombreDynamo: MiTabla
        DynamoAtributo: id
```

### 🧾 Stack hijo (`dynamo-template.yaml`)

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Parameters:
  NombreDynamo:
    Type: String
  DynamoAtributo:
    Type: String
Resources:
  MiTabla:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Ref NombreDynamo
      AttributeDefinitions:
        - AttributeName: !Ref DynamoAtributo
          AttributeType: S
      KeySchema:
        - AttributeName: !Ref DynamoAtributo
          KeyType: HASH
      BillingMode: PAY_PER_REQUEST
```

### 🚀 Pasos para el despliegue

1. ✅ **Sube el stack hijo a S3**

   * Usa un bucket accesible desde CloudFormation.
   * Asegúrate de que el archivo `.yaml` esté en una región compatible.

2. 🏗️ **Despliega el stack principal**

   ```bash
   aws cloudformation create-stack \
     --stack-name stack-principal \
     --template-body file://main-template.yaml \
     --capabilities CAPABILITY_NAMED_IAM
   ```

### 🛑 Consideraciones importantes

* Asegúrate de que la URL en `TemplateURL` esté **pública o accesible desde la cuenta de AWS**.
* Los **parámetros del stack hijo** deben definirse correctamente en el stack principal.
* Puedes usar **salidas (`Outputs`)** en stacks hijos y exportarlas para otros stacks si usas `Fn::ImportValue`.

### Resumen

#### ¿Cómo desplegar recursos en Stacks anidados?

Desplegar recursos en Stacks anidados es una técnica poderosa que te permite organizar y gestionar tus proyectos con mayor eficiencia y claridad. Imagina tener control sobre funciones Lambda, API Gateway y DynamoDB desde un Stack maestro, logrando así una estructura ordenada y fácil de expandir. Este método te ayuda a reutilizar componentes, lo que es especialmente útil en proyectos de gran escala.

#### ¿Qué es un Stack maestro?

Un Stack maestro en Amazon CloudFormation es un conjunto de recursos agrupados. Permite gestionar múltiples Stacks anidados que son instancias individuales de recursos como bases de datos, funciones Lambda y API Gateway.

- **Componentes del Stack:**
 - Lambda Function
 - API Gateway
 - DynamoDB
 
El Stack maestro facilita el control centralizado de recursos independientes, permitiendo enviar parámetros a cada uno, incluso si son completamente diferentes.

#### ¿Cómo gestionar dependencias con DependsOn?

Al desplegar APIs o bases de datos, es fundamental controlar el orden de creación de los recursos. Utilizando la propiedad `DependsOn`, aseguras que ciertos recursos no se creen antes de que los necesarios estén disponibles, lo cual es esencial para evitar errores y mejorar la eficiencia.

```xml
Resources:
  MyApiGateway:
    Type: AWS::ApiGateway::RestApi
    Properties: 
      [Propiedades]
    DependsOn: MyDynamoDB
```

#### ¿Cómo estructurar el código para un Stack anidado?

1. **Estructura del Repositorio:**

 - Repositorio con carpeta nested para Stacks anidados.
 - Código de Lambda en la carpeta lambda_code.

2. Carga del Código:

 - Asegúrate de que el bucket S3 está vaciado y luego sube el nuevo código.
 - Selección y carga de archivos a través de la consola de AWS S3.

3. **Implementación y Configuración**:

 - Clonar el repositorio usando Git:
 
`git clone [URL del repositorio]`

 - Desplegar el template desde el bucket S3 seleccionando Crear Stack en la consola de CloudFormation.
 - Proveer detalles como el nombre del Stack y parámetros funcionales específicos.

### ¿Cómo verificar la creación de recursos?

Después de haber creado el Stack maestro y sus Stacks anidados, es crucial verificar que los recursos se hayan creado correctamente.

- **Consola de DynamoDB**: Comprueba la creación correcta de la tabla.

- **Consola de Lambda**: Asegúrate que las funciones se hayan creado y configurado adecuadamente.

- **Consola de API Gateway**: Confirma que las APIs estén conectadas a las funciones Lambda correctas.

Estas verificaciones garantizan que todos los recursos estén listos para su utilización en la aplicación o sistema que estás desarrollando. Además, utilizar Stacks anidados simplifica futuras actualizaciones, permitiendo replicar fácilmente cualquier recurso específico para otros proyectos.

Recuerda siempre optimizar y estructurar tus proyectos considerando la flexibilidad, claridad y reusabilidad de los recursos, lo cual será un valor añadido en tu ámbito profesional. ¡Avanza en tu aprendizaje y mejora continua!

**Nota**: en la carpeta nsted en el archivo master.yml colocar las url de los Template

## Consulta y Verificación de Datos con AWS Lambda y API Gateway

Para **consultar y verificar datos** usando **AWS Lambda y API Gateway**, se sigue un flujo típico de arquitectura sin servidor (serverless). Aquí tienes una guía práctica y paso a paso para hacerlo:

### 🧩 Arquitectura Básica

1. **API Gateway**: expone un endpoint HTTP (ej. `GET /usuario/{id}`).
2. **AWS Lambda**: función que procesa la solicitud y consulta datos.
3. **Amazon DynamoDB** (u otra fuente de datos): almacén que se consulta.

### ✅ Paso a Paso para Consulta de Datos

### 1. **Crear la tabla DynamoDB**

Supongamos una tabla con `UserId` como clave primaria.

```yaml
Resources:
  UsuariosTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: Usuarios
      AttributeDefinitions:
        - AttributeName: UserId
          AttributeType: S
      KeySchema:
        - AttributeName: UserId
          KeyType: HASH
      BillingMode: PAY_PER_REQUEST
```

### 2. **Función Lambda para consultar**

Código (Python 3.12):

```python
import json
import boto3
import os

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['TABLE_NAME'])

def lambda_handler(event, context):
    user_id = event['pathParameters']['id']
    
    try:
        response = table.get_item(Key={'UserId': user_id})
        item = response.get('Item')
        if not item:
            return {
                'statusCode': 404,
                'body': json.dumps({'message': 'Usuario no encontrado'})
            }
        return {
            'statusCode': 200,
            'body': json.dumps(item)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### 3. **Configurar variables de entorno en Lambda**

En el template de CloudFormation:

```yaml
Environment:
  Variables:
    TABLE_NAME: Usuarios
```

### 4. **Crear API Gateway (REST o HTTP)**

* Método: `GET`
* Ruta: `/usuario/{id}`
* Integración: Lambda Proxy

En CloudFormation (simplificado):

```yaml
Resources:
  ApiGateway:
    Type: AWS::ApiGatewayV2::Api
    Properties:
      Name: UsuarioAPI
      ProtocolType: HTTP

  ApiRoute:
    Type: AWS::ApiGatewayV2::Route
    Properties:
      ApiId: !Ref ApiGateway
      RouteKey: 'GET /usuario/{id}'
      Target: !Sub 'integrations/${ApiIntegration}'

  ApiIntegration:
    Type: AWS::ApiGatewayV2::Integration
    Properties:
      ApiId: !Ref ApiGateway
      IntegrationType: AWS_PROXY
      IntegrationUri: !Sub arn:aws:lambda:${AWS::Region}:${AWS::AccountId}:function:${LambdaFunctionName}
      PayloadFormatVersion: '2.0'
```

### 5. **Permitir que API Gateway invoque Lambda**

```yaml
LambdaInvokePermission:
  Type: AWS::Lambda::Permission
  Properties:
    Action: lambda:InvokeFunction
    FunctionName: !Ref LambdaFunction
    Principal: apigateway.amazonaws.com
```

### 🧪 Prueba Final

Una vez desplegado:

```bash
curl https://<tu-api-id>.execute-api.<region>.amazonaws.com/usuario/123
```

Respuesta esperada:

```json
{
  "UserId": "123",
  "Nombre": "Mario Vargas",
  "Correo": "mario@example.com"
}
```

### Resumen

#### ¿Cómo verificar el funcionamiento de recursos en AWS?

La gestión de recursos en AWS es crucial para maximizar el rendimiento y la eficiencia de los mismos. Empezaremos con comprobar que nuestros recursos en AWS estén funcionando correctamente. Aquí, dirigiremos nuestra atención al uso de AWS Lambda y DynamoDB.

#### ¿Cómo inicializar la tabla DynamoDB?

Para aprovechar al máximo DynamoDB, asegurémonos de ingresar y estructurar correctamente la información. Sigue estos pasos:

1. **Accede a DynamoDB**: Dentro de la consola de AWS, dirígete a DynamoDB.
2. **Selecciona la tabla**: Elige la tabla donde deseas ingresar los datos.
3. **Añadir ítems**:
- Navega a la sección de ítems y selecciona "Crear ítem".
- Inserta valores de tipo "String" para cada campo:
 - Nombre: "Pedro Pérez"
 - Número de cédula: 122,222
 - Dirección: "Avenida Todo grado 123"
 - Puesto de votación: "Puesto número 40"
 
4. **Guarda los cambios**: Asegúrate de que todos los datos queden guardados correctamente.

#### ¿Cómo validar los datos con AWS Lambda?

AWS Lambda es una herramienta esencial para ejecutar código en la nube sin aprovisionar servidores. Aquí te explicamos cómo crear y probar una función Lambda para validar los datos:

1. **Crea un nuevo test en Lambda**:
 - Abre tu función Lambda en la consola.
 - Dirígete a la opción "Test" en la parte superior derecha.
 - Borra el contenido sample y crea un nuevo objeto JSON:
 
```json
{
  "cedula": "222"
}
```

- Nombralo como "MiPrueba".

2. **Ejecuta el test**: Al correr la prueba, recibirás respuesta sobre los datos de Pedro Pérez si estos fueron ingresados correctamente.
3. **Valida los resultados**: Asegúrate de que toda la información como el nombre, número de cédula, dirección, etc., coincide con los datos ingresados.

#### ¿Cómo realizar la consulta por medio de API Gateway?

Realizar consultas mediante API Gateway posibilita interactuar con DynamoDB fácilmente:

1. **Configuración en API Gateway**:
- Abre la consola de API Gateway y selecciona el método POST.
- Crea la estructura del cuerpo para enviar el JSON:

```json
{
  "cedula": "222"
}   
```

2. **Envía y valida la respuesta**:
- Asegúrate de que el estado de respuesta sea 200, indicando que todo está en orden.

#### ¿Cómo integrar con herramientas externas como curl y Postman?

Para completar las pruebas, podemos utilizar herramientas externas como curl y Postman:

- **Uso de Curl:**

 - Ejecuta el siguiente comando:
 
`curl -X POST <API_URL> --data '{"cedula":"222"}'`

Este comando enviará la solicitud a la API y mostrará la respuesta recibida.

- **Uso de Postman**:

- Crea una nueva solicitud de tipo `POST`.
- Adjunta la URL de la API y manda el JSON en el cuerpo de la solicitud.
- La respuesta confirmará la correcta ejecución de la consulta.

La correcta integración y pruebas asegurarán que los recursos de AWS estén correctamente configurados y funcionando de manera óptima. Al dominar estas herramientas y procesos, puedes asegurar una gestión de datos eficiente en tu aplicación. ¡Sigue explorando y dominando AWS para llevar tus habilidades al siguiente nivel!

## Validación de Arquitecturas con AWS CloudFormation Designer

La **validación de arquitecturas con AWS CloudFormation Designer** es una práctica esencial para asegurar que tu infraestructura como código esté correctamente estructurada antes del despliegue. A continuación te explico cómo funciona y cómo puedes usarla efectivamente:

### ✅ ¿Qué es AWS CloudFormation Designer?

Es una herramienta visual integrada en la consola de AWS que te permite:

* **Diseñar arquitecturas** con componentes de AWS arrastrando y soltando.
* **Visualizar relaciones** entre recursos (como Lambda, S3, API Gateway, etc.).
* **Validar plantillas** YAML o JSON.
* **Editar código y diagrama** en tiempo real.

### 🧰 ¿Cómo Validar una Arquitectura en CloudFormation Designer?

### 🔹 1. Accede a CloudFormation Designer

1. Ve a la consola de AWS.
2. Navega a **CloudFormation**.
3. En el panel izquierdo, haz clic en **Designer**.

### 🔹 2. Cargar o crear una plantilla

Puedes:

* Subir una plantilla `.yaml` o `.json`.
* Escribir directamente en el editor.
* Arrastrar recursos desde el panel izquierdo.

### 🔹 3. Validar la plantilla

Una vez que hayas construido o cargado tu infraestructura:

✅ Haz clic en el botón **“Actions” → “Validate Template”**.

* Si es válida, verás un mensaje de éxito.
* Si tiene errores, te mostrará una lista detallada de problemas como:

  * Sintaxis YAML/JSON inválida.
  * Recursos mal referenciados.
  * Tipos de recursos inexistentes o con errores.

### 🛠️ Ejemplo de Error Común Detectado

Si tienes:

```yaml
Resources:
  MyBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucktName: "my-bucket"  # error: propiedad mal escrita
```

Designer te marcará:
**“Unrecognized property 'BucktName'”**, y te dirá la línea exacta.

### 📊 Ventajas de Usar CloudFormation Designer

| Ventaja                           | Descripción                                     |
| --------------------------------- | ----------------------------------------------- |
| **Visualización**                 | Ves gráficamente la arquitectura.               |
| **Detección temprana de errores** | Ahorra tiempo en pruebas.                       |
| **Edición bidireccional**         | Puedes editar tanto el código como el diagrama. |
| **Documentación automática**      | Puedes exportar la arquitectura como imagen.    |

### 🎯 Consejos Prácticos

* Usa **nombres descriptivos** para recursos (por ejemplo, `LambdaProcesaOrdenes` en vez de `Lambda1`).
* Agrupa parámetros y salidas con `Metadata -> AWS::CloudFormation::Interface`.
* Verifica las **referencias cruzadas** (`!Ref`, `!GetAtt`) estén bien conectadas.
* Utiliza **Stack anidados** para organizar arquitecturas grandes.

### Resumen

#### ¿Qué es AWS CloudFormation Designer y cómo nos beneficia?

AWS CloudFormation Designer es una herramienta fundamental para arquitectos y desarrolladores de software. Permite crear y visualizar arquitecturas y recursos en AWS de manera gráfica, facilitando la validación de la infraestructura antes de implementarla. La capacidad de mostrar gráficamente los recursos y sus conexiones es crucial para evitar errores costosos y facilitar el trabajo colaborativo. Utilizar CloudFormation Designer es altamente recomendando durante la fase de planificación de cualquier proyecto en la nube.

#### ¿Cómo cargar un stack en AWS CloudFormation Designer?

Para cargar un stack en CloudFormation Designer, necesitas un repositorio con el código de tu proyecto. A continuación, te comparto un proceso básico para llevar a buen término esta tarea:

1. **Clona el repositorio**: Utiliza Git para clonar el repositorio que contiene el código de tu aplicación.

`git clone <url_del_repositorio>`

2. **Accede a AWS Console**: Inicia sesión en tu cuenta de AWS y dirígete a CloudFormation.

3. **Selecciona Designer**: Busca la opción de Designer en el menú superior izquierdo y da click.

4. **Carga tu Stack**:

- Selecciona la opción para cargar un template.
- Navega a la ubicación de tu stack maestro en el repositorio clonado.
- Haz click en "Abrir".

5. **Visualiza tu Stack**: Al seleccionar ver en Designer, podrás observar gráficamente la estructura de tu stack. Las conexiones entre los diferentes componentes como Lambda, DynamoDB, y API Gateway se mostrarán para ayudarte a validar la arquitectura deseada.

#### ¿Cómo se diferencian los stack simples y los stack anidados?

La principal diferencia entre stack simples y anidados se encuentra en cómo se organizan y despliegan los recursos:

- **Stack Simple**: Todos los recursos están definidos en un solo template. Esto simplifica la visualización y es útil para proyectos pequeños o cuando deseas ver todas las conexiones en un solo lugar. Sin embargo, esta simplificación puede volverse compleja en proyectos más grandes.

- **Stack Anidado**: Estos utilizan múltiples templates menores que representan diferentes partes de la aplicación. Cada sub-stack es una porción de la aplicación y solo se cargan las relaciones entre ellos al visualizar en Designer. Esto division permite un mayor control y organización.

#### ¿Cuándo usar AWS CloudFormation Designer en proyectos reales?

CloudFormation Designer es ideal para la validación de arquitecturas antes del despliegue. Las visualizaciones gráficas proporcionan una confirmación visual que puede prevenir errores y optimizar configuraciones. Sin embargo, para la creación de templates, es preferible escribir el código directamente. Esto mejora la comprensión del código subyacente y ofrece un mayor control sobre los detalles implementados.

Motiva a los desarrolladores a integrarlo en su flujo de trabajo regular, sobre todo al inicio de un nuevo proyecto, para alinear expectativas con el diseño deseado. ¡Continúa explorando y optimizando tus proyectos con Designer para alcanzar nuevos niveles de éxito en AWS!