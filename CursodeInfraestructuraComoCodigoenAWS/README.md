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