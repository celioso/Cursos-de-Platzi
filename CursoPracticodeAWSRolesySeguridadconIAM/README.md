# Curso Práctico de AWS Roles y Seguridad con IAM

## ¿Ya tomaste los cursos introductorios de AWS?

**Archivos de la clase**

[2-slides-aws-iam.pdf](https://static.platzi.com/media/public/uploads/2-slides_aws_iam_3348ea9a-7179-4802-b06f-772fe03486ba.pdf)

**Lecturas recomendadas**

[Curso de Introducción a AWS: Fundamentos de Cloud Computing - Platzi](https://platzi.com/cursos/aws-fundamentos/)

[Curso de Introducción a AWS: Cómputo, Almacenamiento y Bases de Datos - Platzi](https://platzi.com/cursos/aws-computo/)

## Introducción IAM: usuarios, grupos y políticas

IAM (**Identity and Access Management**) es el servicio de AWS que permite gestionar el acceso a los recursos de AWS de manera segura. Con IAM, se pueden crear y administrar **usuarios, grupos y políticas**, asegurando que solo las personas y servicios autorizados puedan acceder a los recursos adecuados.  


### 🔹 **Conceptos Claves de IAM**  

### 🧑‍💻 **Usuarios IAM**  
Son entidades individuales que representan una persona o una aplicación que necesita interactuar con AWS.  
✅ Cada usuario tiene credenciales únicas (contraseña y/o claves de acceso).  
✅ Puede tener permisos asignados directamente o a través de grupos.  
✅ Puede autenticarse con la **Consola de AWS** o usando la **CLI/SDK**.  

### 👥 **Grupos IAM**  
Son colecciones de usuarios que comparten los mismos permisos.  
✅ Facilitan la gestión de permisos en grupos de usuarios.  
✅ Un usuario puede pertenecer a varios grupos.  
✅ Ejemplo: Grupo **"Admins"** con permisos de administración y grupo **"Desarrolladores"** con acceso a servicios específicos.  

### 📜 **Políticas IAM**  
Son documentos en formato **JSON** que definen permisos.  
✅ Especifican **qué acciones** se pueden realizar en **qué recursos** y bajo **qué condiciones**.  
✅ Se pueden asignar a **usuarios, grupos o roles**.  
✅ Ejemplo de política que permite listar los buckets de S3:  
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "s3:ListAllMyBuckets",
            "Resource": "*"
        }
    ]
}
```  

## 🔒 **Buenas Prácticas en IAM**  
✔️ **Principio de menor privilegio**: Otorgar solo los permisos necesarios.  
✔️ **Usar roles en vez de claves de acceso**: Para servicios de AWS que interactúan entre sí.  
✔️ **Activar MFA (Multi-Factor Authentication)**: Mayor seguridad para accesos críticos.  
✔️ **No usar el usuario root** para tareas diarias: Crear usuarios con permisos específicos.  
✔️ **Revisar y auditar permisos regularmente**.

💡 **Resumen:**  
IAM permite gestionar la seguridad y acceso a los servicios de AWS mediante **usuarios, grupos y políticas**, asegurando un control granular sobre los permisos. 🚀

**Introducción a IAM**
- **Concepto de IAM**: AWS Identity and Access Management (IAM) permite administrar de manera segura el acceso a los servicios y recursos de AWS. Con IAM, puedes crear y gestionar usuarios y grupos, y utilizar permisos para permitir o denegar su acceso a los recursos de AWS.
**Usuarios y Grupos**
- **Usuarios IAM**: Representan a una persona o aplicación que interactúa con los servicios de AWS. Cada usuario tiene credenciales únicas para acceder a los recursos.
- **Grupos IAM**: Son colecciones de usuarios IAM. Puedes asignar permisos a un grupo, lo que simplifica la gestión de permisos cuando tienes múltiples usuarios con los mismos requisitos de acceso.
**Políticas IAM**
- **Políticas administradas**: Son políticas creadas y gestionadas por AWS. Puedes adjuntarlas a usuarios, grupos y roles para otorgar permisos.
- **Políticas personalizadas**: Son políticas que creas para satisfacer necesidades específicas de tu organización. Utilizan JSON para definir los permisos.
- **Política de permisos mínimos**: Es una práctica recomendada que implica otorgar solo los permisos necesarios para realizar tareas específicas, minimizando el riesgo de acceso no autorizado.
**Roles IAM**
- **Roles IAM**: Permiten delegar permisos a entidades de confianza sin necesidad de compartir credenciales. Los roles se utilizan ampliamente para dar acceso a servicios dentro de AWS o para permitir que aplicaciones y servicios asuman ciertos permisos.
- **Asunción de roles:** Un usuario o servicio puede asumir un rol para obtener permisos temporales necesarios para realizar una tarea específica.

**Lecturas recomendadas**

[Cloud Computing Services - Amazon Web Services (AWS)](https://aws.amazon.com/)

## Práctica IAM Usuarios y Grupos

Esta práctica te guiará paso a paso en la creación de usuarios y grupos en **AWS IAM**, asignando permisos y configurando accesos de manera segura.

### 🔹 **Paso 1: Acceder a AWS IAM**  
1. Inicia sesión en la [Consola de AWS](https://aws.amazon.com/console/).  
2. En la barra de búsqueda, escribe **IAM** y selecciona el servicio **IAM**.

### 👥 **Paso 2: Crear un Grupo en IAM**  
1. En el menú lateral, selecciona **Grupos de Usuarios** → Clic en **Crear grupo**.  
2. Ingresa un **nombre para el grupo** (Ejemplo: *Desarrolladores* o *Admins*).  
3. En la sección **Permisos**, elige una política de permisos:  
   - Para administradores: **AdministratorAccess**  
   - Para desarrolladores: **AmazonEC2FullAccess**, **AmazonS3ReadOnlyAccess**, etc.  
4. Clic en **Crear grupo**.

### 🧑‍💻 **Paso 3: Crear un Usuario IAM**  
1. En el menú lateral, selecciona **Usuarios** → Clic en **Agregar usuario**.  
2. Ingresa un **nombre de usuario**.  
3. Selecciona **Tipo de credenciales**:  
   - **Acceso a la consola de AWS** (para gestionar desde la web).  
   - **Acceso mediante clave de acceso** (para programadores con AWS CLI o SDK).  
4. Clic en **Siguiente: Permisos**.

### 🔑 **Paso 4: Asignar Permisos al Usuario**  
1. **Agregar usuario a un grupo existente** (Ejemplo: *Desarrolladores*).  
2. **Asignar permisos directamente** (opcional).  
3. Clic en **Siguiente: Etiquetas** (Opcional, puedes agregar etiquetas para organización).  
4. Clic en **Siguiente: Revisar** y luego en **Crear usuario**.  
5. **Descargar las credenciales de acceso** (importante si creaste claves de acceso).

### 🔒 **Paso 5: Buenas Prácticas de Seguridad**  
✔️ **Usar Multi-Factor Authentication (MFA)** para mayor seguridad.  
✔️ **No usar el usuario root para tareas diarias**.  
✔️ **Aplicar el principio de menor privilegio** (solo los permisos necesarios).  
✔️ **Rotar las claves de acceso periódicamente**.  

✅ **¡Listo! Has creado y gestionado usuarios y grupos en AWS IAM con seguridad y control!** 🚀

## Politicas IAM

Las **políticas IAM** son reglas que definen permisos para los usuarios, grupos y roles en AWS. Permiten controlar quién puede hacer qué en los servicios y recursos de AWS.  

## 📌 **Tipos de Políticas IAM**  
1. **Administradas por AWS**: Políticas predefinidas listas para usar (Ej: `AdministratorAccess`, `AmazonS3ReadOnlyAccess`).  
2. **Administradas por el Cliente**: Políticas personalizadas creadas por el usuario.  
3. **Políticas en Línea**: Específicas para un solo usuario, grupo o rol. 

### 📜 **Estructura de una Política IAM (JSON)**  
Una política en IAM sigue un formato JSON con los siguientes elementos clave:  

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::mi-bucket"
    }
  ]
}
```

### 🛠 **Explicación de los elementos**  
- **`Version`**: Define la versión de la política (debe ser `"2012-10-17"` para compatibilidad).  
- **`Statement`**: Lista de reglas en la política.  
- **`Effect`**: `"Allow"` (permitir) o `"Deny"` (denegar).  
- **`Action`**: Acción permitida o denegada (Ejemplo: `"s3:ListBucket"` permite listar objetos en un bucket S3).  
- **`Resource`**: Especifica a qué recurso se aplica la política (Ejemplo: `arn:aws:s3:::mi-bucket`).

### 🎯 **Ejemplo de Política con Múltiples Acciones**  
Permite a los usuarios leer y escribir en un bucket S3 específico:  

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::mi-bucket/*"
    }
  ]
}
```

### 🚫 **Ejemplo de Política de Denegación**  
Deniega la eliminación de objetos en un bucket S3:  

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": "s3:DeleteObject",
      "Resource": "arn:aws:s3:::mi-bucket/*"
    }
  ]
}
```

### 🔄 **Cómo Adjuntar una Política a un Usuario o Grupo**  
1. **Desde la Consola de AWS**:  
   - Ir a **IAM > Usuarios / Grupos / Roles**.  
   - Seleccionar el usuario o grupo.  
   - Ir a la pestaña **Permisos** y hacer clic en **Adjuntar políticas**.  
   - Buscar y seleccionar la política deseada.  

2. **Desde AWS CLI** (Ejemplo: adjuntar `AmazonS3ReadOnlyAccess` a un usuario):  

```sh
aws iam attach-user-policy --user-name MiUsuario --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

### ✅ **Buenas Prácticas en IAM**  
✔ **Aplicar el principio de menor privilegio** (dar solo los permisos necesarios).  
✔ **Usar roles en lugar de usuarios con credenciales permanentes**.  
✔ **Habilitar MFA (Autenticación Multifactor)** para mayor seguridad.  
✔ **Revisar y auditar permisos regularmente** con **IAM Access Analyzer**.

🚀 **¡Ahora tienes el control sobre las políticas IAM en AWS!** 🔐

**Lecturas recomendadas**

[AWS Policy Generator](https://awspolicygen.s3.amazonaws.com/policygen.html)

[https://policysim.aws.amazon.com/home/index.jsp](https://policysim.aws.amazon.com/home/index.jsp)

## Prácticas politicas IAM

Aquí tienes algunas prácticas recomendadas y ejercicios para trabajar con **políticas IAM** en AWS.

### 🏋️ **Prácticas con Políticas IAM en AWS**  

### 📌 **1. Crear una Política Personalizada en IAM**  
**Objetivo**: Crear una política que permita a un usuario ver pero no modificar los recursos en Amazon S3.  

### **Pasos**:  
1. Ir a la consola de **AWS IAM**.  
2. En el menú lateral, seleccionar **Políticas** → **Crear política**.  
3. Seleccionar **JSON** y agregar la siguiente política:  

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetObject"
      ],
      "Resource": [
        "arn:aws:s3:::mi-bucket",
        "arn:aws:s3:::mi-bucket/*"
      ]
    }
  ]
}
```
4. Hacer clic en **Siguiente** y asignar un nombre, por ejemplo: `"S3ReadOnlyPolicy"`.  
5. Guardar la política y adjuntarla a un usuario o grupo en IAM.

### 📌 **2. Crear una Política de Acceso Restringido a una Región**  
**Objetivo**: Permitir que un usuario solo cree instancias EC2 en la región **us-east-1**.  

### **Pasos**:  
1. En IAM, ir a **Políticas** → **Crear política**.  
2. En **JSON**, agregar:  

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "ec2:RunInstances",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "us-east-1"
        }
      }
    }
  ]
}
```
3. Guardar la política y adjuntarla a un usuario de prueba.  
4. Intentar lanzar una instancia en otra región para verificar la restricción.

### 📌 **3. Crear una Política de Acceso Basado en Horarios**  
**Objetivo**: Permitir que un usuario acceda a la consola de AWS solo en horarios laborales (Ejemplo: de 8 AM a 6 PM UTC).  

### **Pasos**:  
1. Crear una nueva política en **IAM**.  
2. En **JSON**, agregar la siguiente regla:  

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "NumericGreaterThan": { "aws:CurrentTime": "18:00:00" },
        "NumericLessThan": { "aws:CurrentTime": "08:00:00" }
      }
    }
  ]
}
```
3. Adjuntar esta política a un usuario y probar acceder fuera del horario permitido.

### 📌 **4. Crear una Política para Bloquear la Eliminación de Recursos Críticos**  
**Objetivo**: Evitar que los usuarios eliminen instancias EC2, pero permitirles iniciarlas y detenerlas.  

### **Pasos**:  
1. Crear una política con la siguiente configuración en **JSON**:  

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": [
        "ec2:TerminateInstances"
      ],
      "Resource": "*"
    }
  ]
}
```
2. Asignar la política a un usuario y verificar que **no pueda eliminar** una instancia EC2.

### ✅ **Buenas Prácticas al Trabajar con IAM**  
✔ **Aplicar el principio de menor privilegio**: Asignar solo los permisos necesarios.  
✔ **Usar roles en lugar de usuarios con claves de acceso permanentes**.  
✔ **Habilitar MFA (Autenticación Multifactor) para usuarios críticos**.  
✔ **Revisar permisos regularmente con AWS IAM Access Analyzer**.  
✔ **Monitorear con AWS CloudTrail para detectar accesos sospechosos**.

🚀 **¡Ahora puedes poner en práctica el manejo de políticas IAM en AWS!** 🔐

## Visión general IAM MFA

AWS **Identity and Access Management (IAM)** es un servicio que te permite gestionar el acceso a los recursos de AWS de manera segura. Con IAM, puedes crear y administrar usuarios, grupos, roles y políticas para controlar quién puede acceder a qué.

### ✅ **¿Qué es MFA en AWS IAM?**  
La **autenticación multifactor (MFA, Multi-Factor Authentication)** agrega una **capa adicional de seguridad** al exigir una segunda forma de autenticación además de la contraseña. Esto reduce el riesgo de acceso no autorizado a cuentas de AWS.  

💡 **Ejemplo:** Un atacante que roba tu contraseña no podrá acceder sin el segundo factor de autenticación.  

### 🏗 **Cómo Funciona MFA en AWS**  
Cuando un usuario intenta iniciar sesión:  
1️⃣ Ingresa su nombre de usuario y contraseña.  
2️⃣ AWS solicita un código de autenticación generado por un dispositivo MFA.  
3️⃣ Si el código es correcto, el acceso es concedido.

### 🔹 **Tipos de MFA en AWS**  

AWS soporta diferentes métodos de MFA:  

| Tipo de MFA | Descripción | Ejemplo de Dispositivo |
|------------|-------------|----------------|
| **Dispositivo virtual MFA** | Usa aplicaciones como **Google Authenticator** o **Authy** para generar códigos de 6 dígitos. | 📱 Móvil o Tablet |
| **Dispositivo MFA basado en hardware** | Un dispositivo físico que genera códigos de acceso. | 🔑 YubiKey |
| **MFA con clave de seguridad FIDO2** | Usa claves de hardware como **YubiKey** o **Titan Security Key**. | 🖥 USB o NFC |
| **MFA con notificación push** *(Recomendado para IAM Identity Center)* | Permite aprobar solicitudes en la aplicación de AWS. | 📲 AWS Authenticator |

### 🔹 **Cómo Configurar MFA para un Usuario IAM**  

1️⃣ Inicia sesión en la **consola de AWS** con privilegios de administrador.  
2️⃣ Ve a **IAM** → **Usuarios** → Selecciona el usuario.  
3️⃣ En la pestaña **Seguridad**, haz clic en **Asignar MFA**.  
4️⃣ Elige un tipo de MFA (virtual, hardware, etc.).  
5️⃣ **Si usas un dispositivo virtual (App Authenticator)**:  
   - Escanea el código QR con una aplicación de autenticación.  
   - Ingresa dos códigos consecutivos generados por la app.  
6️⃣ Guarda los cambios.  

🛑 **Importante**: Asegúrate de guardar códigos de recuperación en caso de perder el dispositivo MFA.

### 🔹 **Forzar el Uso de MFA en IAM con una Política**  

Para exigir MFA a los usuarios IAM, puedes crear una política de IAM:  

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "BoolIfExists": {
          "aws:MultiFactorAuthPresent": "false"
        }
      }
    }
  ]
}
```
📌 **Qué hace esta política**:  
- **Deniega acceso** a todos los recursos si el usuario **NO tiene MFA habilitado**.  
- **Se aplica automáticamente** cuando el usuario inicia sesión sin MFA.  

🔹 **Cómo aplicarla:**  
1️⃣ En IAM, ve a **Políticas** → **Crear política**.  
2️⃣ Usa la pestaña **JSON** e ingresa el código anterior.  
3️⃣ Asigna la política a un grupo o usuario.  

### 🎯 **Beneficios de Habilitar MFA en AWS**  

✔ **Mayor seguridad**: Evita accesos no autorizados incluso si la contraseña es comprometida.  
✔ **Cumplimiento de normativas**: Requerido en auditorías de seguridad y estándares como PCI-DSS.  
✔ **Protección contra ataques**: Reduce el riesgo de **phishing** y **fuerza bruta**.  

🔐 **Conclusión**  
Habilitar MFA en IAM es una de las mejores prácticas de seguridad en AWS. Reforzar el acceso con autenticación de dos factores ayuda a proteger los recursos críticos y reducir el riesgo de accesos no autorizados. 🚀

Lecturas recomendadas**

[AWS CloudHSM](https://aws.amazon.com/es/cloudhsm/)

## Configuración IAM MFA

Habilitar la autenticación multifactor (MFA) en AWS Identity and Access Management (IAM) aumenta la seguridad al requerir un segundo factor de autenticación al iniciar sesión.

### ✅ **Pasos para Configurar MFA en un Usuario IAM**  

### 1️⃣ **Acceder a la Consola de AWS**  
🔹 Inicia sesión en la consola de administración de AWS con una cuenta que tenga permisos de administrador.  

### 2️⃣ **Ir a IAM (Identity and Access Management)**  
🔹 En el menú de AWS, busca **IAM** y selecciona el servicio.  

### 3️⃣ **Seleccionar el Usuario IAM**  
🔹 En la barra lateral izquierda, haz clic en **Usuarios**.  
🔹 Selecciona el usuario IAM al que deseas habilitar MFA.  

### 4️⃣ **Configurar MFA**  
🔹 Dentro del perfil del usuario, ve a la pestaña **Seguridad**.  
🔹 En la sección de **Dispositivos de autenticación multifactor**, haz clic en **Asignar MFA**.  
🔹 Elige el tipo de MFA a configurar:  

| Tipo de MFA | Descripción | Dispositivos Soportados |
|------------|-------------|----------------|
| **Dispositivo virtual MFA** | Usa una app para generar códigos de 6 dígitos. | 📱 Google Authenticator, Authy |
| **Dispositivo MFA basado en hardware** | Genera códigos en un dispositivo físico. | 🔑 YubiKey, Token MFA |
| **MFA con clave de seguridad FIDO2** | Utiliza una llave de seguridad para autenticación. | 🖥 USB/NFC (Ej: Titan Security Key) |

### 5️⃣ **Configurar MFA con un Dispositivo Virtual (Google Authenticator, Authy, etc.)**  
1. Selecciona **Dispositivo virtual MFA** y haz clic en **Siguiente**.  
2. **Escanea el código QR** con una aplicación de autenticación (Google Authenticator, Authy, Microsoft Authenticator).  
3. La app generará un código de 6 dígitos.  
4. **Ingresa dos códigos consecutivos** para verificar la configuración.  
5. Haz clic en **Asignar MFA** y confirma.  

### 6️⃣ **Finalizar y Probar el Inicio de Sesión con MFA**  
🔹 Cierra la sesión y vuelve a iniciar.  
🔹 Ingresa tu usuario y contraseña de AWS.  
🔹 Se te pedirá un **código MFA** generado por la aplicación.  
🔹 Una vez ingresado correctamente, accederás a la consola.

### 🎯 **Recomendaciones de Seguridad**  

✔ **Obliga el uso de MFA** para todos los usuarios con permisos administrativos mediante una política de IAM.  
✔ **Configura múltiples dispositivos MFA** en caso de pérdida o robo del principal.  
✔ **Usa claves de seguridad FIDO2** para mayor protección contra ataques de phishing.  

🔐 **Conclusión**  
Habilitar MFA en IAM es una práctica esencial para reforzar la seguridad en AWS, asegurando que solo usuarios autorizados accedan a la cuenta. 🚀

**Lecturas recomendadas**

[IAM - Multi-Factor Authentication](https://aws.amazon.com/iam/features/mfa/)

## AWS Access Keys, CLI y SDK

Las **AWS Access Keys**, el **AWS Command Line Interface (CLI)** y los **Software Development Kits (SDKs)** permiten interactuar con los servicios de AWS de forma segura y automatizada.

### 🔹 **1. AWS Access Keys (Claves de Acceso)**  
Las Access Keys son credenciales que permiten la autenticación de usuarios y aplicaciones para interactuar con AWS mediante la CLI, SDKs o llamadas a la API.  

### 📌 **Tipos de Credenciales en AWS**  
| Tipo de Credencial | Descripción | Uso Principal |
|----------------|----------------|----------------|
| **Clave de acceso (Access Key ID + Secret Access Key)** | Claves de autenticación para la API, CLI o SDKs. | Scripts, automatización y acceso programático. |
| **Credenciales temporales (STS - Security Token Service)** | Claves temporales generadas por IAM Roles o AWS STS. | Acceso seguro sin claves permanentes. |
| **Perfiles de IAM** | Roles asociados a instancias de EC2. | Acceso automático sin claves explícitas. |

### ✅ **Cómo Generar AWS Access Keys**  
1. **Iniciar sesión** en la consola de AWS.  
2. Ir a **IAM > Usuarios > Seleccionar un usuario**.  
3. En la pestaña **Credenciales de seguridad**, buscar la sección **Claves de acceso**.  
4. Hacer clic en **Crear clave de acceso** y guardar la **Access Key ID** y la **Secret Access Key**.  
   🔹 *¡No compartas estas claves! Son confidenciales.*  

### 🔥 **Buenas Prácticas con Access Keys**  
✔ **Evitar almacenar claves en código fuente** (usar variables de entorno o AWS Secrets Manager).  
✔ **Usar IAM Roles en lugar de claves estáticas** siempre que sea posible.  
✔ **Rotar las claves periódicamente** y revocar las que no se usen.

### ⚡ **2. AWS CLI (Command Line Interface)**  
AWS CLI es una herramienta para interactuar con AWS desde la línea de comandos.  

### ✅ **Instalación de AWS CLI**  
🔹 En **Linux/macOS**:  
```sh
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
```
🔹 En **Windows**: Descargar desde [AWS CLI Installer](https://aws.amazon.com/cli/)  

### ⚙ **Configurar AWS CLI con Access Keys**  
```sh
aws configure
```
🔹 Se solicitarán:  
- **Access Key ID**  
- **Secret Access Key**  
- **Región por defecto** (ej: `us-east-1`)  
- **Formato de salida** (`json`, `table`, `text`)  

### 📌 **Ejemplo de Uso en CLI**  
🔹 Listar los buckets de S3:  
```sh
aws s3 ls
```
🔹 Ver las instancias EC2 en ejecución:  
```sh
aws ec2 describe-instances --filters Name=instance-state-name,Values=running
```

### 💻 **3. AWS SDKs (Software Development Kits)**  
Los SDKs permiten interactuar con AWS en diferentes lenguajes de programación como Python, JavaScript, Java, Go, etc.  

### ✅ **SDKs más usados**  
| Lenguaje | SDK |
|----------|-----------|
| Python | `boto3` |
| JavaScript | `AWS SDK for JavaScript` |
| Java | `AWS SDK for Java` |
| Go | `AWS SDK for Go` |

### 📌 **Ejemplo con Python (`boto3`)**  
🔹 **Instalar el SDK**  
```sh
pip install boto3
```
🔹 **Ejemplo: Listar los buckets de S3**  
```python
import boto3

s3 = boto3.client('s3')
buckets = s3.list_buckets()

for bucket in buckets['Buckets']:
    print(bucket['Name'])
```

### 🚀 **Conclusión**  
🔹 **AWS Access Keys** permiten autenticarse en AWS.  
🔹 **AWS CLI** facilita la administración desde la terminal.  
🔹 **AWS SDKs** permiten la automatización en código.  

Usar IAM Roles y credenciales temporales es la mejor práctica para evitar el uso de Access Keys estáticas. 💡

## Setup AWS CLI en Mac

AWS Command Line Interface (CLI) permite gestionar recursos de AWS desde la terminal. A continuación, te explico cómo instalar y configurar AWS CLI en macOS.

### ✅ **Paso 1: Descargar e Instalar AWS CLI en Mac**  

### 🔹 **Método 1: Usando Homebrew (Recomendado)**
Si tienes **Homebrew** instalado, puedes instalar AWS CLI fácilmente:  
```sh
brew install awscli
```
Para verificar que la instalación fue exitosa:  
```sh
aws --version
```
Debería mostrar algo como:  
```
aws-cli/2.x.x Python/3.x.x Darwin/x86_64
```

### 🔹 **Método 2: Instalación Manual**
1. Descarga el paquete desde [AWS CLI para macOS](https://awscli.amazonaws.com/AWSCLIV2.pkg).  
2. Abre el archivo descargado (`AWSCLIV2.pkg`) y sigue las instrucciones de instalación.  
3. Verifica la instalación con:  
   ```sh
   aws --version
   ```

### ✅ **Paso 2: Configurar AWS CLI**  
Para autenticarte en AWS, necesitas Access Keys.  

🔹 **Ejecuta:**  
```sh
aws configure
```
🔹 **Ingresa:**  
1. **AWS Access Key ID:** (Clave de acceso de IAM)  
2. **AWS Secret Access Key:** (Clave secreta de IAM)  
3. **Región por defecto:** (ej. `us-east-1`, `us-west-2`, `sa-east-1`)  
4. **Formato de salida:** (`json`, `table`, `text`)  

📌 **Ejemplo de entrada:**  
```
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-east-1
Default output format [None]: json
```

### ✅ **Paso 3: Verificar la Configuración**  
🔹 Para probar la conexión con AWS, ejecuta:  
```sh
aws s3 ls
```
Si tienes acceso a S3, verás una lista de buckets. Si no, revisa las credenciales o permisos en IAM.

### ✅ **Paso 4: Uso Básico de AWS CLI**  
Algunos comandos útiles para empezar:  

🔹 **Listar instancias EC2 activas:**  
```sh
aws ec2 describe-instances --filters Name=instance-state-name,Values=running
```

🔹 **Subir un archivo a S3:**  
```sh
aws s3 cp archivo.txt s3://mi-bucket/
```

🔹 **Ver logs en CloudWatch:**  
```sh
aws logs describe-log-groups
```

### 🎯 **Conclusión**  
✅ **AWS CLI en macOS** es fácil de instalar y configurar.  
✅ **Homebrew es el método más rápido y recomendado.**  
✅ **Usar `aws configure` permite establecer credenciales de acceso.**  
✅ **Probar con `aws s3 ls` ayuda a verificar la conexión.**  

Ahora puedes administrar tus recursos de AWS desde la terminal. 🚀

**Lecturas recomendadas**

[Installing or updating the latest version of the AWS CLI - AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

## Setup AWS CLI en Windows

AWS Command Line Interface (CLI) permite administrar servicios de AWS desde la terminal. A continuación, te explico cómo instalar y configurar AWS CLI en **Windows**.

### ✅ **Paso 1: Descargar e Instalar AWS CLI en Windows**  

🔹 **Descargar el instalador**  
1. Ve a la página oficial de AWS CLI:  
   👉 [Descargar AWS CLI para Windows](https://awscli.amazonaws.com/AWSCLIV2.msi)  
2. Ejecuta el archivo `.msi` y sigue las instrucciones del asistente de instalación.  
3. Una vez finalizada la instalación, abre **Símbolo del sistema (CMD)** o **PowerShell** y verifica la instalación con:  
   ```sh
   aws --version
   ```
   Debería mostrar algo como:  
   ```
   aws-cli/2.x.x Python/3.x.x Windows/10
   ```

### ✅ **Paso 2: Configurar AWS CLI**  
Para autenticarte en AWS, necesitas **Access Keys** de una cuenta IAM.  

🔹 **Ejecuta el siguiente comando en CMD o PowerShell:**  
```sh
aws configure
```

🔹 **Ingresa los siguientes datos:**  
1. **AWS Access Key ID:** (Clave de acceso de IAM)  
2. **AWS Secret Access Key:** (Clave secreta de IAM)  
3. **Región por defecto:** (Ejemplo: `us-east-1`, `sa-east-1`)  
4. **Formato de salida:** (`json`, `table`, `text`)  

📌 **Ejemplo de entrada:**  
```
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-east-1
Default output format [None]: json
```

### ✅ **Paso 3: Verificar la Configuración**  
🔹 Para probar la conexión con AWS, ejecuta:  
```sh
aws s3 ls
```
Si tienes acceso a S3, verás una lista de buckets. Si no, revisa las credenciales o permisos en IAM.

### ✅ **Paso 4: Uso Básico de AWS CLI**  
Algunos comandos útiles para empezar:  

🔹 **Listar instancias EC2 activas:**  
```sh
aws ec2 describe-instances --filters Name=instance-state-name,Values=running
```

🔹 **Subir un archivo a S3:**  
```sh
aws s3 cp archivo.txt s3://mi-bucket/
```

🔹 **Ver logs en CloudWatch:**  
```sh
aws logs describe-log-groups
```

### 🎯 **Conclusión**  
✅ **AWS CLI en Windows** es fácil de instalar y configurar.  
✅ **Usar `aws configure` permite establecer credenciales de acceso.**  
✅ **Probar con `aws s3 ls` ayuda a verificar la conexión.**  

¡Ahora puedes administrar tus recursos de AWS desde Windows! 🚀

**Lecturas recomendadas**

[Amazon Web Services Documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.htmll)

## Setup AWS CLI en Linux

AWS Command Line Interface (CLI) te permite administrar los servicios de AWS desde la terminal. A continuación, te explico cómo instalar y configurar AWS CLI en **Linux**.

### ✅ **Paso 1: Descargar e Instalar AWS CLI en Linux**  

### 🔹 **1. Descargar AWS CLI**  
Abre una terminal y ejecuta:  
```sh
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
```

### 🔹 **2. Extraer el archivo**  
```sh
unzip awscliv2.zip
```
Si no tienes `unzip`, instálalo con:  
```sh
sudo apt install unzip  # Debian/Ubuntu  
sudo yum install unzip  # CentOS/RedHat  
```

### 🔹 **3. Instalar AWS CLI**  
```sh
sudo ./aws/install
```

### 🔹 **4. Verificar la instalación**  
```sh
aws --version
```
Debería mostrar algo como:  
```
aws-cli/2.x.x Python/3.x.x Linux/x86_64
```

### ✅ **Paso 2: Configurar AWS CLI**  
Para autenticarte en AWS, necesitas **Access Keys** de una cuenta IAM.  

🔹 **Ejecuta el siguiente comando en la terminal:**  
```sh
aws configure
```

🔹 **Ingresa los siguientes datos:**  
1. **AWS Access Key ID:** (Clave de acceso de IAM)  
2. **AWS Secret Access Key:** (Clave secreta de IAM)  
3. **Región por defecto:** (Ejemplo: `us-east-1`, `sa-east-1`)  
4. **Formato de salida:** (`json`, `table`, `text`)  

📌 **Ejemplo de entrada:**  
```
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-east-1
Default output format [None]: json
```

### ✅ **Paso 3: Verificar la Configuración**  
🔹 Para probar la conexión con AWS, ejecuta:  
```sh
aws s3 ls
```
Si tienes acceso a S3, verás una lista de buckets. Si no, revisa las credenciales o permisos en IAM.

### ✅ **Paso 4: Uso Básico de AWS CLI**  
Algunos comandos útiles para empezar:  

🔹 **Listar instancias EC2 activas:**  
```sh
aws ec2 describe-instances --filters Name=instance-state-name,Values=running
```

🔹 **Subir un archivo a S3:**  
```sh
aws s3 cp archivo.txt s3://mi-bucket/
```

🔹 **Ver logs en CloudWatch:**  
```sh
aws logs describe-log-groups
```

### 🎯 **Conclusión**  
✅ **AWS CLI en Linux** es fácil de instalar y configurar.  
✅ **Usar `aws configure` permite establecer credenciales de acceso.**  
✅ **Probar con `aws s3 ls` ayuda a verificar la conexión.**  

¡Ahora puedes administrar tus recursos de AWS desde Linux! 🚀

**Descargar** 

```CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
```

**Instalar Unzip**

```CLI
sudo apt install unzip
```
**Descomprimir**

```CLI
unzip awscliv2.zip
```

**Instalar**

```CLI
sudo ./aws/install
```

**Lecturas recomendadas**

[Installing or updating the latest version of the AWS CLI - AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

## Configuración AWS CLI con Access Keys

AWS CLI permite gestionar los servicios de AWS desde la terminal. Para autenticarte, puedes usar **Access Keys**, que son credenciales de una cuenta IAM.

### ✅ **Paso 1: Obtener las Access Keys**  
Antes de configurar AWS CLI, necesitas generar una **Access Key ID** y **Secret Access Key** en AWS IAM.  

### 🔹 **Cómo generar Access Keys en AWS IAM**  
1. **Inicia sesión en AWS Console** ([https://aws.amazon.com/](https://aws.amazon.com/)).  
2. **Ve a "IAM" (Identity and Access Management).**  
3. En el menú lateral, selecciona **"Usuarios"**.  
4. Haz clic en el usuario para el cual necesitas las credenciales.  
5. Ve a la pestaña **"Credenciales de seguridad"**.  
6. En la sección **"Claves de acceso"**, haz clic en **"Crear clave de acceso"**.  
7. Copia y guarda **Access Key ID** y **Secret Access Key** (solo se muestran una vez).

### ✅ **Paso 2: Configurar AWS CLI con Access Keys**  

🔹 **Abre una terminal y ejecuta:**  
```sh
aws configure
```

🔹 **Ingresa los siguientes datos cuando se soliciten:**  
1. **AWS Access Key ID:** *(Clave de acceso obtenida en IAM)*  
2. **AWS Secret Access Key:** *(Clave secreta obtenida en IAM)*  
3. **Región por defecto:** *(Ejemplo: `us-east-1`, `sa-east-1`)*  
4. **Formato de salida:** *(Opcional: `json`, `table`, `text`)*  

📌 **Ejemplo de entrada:**  
```
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-east-1
Default output format [None]: json
```

### ✅ **Paso 3: Verificar la Configuración**  

🔹 **Prueba que AWS CLI funciona correctamente:**  
```sh
aws s3 ls
```
Si tienes acceso a S3, verás una lista de buckets. Si no, revisa las credenciales o permisos en IAM.

### ✅ **Paso 4: Configurar un Perfil Adicional (Opcional)**  

Si trabajas con múltiples cuentas de AWS, puedes configurar perfiles adicionales:  
```sh
aws configure --profile nombre_del_perfil
```
Luego, para usar este perfil en los comandos:  
```sh
aws s3 ls --profile nombre_del_perfil
```

Lista de usuarios

```sh
aws iam list-users
```

### 🎯 **Conclusión**  
✅ **AWS CLI se configura con `aws configure`** usando Access Keys.  
✅ **Es importante guardar las Access Keys en un lugar seguro.**  
✅ **Puedes verificar la configuración con `aws s3 ls`.**  

¡Ahora puedes administrar AWS desde la terminal con tus Access Keys! 🚀

**Lecturas recomendadas**

[list-users — AWS CLI 1.23.7 Command Reference](https://docs.aws.amazon.com/cli/latest/reference/iam/list-users.html)

## AWS CloudShell

AWS CloudShell es una terminal en línea completamente administrada por AWS que permite ejecutar comandos de AWS CLI directamente desde el navegador, sin necesidad de instalar herramientas adicionales en tu máquina local.

### **🔹 Características principales**
✅ **Preconfigurado**: AWS CloudShell ya viene con AWS CLI, Python, Git y otros paquetes útiles instalados.  
✅ **Acceso seguro**: Usa automáticamente las credenciales de IAM de tu sesión de AWS.  
✅ **Almacenamiento persistente**: Tiene 1 GB de almacenamiento por región para guardar archivos y scripts.  
✅ **Compatibilidad**: Disponible en múltiples regiones de AWS.  
✅ **Soporte para múltiples shells**: Puedes usar **Bash**, **PowerShell** y **Zsh**.

### **🔹 ¿Cómo acceder a AWS CloudShell?**
1️⃣ **Inicia sesión en AWS Console** 👉 [AWS CloudShell](https://console.aws.amazon.com/cloudshell)  
2️⃣ En la barra superior de la consola de AWS, haz clic en el ícono de **CloudShell**.  
3️⃣ Espera unos segundos mientras se inicia el entorno.  
4️⃣ ¡Listo! Ahora puedes ejecutar comandos de AWS CLI directamente.

### **🔹 Comandos útiles en AWS CloudShell**
✅ Verificar la versión de AWS CLI:
```sh
aws --version
```
✅ Listar los buckets de S3:
```sh
aws s3 ls
```
✅ Consultar las instancias en EC2:
```sh
aws ec2 describe-instances
```

### **📌 Cuándo usar AWS CloudShell**
🔹 Cuando necesitas ejecutar comandos de AWS CLI sin instalar nada en tu computadora.  
🔹 Para administrar recursos de AWS desde cualquier dispositivo con acceso a internet.  
🔹 Si trabajas con diferentes configuraciones y no quieres modificar tu máquina local.

🚀 **AWS CloudShell es una gran herramienta para administrar AWS sin complicaciones. ¡Pruébalo!**

**Lecturas recomendadas**

[AWS CloudShell endpoints and quotas - AWS General Reference](https://docs.aws.amazon.com/general/latest/gr/cloudshell.html)

[Curso de Terminal y Línea de Comandos - Platzi](https://platzi.com/cursos/terminal/)

## Roles IAM para AWS

Un **rol IAM** en AWS es una identidad con permisos específicos que puedes asignar a servicios, usuarios u otras cuentas de AWS. A diferencia de los usuarios de IAM, los roles no requieren credenciales (como contraseñas o claves de acceso); en su lugar, utilizan **credenciales temporales** que AWS genera automáticamente.

### **🔹 ¿Para qué sirven los roles IAM?**  

Los roles IAM permiten **asignar permisos de acceso temporal** a diferentes entidades, como:  

✅ **Servicios de AWS** (Ejemplo: permitir que una Lambda acceda a un bucket S3).  
✅ **Usuarios en la misma cuenta de AWS** (Ejemplo: acceso temporal a EC2 sin credenciales).  
✅ **Usuarios en otra cuenta de AWS** (Ejemplo: una cuenta externa accede a recursos compartidos).  
✅ **Aplicaciones en servidores on-premise** (Ejemplo: usar IAM Roles con federación de identidad).

### **🔹 Cómo crear un rol IAM en AWS**
### **📌 Opción 1: Desde la consola de AWS**
1️⃣ Ir a **AWS IAM** 👉 [Consola IAM](https://console.aws.amazon.com/iam/)  
2️⃣ En el menú de la izquierda, seleccionar **Roles**.  
3️⃣ Clic en **Crear rol**.  
4️⃣ **Seleccionar la entidad de confianza**:
   - AWS Service (para EC2, Lambda, etc.).
   - Another AWS Account (para acceso entre cuentas).
   - Web Identity o SAML 2.0 (para autenticación externa).  
5️⃣ **Asignar permisos** mediante políticas de acceso.  
6️⃣ **Nombrar el rol** y revisar los detalles.  
7️⃣ **Crear rol** y usarlo en el servicio correspondiente.

### **📌 Opción 2: Crear un rol IAM usando AWS CLI**
```sh
aws iam create-role --role-name MiRolS3 \
  --assume-role-policy-document file://policy.json
```
📌 **Ejemplo de `policy.json`** (permite a EC2 asumir el rol):  
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### **🔹 Casos de uso de los roles IAM**
💡 **Ejemplo 1:** Un **rol IAM para Lambda** que le permite acceder a S3:  
- Servicio de confianza: `lambda.amazonaws.com`.  
- Permisos: `AmazonS3ReadOnlyAccess`.  

💡 **Ejemplo 2:** Un **rol IAM para EC2** que permite acceder a DynamoDB:  
- Servicio de confianza: `ec2.amazonaws.com`.  
- Permisos: `AmazonDynamoDBFullAccess`.  

💡 **Ejemplo 3:** Un **rol IAM para acceso entre cuentas** (cross-account):  
- Permite que una cuenta externa asuma el rol con permisos limitados.

### **🔹 Diferencia entre un rol y un usuario IAM**
| Característica | Usuario IAM | Rol IAM |
|--------------|------------|--------|
| Usa credenciales fijas | ✅ Sí | ❌ No |
| Credenciales temporales | ❌ No | ✅ Sí |
| Puede usarse por servicios de AWS | ❌ No | ✅ Sí |
| Se usa para acceso entre cuentas | ❌ No | ✅ Sí |

### **🚀 Conclusión**
Los **roles IAM** permiten gestionar el acceso seguro a los recursos en AWS sin necesidad de credenciales estáticas. Son esenciales para automatización, buenas prácticas de seguridad y acceso entre cuentas o servicios.  

¿Quieres probarlo en la práctica? 🎯 **¡Crea un rol IAM y úsalo en EC2 o Lambda!** 🚀

## Práctica de roles en IAM

En esta práctica, crearás un **rol IAM** y lo asignarás a una instancia EC2 para que pueda acceder a un **bucket S3** sin necesidad de credenciales.

### **✅ Paso 1: Crear un rol IAM en la Consola de AWS**  
1️⃣ Ir a **AWS IAM** 👉 [Consola IAM](https://console.aws.amazon.com/iam/)  
2️⃣ En el menú izquierdo, seleccionar **Roles**.  
3️⃣ Clic en **Crear rol**.  
4️⃣ En la sección **Entidad de confianza**, seleccionar **AWS Service** y luego elegir **EC2**.  
5️⃣ Clic en **Siguiente**.  
6️⃣ En **Permisos**, buscar y seleccionar la política **AmazonS3ReadOnlyAccess**.  
7️⃣ Clic en **Siguiente** y asignar un nombre al rol, por ejemplo: `EC2S3ReadOnlyRole`.  
8️⃣ Revisar la configuración y hacer clic en **Crear rol**.

### **✅ Paso 2: Asignar el Rol IAM a una Instancia EC2**  
1️⃣ Ir a **AWS EC2** 👉 [Consola EC2](https://console.aws.amazon.com/ec2/)  
2️⃣ Seleccionar la instancia EC2 a la que se le asignará el rol.  
3️⃣ Clic en **Acciones** > **Seguridad** > **Modificar rol de IAM**.  
4️⃣ Seleccionar el rol creado (`EC2S3ReadOnlyRole`).  
5️⃣ Guardar los cambios.

### **✅ Paso 3: Probar el Acceso desde la Instancia EC2**  
1️⃣ Conectarse a la instancia EC2 usando SSH:  
```sh
ssh -i "llave.pem" ec2-user@<IP_PUBLICA_EC2>
```
2️⃣ Ejecutar el siguiente comando para verificar que el rol se ha asignado correctamente:  
```sh
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
```
Debería mostrar el nombre del rol asignado.  

3️⃣ Probar el acceso a S3 ejecutando:  
```sh
aws s3 ls
```
Si el rol está configurado correctamente, verás la lista de buckets S3 disponibles.

### **✅ Paso 4: Eliminar el Rol (Opcional)**
Si deseas eliminar el rol después de la prueba:  
1️⃣ En **IAM**, ir a **Roles** y seleccionar `EC2S3ReadOnlyRole`.  
2️⃣ Clic en **Eliminar rol** y confirmar.  

### **🚀 Conclusión**
Has creado un **rol IAM**, lo has asignado a una instancia **EC2**, y has verificado que puede acceder a **S3 sin credenciales**. 🔥 ¡Ahora puedes usar esta técnica en otros servicios de AWS! 💪

## Herramientas de seguridad en IAM

IAM proporciona varias herramientas y mejores prácticas para mejorar la seguridad de los accesos en AWS. A continuación, se presentan algunas de las más importantes:

### **✅ 1. Uso de Multi-Factor Authentication (MFA)**  
**MFA** agrega una capa adicional de seguridad al requerir un código temporal además de la contraseña.  
📌 **Herramienta**: IAM permite habilitar MFA para usuarios de la cuenta de AWS.  

**💡 Práctica recomendada:**  
- Habilitar MFA para usuarios root y administradores.  
- Utilizar MFA basado en hardware o aplicaciones como **Google Authenticator** o **AWS Virtual MFA**.

🔗 [Configuración IAM MFA](https://aws.amazon.com/iam/features/mfa/)

### **✅ 2. IAM Access Analyzer**  
📌 **Función:** Identifica **permisos excesivos** y posibles riesgos de acceso a recursos de AWS.  
📌 **Herramienta:** AWS IAM Access Analyzer.  

**💡 Práctica recomendada:**  
- Revisar permisos públicos en buckets S3, roles IAM, y políticas de acceso.  
- Configurar alertas cuando se detecten permisos abiertos innecesarios.  

🔗 [IAM Access Analyzer](https://aws.amazon.com/iam/features/analyze-access/)

### **✅ 3. AWS IAM Credential Report**  
📌 **Función:** Genera un informe con información sobre credenciales de los usuarios IAM.  

**💡 Práctica recomendada:**  
- Revisar credenciales que no se han usado en los últimos 90 días.  
- Eliminar **Access Keys** no utilizadas.  

📌 **Ejemplo de generación del informe (AWS CLI):**  
```sh
aws iam generate-credential-report
aws iam get-credential-report
```

🔗 [Credenciales de IAM](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_getting-report.html)

### **✅ 4. AWS IAM Policy Simulator**  
📌 **Función:** Permite probar políticas IAM antes de aplicarlas.  
📌 **Herramienta:** AWS IAM Policy Simulator.  

**💡 Práctica recomendada:**  
- Validar políticas antes de aplicarlas a roles, usuarios o grupos.  
- Probar si un usuario tiene permisos específicos para evitar configuraciones incorrectas.  

🔗 [IAM Policy Simulator](https://policysim.aws.amazon.com/)

### **✅ 5. AWS CloudTrail**  
📌 **Función:** Registra **todas las acciones** realizadas en IAM y otros servicios de AWS.  

**💡 Práctica recomendada:**  
- Habilitar CloudTrail para auditar accesos y cambios en IAM.  
- Configurar alertas en **AWS CloudWatch** para detectar accesos sospechosos.  

🔗 [AWS CloudTrail](https://aws.amazon.com/cloudtrail/)

### **✅ 6. AWS Organizations y SCP (Service Control Policies)**  
📌 **Función:** Permite restringir accesos a cuentas en AWS Organizations mediante **SCPs**.  

**💡 Práctica recomendada:**  
- Usar **SCPs** para bloquear servicios o regiones no permitidas.  
- Aplicar el principio de **mínimos privilegios** en todas las cuentas.  

🔗 [AWS Organizations](https://aws.amazon.com/organizations/)

### **✅ 7. IAM Roles y Access Keys Rotations**  
📌 **Función:** Los roles IAM eliminan la necesidad de usar **Access Keys** en instancias EC2 y otros servicios.  

**💡 Práctica recomendada:**  
- Usar **roles IAM** en lugar de **Access Keys**.  
- Si es necesario usar **Access Keys**, rotarlas periódicamente.  

🔗 [Rotación de claves](https://docs.aws.amazon.com/general/latest/gr/aws-access-keys-best-practices.html)

### **🚀 Conclusión**
Estas herramientas ayudan a **fortalecer la seguridad** en AWS IAM. Implementarlas reducirá riesgos y mejorará la administración de accesos en la nube. 🔐✨

## Práctica de las herramientas de seguridad

Aquí tienes una práctica guiada para aplicar las herramientas de seguridad en **IAM (Identity and Access Management)** en AWS.

### **🛠 Práctica de Herramientas de Seguridad en IAM**  

### **📌 Objetivo:**  
- Configurar y usar herramientas de seguridad en IAM.  
- Mejorar la protección de usuarios, roles y accesos en AWS.  

**🔷 Requisitos:**  
✅ Tener una cuenta de AWS con permisos de administrador.  
✅ Tener AWS CLI configurado en tu sistema.

### **1️⃣ Habilitar MFA en un usuario IAM**  
**Paso 1:** Inicia sesión en la consola de AWS y ve a **IAM → Usuarios**.  
**Paso 2:** Selecciona un usuario y haz clic en la pestaña **Credenciales de inicio de sesión**.  
**Paso 3:** En la sección de MFA, haz clic en **Asignar MFA** y sigue los pasos para configurar un dispositivo MFA (Google Authenticator o un token físico).  

📌 **Verificación:**  
- Intenta iniciar sesión con el usuario IAM y confirma que se requiere MFA.

### **2️⃣ Generar un informe de credenciales IAM**  
**Paso 1:** Ejecuta el siguiente comando en AWS CLI:  
```sh
aws iam generate-credential-report
aws iam get-credential-report --output text > credential_report.csv
```
📌 **Verificación:**  
- Abre el archivo `credential_report.csv` y revisa el estado de las credenciales de los usuarios.  

💡 **Acción recomendada:**  
- Deshabilita o elimina usuarios con **Access Keys** no usadas.

### **3️⃣ Simular una política IAM antes de aplicarla**  
**Paso 1:** Ve a [AWS IAM Policy Simulator](https://policysim.aws.amazon.com/).  
**Paso 2:** Selecciona un usuario o rol IAM.  
**Paso 3:** Agrega una política JSON y haz clic en **Run Simulation** para verificar qué acciones están permitidas.  

📌 **Verificación:**  
- Si la política es demasiado permisiva, ajústala antes de aplicarla.

### **4️⃣ Configurar CloudTrail para auditar eventos en IAM**  
**Paso 1:** Ve a la consola de **AWS CloudTrail**.  
**Paso 2:** Crea un nuevo **Trail** con almacenamiento en un bucket de S3.  
**Paso 3:** Habilita la opción de registrar **eventos de IAM**.  

📌 **Verificación:**  
- Revisa los registros en **S3** o en **CloudWatch Logs** para detectar accesos sospechosos.

### **5️⃣ Usar Access Analyzer para identificar accesos innecesarios**  
**Paso 1:** Ve a **IAM → Access Analyzer** en la consola de AWS.  
**Paso 2:** Crea un nuevo analizador.  
**Paso 3:** Revisa los reportes para identificar accesos abiertos al público o a otras cuentas de AWS.  

📌 **Verificación:**  
- Si encuentras accesos innecesarios, ajusta las políticas de IAM.

### **🚀 Conclusión**  
Con esta práctica, has aplicado herramientas clave para **proteger IAM** en AWS. Repite estos pasos regularmente para mantener una seguridad óptima. 🔐🚀

## Mejores prácticas dentro de IAM 

IAM es un servicio clave en AWS para gestionar usuarios, permisos y accesos. Aplicar mejores prácticas es esencial para garantizar la seguridad de los recursos en la nube.

### **1️⃣ Utilizar Usuarios y Roles en lugar de la Cuenta Root**  
📌 **Regla de oro:** La cuenta root debe usarse solo para tareas críticas (como configurar la facturación).  

✅ Crea usuarios IAM con permisos mínimos necesarios.  
✅ Usa **Roles IAM** en lugar de credenciales de acceso permanentes.  
✅ Habilita **MFA (Multi-Factor Authentication)** en la cuenta root.

### **2️⃣ Aplicar el Principio de Privilegios Mínimos (Least Privilege)**  
📌 **Regla:** Ningún usuario debe tener más permisos de los necesarios.  

✅ Asigna permisos específicos en lugar de `AdministratorAccess`.  
✅ Usa **políticas basadas en roles** en lugar de permisos directos a usuarios.  
✅ Revisa periódicamente los permisos de los usuarios y elimina accesos innecesarios.

### **3️⃣ Habilitar MFA en Todos los Usuarios IAM**  
📌 **Regla:** MFA reduce el riesgo de accesos no autorizados.  

✅ Usa **MFA basado en aplicaciones** (Google Authenticator, Authy, etc.).  
✅ Habilita MFA en la **cuenta root** y todos los usuarios IAM con acceso a la consola.  
✅ Para accesos programáticos, considera el uso de claves temporales de **AWS STS**.

### **4️⃣ Usar Roles IAM en lugar de Access Keys**  
📌 **Regla:** Las **Access Keys** permanentes deben evitarse o rotarse regularmente.  

✅ Usa **Roles IAM** para instancias EC2, Lambda, ECS, etc., en lugar de Access Keys.  
✅ Si necesitas Access Keys, usa **AWS Secrets Manager** para gestionarlas.  
✅ Habilita **Access Analyzer** para auditar accesos innecesarios a tus recursos.

### **5️⃣ Aplicar Políticas de Contraseñas Fuertes**  
📌 **Regla:** Implementar una política de contraseñas segura para usuarios IAM.  

✅ Configura políticas de contraseñas con:  
   - Longitud mínima de 12 caracteres.  
   - Requisitos de caracteres especiales, números y mayúsculas.  
   - Rotación periódica (cada 90 días).  
✅ Deshabilita el acceso de usuarios inactivos.

### **6️⃣ Monitorear y Auditar el Uso de IAM**  
📌 **Regla:** Siempre supervisa los accesos y actividades sospechosas.  

✅ Usa **AWS CloudTrail** para registrar cambios en IAM y accesos no autorizados.  
✅ Habilita **AWS Config** para rastrear cambios en políticas y roles.  
✅ Revisa regularmente el **Credential Report** en IAM para detectar claves antiguas o usuarios inactivos.

### **7️⃣ Usar AWS Organizations y Control Tower para Gestión Centralizada**  
📌 **Regla:** Gestiona múltiples cuentas AWS de forma segura con AWS Organizations.  

✅ Usa **Service Control Policies (SCPs)** para restringir acciones en cuentas secundarias.  
✅ Configura **Control Tower** para establecer reglas de seguridad en todas las cuentas.  
✅ Aplica la segmentación de recursos con **AWS Accounts y Organizational Units (OUs)**.

### **8️⃣ Limitar Accesos Públicos y No Necesarios**  
📌 **Regla:** Evita accesos públicos innecesarios a recursos de AWS.  

✅ Usa **AWS IAM Access Analyzer** para identificar accesos públicos o compartidos.  
✅ Configura **bucket policies** en S3 para evitar permisos globales (`"Principal": "*" `).  
✅ Usa **VPC Endpoints** para limitar el acceso a AWS desde redes privadas.

### **🚀 Conclusión**  
Seguir estas mejores prácticas en IAM reduce significativamente los riesgos de seguridad en AWS. La clave es **minimizar accesos**, **monitorear constantemente** y **aplicar MFA y roles IAM** en cada nivel.  

🔹 **¿Qué sigue?** Revisa regularmente las políticas y usa herramientas como AWS CloudTrail y Access Analyzer para detectar problemas de seguridad. ✅

## Modelo de responsabilidad compartida en IAM

AWS utiliza un **modelo de responsabilidad compartida**, lo que significa que tanto **AWS** como el **cliente** tienen responsabilidades en la seguridad de la nube.  

### **🔹 ¿Qué significa el modelo de responsabilidad compartida?**  
AWS divide la seguridad en dos partes:  
1️⃣ **Seguridad de la nube** (Responsabilidad de AWS).  
2️⃣ **Seguridad en la nube** (Responsabilidad del Cliente).

### **🔹 Responsabilidades de AWS en IAM**  
📌 AWS es responsable de la infraestructura subyacente y su seguridad.  

✅ Seguridad del hardware, redes y centros de datos.  
✅ Protección de la infraestructura global (servidores, almacenamiento, bases de datos).  
✅ Aplicación de parches y mantenimiento de hardware y software base.  
✅ Disponibilidad y redundancia de los servicios AWS.  

🛑 **Ejemplo:** AWS garantiza que IAM esté disponible y protegido contra vulnerabilidades en su infraestructura.

### **🔹 Responsabilidades del Cliente en IAM**  
📌 El cliente es responsable de **configurar y administrar los accesos y permisos** dentro de AWS.  

✅ **Gestión de usuarios y roles IAM** (crear y asignar permisos correctos).  
✅ **Aplicación del principio de privilegios mínimos** (evitar accesos innecesarios).  
✅ **Habilitar MFA** para proteger las cuentas de IAM.  
✅ **Rotación y protección de Access Keys** (evitar claves de acceso permanentes).  
✅ **Auditoría de accesos con AWS CloudTrail y Access Analyzer**.  

🛑 **Ejemplo:** Si un usuario IAM tiene permisos excesivos y su clave de acceso es comprometida, el cliente es responsable de ese fallo de seguridad.

### **🔹 Resumen Visual del Modelo de Responsabilidad Compartida**  

| **Responsabilidad**               | **AWS** ✅ | **Cliente** ✅ |
|-----------------------------------|-----------|---------------|
| Seguridad de la infraestructura  | ✅         | ❌             |
| Gestión de usuarios IAM          | ❌         | ✅             |
| Aplicación de MFA                | ❌         | ✅             |
| Configuración de roles y permisos| ❌         | ✅             |
| Monitoreo de accesos con CloudTrail | ❌     | ✅             |
| Protección contra ataques a la infraestructura | ✅ | ❌          |

### **🔹 Buenas Prácticas para la Seguridad en IAM**  
✅ Usa **roles IAM** en lugar de Access Keys.  
✅ Habilita **MFA en todos los usuarios IAM**.  
✅ Aplica el **principio de privilegios mínimos**.  
✅ Usa **AWS CloudTrail** para monitorear accesos y cambios.  
✅ Revisa periódicamente el **Credential Report** en IAM.

### **🚀 Conclusión**  
El **modelo de responsabilidad compartida en AWS IAM** establece que AWS protege la infraestructura, pero el cliente **debe administrar correctamente los accesos y permisos** dentro de la nube.  

🔹 **¿Qué sigue?** Implementar buenas prácticas en IAM para evitar brechas de seguridad. ✅