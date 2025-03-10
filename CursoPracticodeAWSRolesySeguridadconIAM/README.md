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

