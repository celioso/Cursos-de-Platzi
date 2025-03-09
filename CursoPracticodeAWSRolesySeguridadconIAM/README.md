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