# Curso Práctico de Cloud Computing con AWS

## Computación en la Nube con AWS: S2, Docker, Lambda y más

Esto es lo que verás en este curso de **Cloud Computing con AWS**:

- **Introducción**: AWS cómputo se refiere a cualquier producto de AWS que nos permite servir algún archivo, procesar o calcular algo.
- **EC2**: Son máquinas virtuales que nos renta Amazon por segundo. Hay Linux o Windows. Podemos elegir número de CPUs, RAM, discos duros, tipo de conectividad, entre otras opciones.
- **Lightsail**: Es un producto particular porque es un VPS sobre Amazon similar a Dreamhost o Digital Ocean estando en la red de Amazon conservando los bajos costos de los VPS comerciales.
- **ECR/ECS/EKS**: ECR es donde registramos contenedores, ECS es el producto de Amazon para Docker y EKS es el producto de Amazon para Kubernetes.
- **Lambda**: Es la infraestructura de Amazon para poder correr diferentes funciones.
- **Elastic Beanstalk**: Permite correr diversos software o cargas productivas, pudiendo autoescalar hacia arriba o hacia abajo de manera automática.

Nuestro proyecto será un software que nos permitirá mostrar diferentes citas en pantalla. Cada que recarguemos pantalla veremos una nueva cita.

**Lecturas recomendadas**

[Amazon Lightsail](https://aws.amazon.com/es/lightsail/ "Amazon Lightsail")

[Amazon ECR | Amazon Web Services](https://aws.amazon.com/es/ecr/ "Amazon ECR | Amazon Web Services")

[AWS | Gestión de contenedores (ECS) compatible con los de Docker](https://aws.amazon.com/es/ecs/ "AWS | Gestión de contenedores (ECS) compatible con los de Docker")

[Amazon EKS – Servicio de Kubernetes administrado](https://aws.amazon.com/es/eks/ "Amazon EKS – Servicio de Kubernetes administrado")

[AWS | Lambda - Gestión de recursos informáticos](https://aws.amazon.com/es/lambda/ "AWS | Lambda - Gestión de recursos informáticos")

[AWS | Elastic beanstalk para aplicaciones web desarrolladas con Java](https://aws.amazon.com/es/elasticbeanstalk/ "AWS | Elastic beanstalk para aplicaciones web desarrolladas con Java")

## Configuración de Instancias S2 en Amazon AWS

Parece que mencionaste **"Instancias S2"**, pero probablemente te refieras a **Instancias EC2 (Elastic Compute Cloud)** en Amazon AWS. No existen “S2” en el contexto de cómputo, por lo que asumiré que te interesa la **configuración de instancias EC2**, que es el servicio de Amazon para ejecutar servidores virtuales.

A continuación te explico cómo **configurar una instancia EC2 paso a paso**:

### 🛠️ **Pasos para configurar una instancia EC2 en AWS**

#### 1. **Ingresar a la consola de AWS**

* Ve a [https://console.aws.amazon.com/](https://console.aws.amazon.com/)
* Inicia sesión con tu cuenta.

#### 2. **Buscar EC2**

* En el buscador de servicios, escribe “EC2” y haz clic.

#### 3. **Lanzar una nueva instancia**

* En el panel de EC2, haz clic en **“Launch Instance”**.

#### 4. **Configurar la instancia**

1. **Nombre de la instancia** (opcional): Asigna un nombre como “Servidor-Pruebas”.
2. **Sistema Operativo** (AMI): Elige una Amazon Machine Image, como:

   * Amazon Linux 2
   * Ubuntu Server 20.04
   * Windows Server
3. **Tipo de instancia**:

   * T2.micro (gratis elegible)
   * T3.medium, T3.large (más recursos)
4. **Par de claves (Key Pair)**:

   * Crea o selecciona un par de claves para conectarte por SSH.
5. **Configuración de red y firewall**:

   * Elige una VPC o usa la predeterminada.
   * Asegúrate de abrir los puertos necesarios (ej. 22 para SSH, 80 para HTTP).

#### 5. **Volumen de almacenamiento**

* Usa el predeterminado o cambia según tus necesidades.
* Ejemplo: 8GB SSD para uso general.

#### 6. **Revisar y lanzar**

* Revisa la configuración y haz clic en **“Launch Instance”**.

#### 7. **Conectarte a la instancia**

* Una vez lanzada, selecciona la instancia y haz clic en **“Connect”**.
* Usa la terminal o PuTTY (Windows) con la clave privada (`.pem`) para conectarte por SSH.

### 📌 Consejos Adicionales

* Usa **grupos de seguridad (Security Groups)** para controlar el acceso.
* Crea una **Elastic IP** si necesitas una IP pública fija.
* Usa **Amazon CloudWatch** para monitoreo básico.

### Resumen

#### ¿Qué son los EC2?

- Son máquinas virtuales, llamadas instancias en Amazon que te van a permitir correr diferentes software en diferentes sistemas operativos con diferentes configuraciones.
- Amazon tiene ya unas imágenes preconfiguradas llamadas **AMIs** .
- Podremos seleccionar diferentes tamaños de CPU´s y su cantidad, cantidad de RAM, espacio en disco, diferente conectividad, entre otros. El costo depende de las capacidades que especifiquemos.

#### Arquitectura de EC2:

Podemos crear diferentes imágenes, por ejemplo una con Ubuntu, configurando o instalando diferentes software, finalmente haciendo una imágen con ello. Las imágenes van hacia una instancia de [EC2](https://platzi.com/clases/1323-aws-cloud-practico/12577-que-es-ec2/ "EC2"), seleccionando capacidad que necesitamos en la máquina virtual.
Asociado a esto, están los temas de redes como los grupos de seguridad, los balanceadores de cargas hacia los cuales llega el tráfico de la web externo como interno.
De storage tenemos uno que es efímero que sólo existe mientras la máquina esté prendida, y el otro es un bloque elástico que permanece a pesar de borrar la máquina y de él podemos hacer copias en caso de que vaya evolucionando otro proyecto.

![EC2 arquitectura](images/ArquitecturaElasticComputeCloud(EC2).png)

## Configuración de Instancias EC2 en Amazon AWS

¡Perfecto! Aquí tienes una **guía completa y clara para configurar instancias EC2 en Amazon AWS**, desde cero:

### ☁️ Configuración de Instancias EC2 en Amazon AWS

### ¿Qué es EC2?

Amazon EC2 (Elastic Compute Cloud) te permite crear **servidores virtuales** (instancias) en la nube para alojar aplicaciones, sitios web, entornos de desarrollo, etc.

### 🔧 Pasos para configurar una Instancia EC2

### 1. **Ingresar a la consola de AWS**

* Ve a: [https://aws.amazon.com/console/](https://aws.amazon.com/console/)
* Inicia sesión con tu cuenta de AWS.

### 2. **Ir al servicio EC2**

* En la barra de búsqueda escribe `EC2` y haz clic en el resultado.
* Esto te llevará al **Dashboard de EC2**.

### 3. **Lanzar una nueva instancia**

Haz clic en el botón **"Launch Instance"**.

### 4. **Configurar la nueva instancia**

#### 📌 a) Nombre

* Ponle un nombre, por ejemplo: `Servidor-Web-Ubuntu`.

#### 📌 b) Amazon Machine Image (AMI)

Selecciona una imagen del sistema operativo, como:

* Ubuntu Server 22.04 LTS
* Amazon Linux 2
* Windows Server 2022
* Red Hat, Debian, etc.

#### 📌 c) Tipo de instancia

Selecciona el tipo de hardware:

* `t2.micro` o `t3.micro`: 1 vCPU y 1 GB de RAM (elegible para capa gratuita)
* `t3.medium`, `m5.large`, etc., si necesitas más recursos.

#### 📌 d) Par de claves (Key pair)

* Crea o selecciona un **Key Pair (.pem)** para acceder a tu servidor vía SSH.

  * **Importante:** descarga y guarda el archivo `.pem` con seguridad.
  * Si usas Windows, usa PuTTY o WSL con el `.pem`.

#### 📌 e) Configuración de red y seguridad

* Elige una VPC y subred (usa la predeterminada si no sabes).
* **Security Group (firewall):**

  * Agrega las reglas necesarias, por ejemplo:

    * SSH: puerto 22 desde tu IP
    * HTTP: puerto 80 (si vas a montar un sitio web)
    * HTTPS: puerto 443 (opcional)

### 5. **Almacenamiento (volumen)**

* Por defecto, se asigna un volumen de 8 GB (puedes modificarlo).
* Tipo `gp2` o `gp3` (SSD uso general).

### 6. **User Data (opcional)**

Puedes insertar un script para configurar automáticamente tu instancia al arrancar. Por ejemplo, para instalar Apache:

```bash
#!/bin/bash
sudo apt update
sudo apt install apache2 -y
sudo systemctl start apache2
```

### 7. **Revisar y lanzar**

* Verifica todos los datos.
* Haz clic en **"Launch Instance"**.

---

### 8. **Conectarte a la instancia**

Una vez en estado “running”:

1. Haz clic en **Connect**
2. Usa SSH desde la terminal (Linux/macOS) o WSL (Windows):

```bash
chmod 400 tu-clave.pem
ssh -i "tu-clave.pem" ubuntu@<IP_PUBLICA>
```

### ✅ Buenas Prácticas

* Usa **Elastic IP** si necesitas IP pública fija.
* Configura **CloudWatch** para monitorear rendimiento.
* Utiliza **IAM Roles** si tu instancia accede a otros servicios AWS.
* Realiza **Backups (Snapshots)** regularmente.

### Resumen

Cosas a tener en cuenta al momento de crear tu **EC2**:

- Hay ocasiones en las cuales puedes entrar y no ver tus instancias creadas. Esto puede pasar porque no seleccionaste la región adecuada o la que tenías al momento de crearlas.
- Al momento de crear las imágenes se recomienda usar la de Amazon ya que viene actualizada con los últimos drivers.
- La sección T2/T3 Unlimited en la configuración de la instancia nos sirve si necesitamos mucha CPU o red, al habilitarla, Amazon nos lo dará sin límite. El problema es que tiende a ser más costoso.
- Es muy útil al momento de poner tag que se use uno aunque sea un nombre para recordar para qué sirve la máquina.
- Para conectarnos a la máquina debemos crear una llave. Es importante guardarla en un lugar seguro haciéndole una copia de seguridad ya que si no se tiene la llave, no es posible conectarse por medio de SSH.

[¿Qué es EC2?](https://platzi.com/clases/1323-aws-cloud-practico/12577-que-es-ec2/ "¿Qué es EC2?")

## Instalación de Proyecto de Prueba en Instancia EC2 con Secure Shell

Perfecto, Mario. A continuación te muestro **cómo instalar un proyecto de prueba en una instancia EC2 usando SSH (Secure Shell)**. Este ejemplo usará una aplicación simple con Node.js, pero si prefieres Python, PHP o cualquier otro stack, puedo adaptarlo.

### 🚀 Instalación de un Proyecto de Prueba en Instancia EC2 (vía SSH)

### 🛠 Requisitos Previos

1. Tener una instancia EC2 corriendo (por ejemplo: Ubuntu).
2. Tener el archivo `.pem` (par de llaves de la instancia).
3. Tener tu IP autorizada en el Security Group (puerto 22 abierto para SSH).

### ✅ 1. Conéctate por SSH a tu instancia

```bash
chmod 400 tu-clave.pem
ssh -i "tu-clave.pem" ubuntu@<IP_PUBLICA_DE_LA_INSTANCIA>
```

### ✅ 2. Actualiza el sistema

```bash
sudo apt update && sudo apt upgrade -y
```

### ✅ 3. Instala Node.js y Git

```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs git
```

> Puedes verificar:

```bash
node -v
npm -v
```

### ✅ 4. Clona un proyecto de prueba

```bash
git clone https://github.com/heroku/node-js-sample.git proyecto-prueba
cd proyecto-prueba
```

### ✅ 5. Instala las dependencias

```bash
npm install
```

### ✅ 6. Ejecuta la aplicación

```bash
npm start
```

Por defecto, esta app corre en el puerto `5000`.

### ✅ 7. Abre el puerto en el Security Group (si aún no lo hiciste)

En la consola de AWS:

* EC2 > Security Groups > selecciona tu grupo.
* Edita las reglas de entrada > Agrega:

  * **Tipo:** Custom TCP
  * **Puerto:** 5000
  * **Origen:** Tu IP o 0.0.0.0/0 (¡solo para pruebas!)

### ✅ 8. Accede desde el navegador

Abre:

```
http://<IP_PUBLICA_DE_LA_INSTANCIA>:5000
```

### 🧠 Tips adicionales

* Para apps en producción, puedes usar **Nginx como proxy reverso**.
* Si prefieres mantener la app corriendo aunque cierres la terminal SSH, puedes usar `pm2`:

```bash
sudo npm install -g pm2
pm2 start index.js
pm2 startup
pm2 save
```

### Resumen

Cosas a tener en cuenta al momento de instalar tu proyecto:

- Si tienes Linux o MAC, ya cuentas con la terminal para poderte conectar por medio de SSH; si tienes Windows, es necesario usar un software como MobaXterm que es gratis para uso personal.
- El comando que debes usar es “sudo apt install apache2 git libapache2-mod-php -y”
- Si acabas de iniciar tu máquina, es posible que no encuentre los paquetes, ya que los DNS no son los correctos. Con “apt-get update” lo solucionas.
- La dirección del repositorio usado en clase es: [https://github.com/mauropm/quotes-generator](https://github.com/mauropm/quotes-generator "https://github.com/mauropm/quotes-generator")

**Instalar complementos:**

`sudo yum install -y httpd git php` 

**actualaizar** 

`yum update` y `yum upgrade`

**mover los archivos**

`cp index.php quotes.txt /var/www/html/`

El error que estás viendo se debe a que estás usando `yum`, el gestor de paquetes de **Amazon Linux** o **CentOS/RHEL**, y estás intentando instalar paquetes que son propios de **Ubuntu/Debian**, como `apache2` y `libapache2-mod-php`.


### ✅ Solución para Amazon Linux / RHEL / CentOS:

Debes usar los nombres correctos de paquetes para esta familia de sistemas. Aquí va el equivalente correcto:

```bash
sudo yum install -y httpd git php
```

### 🧾 Explicación:

* `httpd`: es el nombre del paquete de **Apache** en Amazon Linux / CentOS.
* `php`: instala PHP.
* `git`: para clonar o descargar código.

### ✅ Luego de la instalación:

1. **Inicia Apache**:

```bash
sudo systemctl start httpd
```

2. **Activa Apache al inicio**:

```bash
sudo systemctl enable httpd
```

3. **Coloca tus archivos en** `/var/www/html/`:

```bash
sudo rm -rf /var/www/html/*
sudo cp -r tu_proyecto/* /var/www/html/
```

4. **Reinicia el servicio si haces cambios**:

```bash
sudo systemctl restart httpd
```



**Lecturas recomendadas**

[MobaXterm free Xserver and tabbed SSH client for Windows](https://mobaxterm.mobatek.net/ "MobaXterm free Xserver and tabbed SSH client for Windows")

[GitHub - mauropm/quotes-generator: A simple PHP application to generate a random quote as HTML.](https://github.com/mauropm/quotes-generator "GitHub - mauropm/quotes-generator: A simple PHP application to generate a random quote as HTML.")

[Apache2 Ubuntu Default Page: It works Momo!](https://annex.exploratorium.edu/ "Apache2 Ubuntu Default Page: It works Momo!")

## Creación de Imágenes AMI en AWS EC2 paso a paso

Aquí tienes una **guía paso a paso** para crear una imagen AMI (Amazon Machine Image) desde una instancia EC2 en AWS. Esto es útil para clonar una instancia con su configuración actual, aplicaciones y datos.

### ✅ Paso a Paso: Crear una AMI desde una Instancia EC2

### 🔹 1. Accede a la Consola de AWS

* Ve a [https://console.aws.amazon.com/ec2/](https://console.aws.amazon.com/ec2/)
* Asegúrate de estar en la **región correcta** (arriba a la derecha).

### 🔹 2. Selecciona tu Instancia EC2

* En el menú izquierdo, haz clic en **"Instances"**.
* Marca la casilla de la instancia que quieres convertir en AMI.

### 🔹 3. Crear Imagen (AMI)

* Haz clic en **Actions** > **Image and templates** > **Create image**.

### 🔹 4. Configura la Imagen

Completa los siguientes campos:

* **Image name**: nombre de tu AMI (ej. `ami-mi-servidor-web`).
* **Image description** (opcional): breve descripción.
* **No reboot**: si la marcas, no reiniciará la instancia, pero puede hacer que la imagen no sea consistente.
* **Instance volumes**: confirma los volúmenes (puedes cambiarlos si quieres).

### 🔹 5. Crear Imagen

* Haz clic en **Create image**.
* Aparecerá un mensaje indicando que la imagen está en proceso.

### 🔹 6. Verificar el Progreso

* En el menú izquierdo, ve a **AMIs (Amazon Machine Images)**.
* Verifica el estado de tu imagen bajo la columna **Status** (espera que diga `available`).

### 🔹 7. Lanzar Instancia desde la AMI

Una vez la AMI esté disponible:

1. En la sección **AMIs**, selecciona tu imagen.
2. Haz clic en **Launch instance from image**.
3. Configura la nueva instancia como cualquier otra (tipo, red, almacenamiento, etc.).

### ✅ ¿Cuándo usar AMIs?

* Para crear ambientes idénticos rápidamente.
* Para tener un backup de una configuración específica.
* Para escalar horizontalmente tu aplicación.

### Resumen

Crear una imagen es muy útil porque cuando quieras crear una instancia nueva, podrás seleccionar la imagen, ahorrándote los pasos de instalación.

Cosas a tener en cuenta al momento de crear imágenes de instancias:

Creando una imagen te encontrarás con la opción de No reboot, si no se selecciona, Amazon va a apagar nuestra instancia para poder hacer la copia; si se selecciona, la instancia no se apagará, corriendo el riesgo de que pueda salir una copia corrupta. Se recomienda reiniciar la instancia.
Si estás en producción y la instancia que tienes se quedó corta en capacidades, seleccionarías que no se reinicie, para hacer tu copia y crear una nueva instancia con esta copia.
Si seleccionaste que sí se reiniciara la instancia, tu IP pública cambiará y no podrás conectarte a tu máquina con la anterior IP.

## Creación y gestión de snapshots en Amazon S2

Parece que hubo una pequeña confusión: **Amazon S2 no existe** como servicio. Tal vez te referías a:

* **Amazon S3 (Simple Storage Service)**: usado para almacenar objetos (archivos).
* **Amazon EC2 / EBS Snapshots**: gestión de snapshots de volúmenes de almacenamiento **EBS** asociados a **instancias EC2**.

Dado que hablaste de “snapshots”, lo correcto es:
👉 **Creación y gestión de snapshots en Amazon EBS** (no S3 ni S2).

### 📸 ¿Qué es un Snapshot en Amazon EBS?

Un **snapshot** es una copia puntual de un volumen EBS. Se almacena de forma incremental en S3 (de forma interna), y permite restaurar datos o crear nuevos volúmenes idénticos.

### ✅ Paso a Paso: Crear un Snapshot en EBS

### 🔹 1. Accede a la Consola de AWS

* Ve a [https://console.aws.amazon.com/ec2](https://console.aws.amazon.com/ec2)

### 🔹 2. Encuentra tu Volumen EBS

* En el panel izquierdo: **Elastic Block Store > Volumes**
* Selecciona el volumen del que deseas hacer snapshot.

### 🔹 3. Crear Snapshot

* Haz clic en **Actions** > **Create snapshot**
* Llena:

  * **Name**: nombre del snapshot.
  * **Description**: algo descriptivo (opcional).
* Haz clic en **Create snapshot**

### 🔹 4. Verifica el Progreso

* Ve a **Elastic Block Store > Snapshots**
* Aparecerá el snapshot con su **Status** (`pending` → `completed`)

### ✅ Restaurar un Volumen desde un Snapshot

1. En **Snapshots**, selecciona el snapshot.
2. Clic en **Actions > Create volume**.
3. Elige la zona de disponibilidad (debe coincidir con la de la instancia donde lo vas a usar).
4. Haz clic en **Create volume**.
5. Luego puedes **adjuntar ese volumen** a una instancia EC2 desde **Volumes > Attach volume**.

### ⚙️ Gestión por Línea de Comandos (AWS CLI)

**Crear snapshot:**

```bash
aws ec2 create-snapshot \
  --volume-id vol-xxxxxxxx \
  --description "Snapshot de respaldo manual"
```

**Listar snapshots:**

```bash
aws ec2 describe-snapshots --owner-ids self
```

**Eliminar snapshot:**

```bash
aws ec2 delete-snapshot --snapshot-id snap-xxxxxxxx
```

### 🧠 Buenas prácticas

* Automatiza snapshots con **Data Lifecycle Manager (DLM)**.
* Etiqueta snapshots para facilitar la búsqueda.
* Elimina snapshots antiguos si ya no son necesarios.

### Resumen

Cuando creas una imagen, vas a poder reproducir esa instancia con el mismo sistema operativo, software y capacidades, estás haciendo una copia del sistema al completo. Si quisieras hacer una copia de una sola de sus características, por ejemplo el software, ahí es donde usarías un **Snapshot** del volumen que es el disco duro. Esto se hace en situaciones especiales para añadir un volumen a una máquina virtual que ya esté corriendo.

Se recomienda crear una imagen nueva o AMI cada vez que hagas un cambio mayor en la instancia, versionando a través de imágenes para hacer rollback en caso de que el update falle o la configuración sea errónea.

**Lecturas recomendadas**

[Precios de Amazon EBS: Amazon Web Services](https://aws.amazon.com/es/ebs/pricing/ "Precios de Amazon EBS: Amazon Web Services")

## Configuración de IPs elásticas en instancias S2 de Amazon

Parece que nuevamente te refieres a **instancias S2**, pero en AWS no existe tal servicio. El servicio correcto es:

👉 **Amazon EC2 (Elastic Compute Cloud)**
Y lo que necesitas es:

🎯 **Configuración de direcciones IP elásticas (Elastic IPs) en instancias EC2**

### ✅ ¿Qué es una IP Elástica?

Una **Elastic IP** (EIP) es una dirección IP pública fija que puedes asociar a una instancia EC2. A diferencia de una IP pública estándar (que cambia si detienes y arrancas una instancia), la Elastic IP se mantiene.

### 🧭 Paso a Paso: Configurar una IP Elástica en EC2

### 🔹 1. Accede a la Consola EC2

Ve a: [https://console.aws.amazon.com/ec2](https://console.aws.amazon.com/ec2)

### 🔹 2. Asignar una Elastic IP

1. En el menú izquierdo: **Network & Security > Elastic IPs**
2. Clic en **Allocate Elastic IP address**
3. Deja la opción por defecto (**Amazon's pool of IPv4 addresses**) y presiona **Allocate**

### 🔹 3. Asociar la Elastic IP a tu Instancia

1. Selecciona la nueva Elastic IP.
2. Clic en **Actions > Associate Elastic IP address**
3. Selecciona:

   * **Instance**: la instancia EC2 a la que deseas asociarla.
   * **Private IP address**: déjalo por defecto si solo tienes una IP privada.
4. Clic en **Associate**

✅ ¡Listo! Ahora tu instancia EC2 tiene una IP pública fija.

### 🚨 Consideraciones Importantes

* **Amazon cobra por IPs elásticas no asociadas** (inactivas).
* Una Elastic IP se puede **desasociar y reasociar** a otra instancia.
* Si eliminas una instancia, la IP elástica no se pierde (pero sí queda inactiva hasta que la reutilices).

### ⚙️ Configuración desde AWS CLI

**Asignar IP elástica:**

```bash
aws ec2 allocate-address --domain vpc
```

**Asociar IP a instancia:**

```bash
aws ec2 associate-address \
  --instance-id i-xxxxxxxxxxxxxxxxx \
  --allocation-id eipalloc-xxxxxxxx
```

**Resumen**

Cuando reinicies o apagues tu instancia, la IP pública asignada muy probablemente cambiará. En muchos casos esto no es lo deseado y vamos a querer tener una IP que no cambie.
Amazon tiene la solución a este problema ya que nos ofrece el servicio para comprar una IP estática y asignarla a cualquiera de nuestras instancias.

## Configuración de Balanceadores de Carga en S2

Parece que otra vez mencionas "S2", pero lo correcto es **EC2** en Amazon Web Services. Vamos a enfocarnos en lo que realmente buscas:

### ✅ **Configuración de Balanceadores de Carga (Load Balancers) en Amazon EC2**

Amazon proporciona el servicio **Elastic Load Balancing (ELB)** para distribuir automáticamente el tráfico entrante entre varias instancias EC2 y garantizar alta disponibilidad, escalabilidad y tolerancia a fallos.

### 🧭 Paso a Paso para Configurar un Load Balancer (ELB)

### 🔹 1. Accede a la Consola EC2

Ve a: [https://console.aws.amazon.com/ec2](https://console.aws.amazon.com/ec2)

### 🔹 2. Crear el Load Balancer

1. En el menú izquierdo, haz clic en **Load Balancers**
2. Clic en **Create Load Balancer**
3. Selecciona el tipo:

   * **Application Load Balancer (ALB)** – para HTTP/HTTPS con balanceo de nivel 7
   * **Network Load Balancer (NLB)** – para alto rendimiento (nivel 4, TCP/UDP)
   * **Classic Load Balancer** – versión antigua, no recomendada para nuevos proyectos

✅ Recomendado: **Application Load Balancer (ALB)** para sitios web

### 🔹 3. Configuración del ALB

1. **Nombre**: Ponle un nombre al LB
2. **Esquema**: Internet-facing (público) o Internal (privado)
3. **Listeners**: HTTP (puerto 80), HTTPS (puerto 443) si tienes certificado SSL
4. **VPC y Subnets**: selecciona al menos 2 subredes en distintas zonas de disponibilidad

### 🔹 4. Configurar el Target Group

1. Tipo de destino: **Instance** o **IP**
2. Protocolo: HTTP
3. Puerto: 80 (o el puerto de tu app)
4. Crea un nuevo Target Group
5. **Asocia tus instancias EC2**

### 🔹 5. Configurar el Health Check

* Path: `/` o tu endpoint de salud
* Thresholds: configura los valores de éxito/error para marcar instancias como saludables

### 🔹 6. Revisa y Crea

* Revisa toda la configuración
* Haz clic en **Create**

✅ ¡Tu balanceador de carga estará listo en minutos!

### 🌐 Acceder a tu aplicación

Una vez activo, usa el **DNS del Load Balancer** para acceder a tu app. Lo encuentras en la sección de descripción del ELB, algo así como:

```
my-load-balancer-1234567890.us-east-1.elb.amazonaws.com
```

### 🚀 Extras recomendados

* **Auto Scaling**: para que tus instancias EC2 se escalen automáticamente con base en la carga.
* **HTTPS**: usa **AWS Certificate Manager (ACM)** para instalar un certificado SSL gratuito y seguro.
* **Route 53**: puedes asociar un nombre de dominio personalizado al Load Balancer.

### Resumen

Un **Load balancer** o balanceador de carga lo puedes conectar a una o más instancias y lo que hace es balancear las solicitudes que le llegan pudiendo generar respuestas de una de las instancias que tiene a disposición dependiendo de su disponibilidad. Puedes balancear peticiones HTTP, HTTPS o TCP con los servicios de **AWS**.

Cuando creamos un load balancer, podemos ver en sus configuraciones básicas un DNS el cual podemos usar en **Route 53** como *CNAME* para ponerme un nombre de dominio o subdominio.

## Creación de Certificados y Balanceadores de Carga en AWS

### Introducción

Normalmente, cuando usas un balanceador de cargas, quieres prover dos distintos servicios:

- Https - puerto 443
- Http - puerto 80

Para dar servicio en el puerto 443, sigue las instrucciones que viene en la clase de load balancer, y la hora de anexar el puerto 443, te pedirá un certificado. Vamos a crear un nuevo certificado antes, para que solo selecciones el correcto.

Creando un certificado nuevo.

### Requisitos

- Poseer algún dominio o subdominio, que asignaras al certificado inicialmente, y después al balanceador de carga.
- Tener acceso a recibir el correo por parte del administrador del dominio, para poder anexar el certificado del lado de AWS. Si no lo tienes, necesitas acceso al DNS de ese dominio, para anexar un “entry” en el DNS, relacionado con la autenticación que requiere AWS para que muestres que eres el dueño del dominio (o que tienes control para él, si es que eres el administrador para alguna compañía).

### Actividades

- Ve al Certificate Manager. Visita la consola de amazon [http://console.aws.amazon.com/](http://console.aws.amazon.com/ "http://console.aws.amazon.com/") y de ahi ponle en el search “Certificate Manager”.
- Dale click en “Provision certificates”-> Get Started.
- Selecciona “Request a public certificate”
- Click en el botón “Request a certificate”.
- En la sección “Add a domain name”, pon un dominio wildcard al menos. Por ejemplo, en la clase de Networking & CDN en AWS compramos un dominio pruebaplatzi.de. En mi caso, pondría *.pruebaplatzi.de”. Tu tienes que poner *.tudominio.com, pensando que tu dominio se llama tudominio.com. Puedes anexar mas, como por ejemplo [www.tudominio.com](http://www.tudominio.com/ "www.tudominio.com"), test.tudominio.com, etc. Puedes anexar tantos como necesites, de tal manera que ese certificado cubra a todos tus servidores de prueba o desarrollo. Te recomiendo que si estas haciendo un producto, tu dominio producto.tudominio.com tenga su propio certificado solito, para que la gente no se confunda cuando lo revisa en el candado verde en tu navegador.
- Dale ‘Next’
- Selecciona que tipo de validación puedes hacer: si te llegan los mails de quien registro el dominio, selecciona mail. Si no es así, pero tienes acceso al DNS, selecciona DNS.
- En el caso de que manejes el dominio y el DNS del dominio en Route53, es mas sencillo ponerle DNS, y puedes ver la clase de Route53 para ver como anexas subdominios con el valor que te solicita AWS.
- Click en “Confirm request” y listo.
- En el caso que selecciones mail, revisa tu mail y dale click al url que te incluyen.
- Una vez que termines la validación, ya te aparecerá listado en los certificados.

### Creando el balanceador de carga

Ahora que ya tienes el certificado, puedes ir directamente a la consola de AWS, y crear o editar el balanceador de cargas, anexa el puerto 443/https, y cuando te pida el certificado, utiliza el que recién creaste.

Si tienes alguna duda o quisieras una guía paso a paso, ve al [Curso de Networking y Content Delivery en AWS](https://platzi.com/clases/1419-networking-content/15782-antes-de-empezar-el-curso/ "Curso de Networking y Content Delivery en AWS").

## Exploración de AMIs en Amazon Marketplace: Selección y Configuración

### ✅ **Exploración de AMIs en Amazon Marketplace: Selección y Configuración**

Las **Amazon Machine Images (AMIs)** son plantillas listas para lanzar instancias EC2 con un sistema operativo y, a menudo, aplicaciones preinstaladas. Puedes encontrarlas en el **AWS Marketplace** para ahorrar tiempo de configuración.

### 🔹 ¿Qué es el AWS Marketplace?

El [**AWS Marketplace**](https://aws.amazon.com/marketplace) es una tienda digital donde puedes encontrar **AMIs comerciales y gratuitas**, listas para lanzar, que incluyen:

* Sistemas operativos personalizados (Ubuntu con Docker, Amazon Linux con NGINX, etc.)
* Aplicaciones (WordPress, GitLab, ERP, CRM)
* Soluciones empresariales (SAP, Fortinet, Jenkins)

### 🧭 PASOS PARA EXPLORAR Y USAR UNA AMI DEL MARKETPLACE

### 1. **Entrar a la Consola EC2**

* Ir a: [https://console.aws.amazon.com/ec2](https://console.aws.amazon.com/ec2)
* Clic en **"Launch instance"**

### 2. **Explorar AMIs desde el Marketplace**

En el paso **"Application and OS Images (Amazon Machine Image)"**:

* Haz clic en **"Browse more AMIs"**
* Cambia a la pestaña **"AWS Marketplace"**
* Usa el buscador: ejemplos → `WordPress`, `Node.js`, `Docker`, `Nginx`, `Ubuntu`, `Jenkins`, etc.
* Fíjate en:

  * Calificación y soporte
  * Costo (muchas son **free tier eligible**)
  * Región donde está disponible

### 3. **Seleccionar la AMI**

* Haz clic en **"Select"** al lado de la imagen que desees
* Revisa los detalles y costos (algunas imágenes tienen tarifas por uso)
* Clic en **"Continue"**

### 4. **Configurar la Instancia EC2**

1. **Nombre de la instancia**
2. **Tipo de instancia** (ej: `t2.micro` si usas free tier)
3. **Par de claves (Key pair)** para conectarte vía SSH
4. **Configuración de red**:

   * VPC y Subnet
   * Grupo de seguridad (abre los puertos necesarios, ej. 22 para SSH, 80/443 para web)
5. **Almacenamiento**: revisa el tamaño del disco sugerido

### 5. **Lanzamiento y Acceso**

* Revisa el resumen
* Clic en **"Launch Instance"**
* Una vez iniciada, copia la **IPv4 pública** para conectarte o probar tu app.

### 🧰 CONSEJOS ÚTILES

* 🔒 **Seguridad**: Siempre revisa el **grupo de seguridad** y limita el acceso a los puertos que realmente necesitas.
* 🧪 **Pruebas**: Puedes probar primero una instancia pequeña antes de escalar.
* 💸 **Costo**: Algunas AMIs tienen cargos adicionales. Siempre revisa la letra pequeña.

### Resumen

La URL para acceder al marketplace es: [https://aws.amazon.com/marketplace](https://aws.amazon.com/marketplace "https://aws.amazon.com/marketplace")

En el marketplace podrás encontrar una gran cantidad de imágenes generadas para crear instancias. Algunas de ellas serán de pago y otras serán gratuitas sólo cobrando por la infraestructura de Amazon.
Esto puede resultar muy útil porque nos ahorra el tiempo de creación de instancias y sus copias, dándonos configuraciones perfectas para nuestras necesidades, que alguien ya resolvió con anterioridad.

**Lecturas recomendadas**

[AWS Marketplace: Homepage](https://aws.amazon.com/marketplace/ "AWS Marketplace: Homepage")

## Creación y configuración de instancias S2 en AWS

Parece que quieres saber sobre la **creación y configuración de instancias EC2 en AWS**, pero mencionaste "S2", que puede haber sido un error tipográfico. En Amazon Web Services (AWS), el servicio de cómputo bajo demanda correcto se llama **EC2 (Elastic Compute Cloud)**. Aquí tienes una guía paso a paso para **crear y configurar una instancia EC2**.

### ✅ **Creación y Configuración de Instancias EC2 en AWS**

### 🧭 PASO 1: Iniciar Sesión en la Consola de AWS

1. Ve a: [https://console.aws.amazon.com/](https://console.aws.amazon.com/)
2. Inicia sesión con tu cuenta de AWS.

### 🔹 PASO 2: Ir a EC2

1. En el buscador de servicios, escribe **EC2**.
2. Selecciona **EC2 (Elastic Compute Cloud)**.

### 🔹 PASO 3: Lanzar una Nueva Instancia

1. Haz clic en **"Launch Instance"** o **"Lanzar Instancia"**.
2. Completa los siguientes campos:

#### 3.1 Nombre de la instancia

* Ej: `ServidorWebPrueba`

#### 3.2 Selección de AMI (Imagen del Sistema Operativo)

* Amazon Linux 2023 (gratuito)
* Ubuntu Server
* Windows Server
* Marketplace (si necesitas algo específico como WordPress o Docker)

#### 3.3 Tipo de instancia

* **t2.micro** o **t3.micro** (elegible para capa gratuita si aplica)

#### 3.4 Par de claves (Key Pair)

* Selecciona un par existente o crea uno nuevo
* Este archivo `.pem` es necesario para conectarte vía SSH

#### 3.5 Configuración de Red

* VPC: deja por defecto (a menos que tengas una personalizada)
* Subred: selecciona una zona de disponibilidad
* Grupo de seguridad:

  * Abre el puerto **22 (SSH)** para Linux
  * Abre el puerto **80 (HTTP)** si es un servidor web
  * Opcional: 443 (HTTPS)

#### 3.6 Almacenamiento

* Tamaño predeterminado: 8 GB
* Puedes aumentarlo si necesitas más espacio

### 🔹 PASO 4: Lanzar la Instancia

1. Revisa los detalles
2. Haz clic en **"Launch Instance"**
3. Espera unos minutos hasta que el estado sea “running”

### 🔹 PASO 5: Conectarse a la Instancia

#### Desde Linux/macOS:

```bash
chmod 400 tu-par-de-claves.pem
ssh -i "tu-par-de-claves.pem" ec2-user@<IPv4 pública>
```

#### Desde Windows:

Usa PuTTY o Windows Terminal. Convierte el archivo `.pem` a `.ppk` si usas PuTTY.

### 🎯 PASO 6: Configurar tu Servidor (opcional)

Una vez dentro, puedes instalar software como Apache, Nginx, Git, etc.

#### Ejemplo para Amazon Linux:

```bash
sudo yum update -y
sudo yum install httpd -y
sudo systemctl start httpd
sudo systemctl enable httpd
```

### 🛡️ Recomendaciones de Seguridad

* **No dejes el puerto 22 abierto a todo el mundo** (`0.0.0.0/0`). Restringe por IP si es posible.
* **Usa roles IAM** si necesitas que tu instancia acceda a otros servicios de AWS (S3, DynamoDB, etc.).
* **Haz backups o snapshots regularmente**.

**Resumen**

El reto de esta clase consiste en crear una instancia de **EC2** y configurarle nuestro proyecto de frases motivacionales. Para probar que lo hiciste bien, copia la IP pública de la instancia en tu navegador y deberías poder ver una de las frases.

## Uso de VPS Lysel de Amazon para Proyectos Web

Parece que estás preguntando sobre el **uso de VPS Lysel de Amazon para proyectos web**, pero **"Lysel" no es un servicio oficial de AWS (Amazon Web Services)**. Es probable que haya un error de escritura o confusión en el nombre. Lo más cercano y relevante sería:

### 🔹 **Amazon EC2 (Elastic Compute Cloud) como VPS**

Amazon no ofrece un servicio llamado "Lysel", pero **EC2** cumple el rol de un **VPS (Servidor Privado Virtual)** y es ampliamente utilizado para **proyectos web**. A continuación te explico cómo puedes usar una instancia EC2 como un VPS para tus desarrollos web.

### ✅ **Cómo Usar Amazon EC2 como un VPS para Proyectos Web**

### 🧰 1. Elige tu sistema operativo

Cuando creas una instancia EC2, puedes seleccionar:

* **Amazon Linux 2023**
* **Ubuntu**
* **Debian**
* **Red Hat / CentOS**
* **Windows Server**

Para desarrollo web, **Ubuntu** es una opción muy común y fácil de usar.

### 🌐 2. Instala tu stack web

Una vez conectes por SSH, puedes instalar cualquier entorno como:

#### Ejemplo con Apache, PHP y MySQL (LAMP)

```bash
sudo apt update
sudo apt install apache2 php mysql-server -y
```

#### O usar Node.js + Express:

```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
```

### 🔐 3. Seguridad y acceso

* Abre los puertos 22 (SSH), 80 (HTTP) y 443 (HTTPS) en tu grupo de seguridad.
* Configura certificados SSL (puedes usar Let’s Encrypt).
* Puedes usar Amazon Route 53 para apuntar un dominio a tu VPS.

### 🧱 4. Almacenamiento y backups

* Configura volúmenes EBS adicionales si tu sitio requiere más espacio.
* Toma snapshots de tus volúmenes para respaldo.

### 🧠 Alternativas si quieres menos administración:

Si prefieres una plataforma más gestionada que EC2, puedes usar:

* **Elastic Beanstalk:** Despliegue automático de tu app con balanceo de carga y escalado.
* **Lightsail:** VPS simplificado y fácil de usar (incluye WordPress, Node.js, etc.).

### 🎯 Conclusión

Si te referías a un **VPS en Amazon para alojar proyectos web**, la mejor ruta es:

* EC2 si quieres control total
* Lightsail si quieres facilidad de uso

### Resumen

- Es un VPS (Virtual Private Server) como lo es **Digital Ocean** o el mismo **EC2** de Amazon. Tiene una IP pública y un dominio gratis. Su mayor diferencia con **EC2** es el precio más bajo.
- Se inicia en segundos
- Viene con varios templates pre-configurados como LAMP, Wordpress, Magento, etc.
- Su principal ventaja es su costo, bajo y predecible.
- Puedes aumentar o disminuir su capacidad cuando lo quieras, al alcance de un click.
- Puedes usar bases de datos.
- Puedes hacer respaldos como los **Snapshots**.
- Te ofrece la opción de restauración.
- Puede ser multi-región o multi-zonas (que en la misma zona geográfica tendrás diferentes data centers).

**Lecturas recomendadas**

[Amazon Lightsail](https://aws.amazon.com/es/lightsail/ "Amazon Lightsail")

## Creación y Gestión de VPS con Lysel en Amazon Marketplace

Gracias por la aclaración. Aun así, **no existe un producto llamado "Lysel" oficial de Amazon** ni en [Amazon Marketplace](https://aws.amazon.com/marketplace) ni como servicio dentro de AWS.

### 🔍 Posible confusión:

Es probable que te estés refiriendo a uno de estos:

#### 1. **Amazon Lightsail** (Probablemente querías decir "Lightsail", no "Lysel")

> 💡 **Amazon Lightsail** es el servicio de AWS que ofrece **VPS (Servidores Privados Virtuales)** listos para usar, con una configuración simplificada en comparación con EC2.

### ✅ Guía para la **Creación y Gestión de VPS con Amazon Lightsail**

### 🧾 Paso 1: Iniciar sesión en AWS

* Accede a: [https://lightsail.aws.amazon.com/](https://lightsail.aws.amazon.com/)
* Inicia sesión con tu cuenta de AWS.

### ⚙️ Paso 2: Crear tu VPS

1. Haz clic en **“Create instance”**.
2. Selecciona:

   * Región y zona.
   * Sistema operativo (**Linux/Unix** o **Windows**).
   * Imagen base (por ejemplo, Ubuntu, Debian, WordPress, Node.js, etc.).
3. Escoge el plan (desde \$3.50 USD/mes con 512MB RAM).
4. Asigna un nombre a la instancia.
5. Clic en **“Create Instance”**.

### 🔐 Paso 3: Conexión al VPS

* Desde el panel de Lightsail, puedes conectarte vía **SSH desde el navegador** o usar una llave `.pem` descargada para conexión remota:

```bash
ssh -i "LightsailDefaultKey.pem" ubuntu@your-static-ip
```

### 🌐 Paso 4: Configurar tu aplicación

Según el tipo de proyecto, puedes instalar:

* Apache/Nginx + PHP + MySQL (LAMP)
* Node.js
* WordPress
* Django, Rails, etc.

### 🌍 Paso 5: Asignar IP pública estática

* Desde el panel de Lightsail → “Networking”
* Asocia una IP estática a tu instancia

### 🔒 Paso 6: Configurar el firewall

* Desde el panel, permite los puertos necesarios (22, 80, 443, etc.)

### 🔄 Paso 7: Backups y snapshots

* Puedes crear **snapshots manuales o automáticos** desde el panel de instancia.

### 📦 Extras útiles

* **Conectar dominio**: usar Amazon Route 53 o tu proveedor DNS para apuntar tu dominio a la IP.
* **Certificados SSL gratuitos**: vía Let’s Encrypt (puedes usar Certbot).

### 🎯 Conclusión

Si buscabas "Lysel", muy probablemente te referías a **Amazon Lightsail**. Es una opción ideal para tener un VPS listo para proyectos web con mínimo esfuerzo técnico y buena escalabilidad.

### Resumen

El marketplace de **Lightsail** te permite elegir entre Linux y Windows, siendo esta opción la manera más económica de tener Windows de todos los servicios de Amazon.
Puedes instalar el SO más aplicaciones como Wordpress o Node.js; también puedes decidir por inicializar la imagen sólo con el sistema operativo, teniendo muchas opciones en la familia de Linux.
Instalar todos los parches de seguridad o actualizaciones es tu responsabilidad al igual que en **EC2**.

## Comparativa de Dicel vs EC2: Costos y Beneficios para Proyectos

Parece que hubo una confusión con el nombre: **“Dicel” no es un servicio oficial de AWS**. Probablemente estás refiriéndote nuevamente a **Amazon Lightsail**, ya que muchos usuarios lo confunden con nombres como “Lysel”, “Dicel”, etc.

Así que haremos una comparativa clara y útil entre:

> ✅ **Amazon Lightsail** (VPS simplificado)
> 🔄 **Amazon EC2** (Elastic Compute Cloud, más avanzado)


### 🧾 Comparativa: **Amazon Lightsail vs Amazon EC2**

| Característica                     | **Amazon Lightsail**                             | **Amazon EC2**                                    |
| ---------------------------------- | ------------------------------------------------ | ------------------------------------------------- |
| 🛠️ Configuración inicial          | Muy sencilla, ideal para principiantes           | Compleja, requiere experiencia técnica            |
| 💸 Precios                         | Fijos (desde \$3.50/mes)                         | Variables por uso (pago por hora/segundo)         |
| 📈 Escalabilidad                   | Limitada, ideal para proyectos pequeños/medianos | Escalable a nivel empresarial                     |
| 🧱 Recursos personalizables        | Planes fijos (RAM, CPU, SSD predefinidos)        | Totalmente personalizable (instancias, storage)   |
| 📡 Dirección IP estática           | Incluida fácilmente                              | Requiere configuración de Elastic IP              |
| 🔧 Acceso al sistema               | SSH / navegador / llaves .pem                    | SSH / llaves .pem                                 |
| 📦 Imágenes preinstaladas          | WordPress, LAMP, Node.js, etc.                   | AMIs básicas, tú eliges e instalas lo que quieras |
| 📊 Monitoreo                       | Básico                                           | Avanzado con CloudWatch                           |
| 🔐 Seguridad                       | Firewall básico en Lightsail                     | Control completo vía Grupos de Seguridad y IAM    |
| 🔁 Snapshots                       | Manuales y automáticos disponibles               | Volúmenes EBS con snapshots                       |
| ☁️ Integración con otros servicios | Limitada                                         | Total con todo el ecosistema AWS                  |
| 🧑 Público ideal                   | Freelancers, emprendedores, pequeñas empresas    | Empresas medianas y grandes, desarrolladores pro  |

### 💰 **Comparativa de costos** (Referencia: mayo 2025)

### 🔹 **Amazon Lightsail** (Precios fijos mensuales)

| Plan             | RAM    | CPUs | SSD   | Transferencia | Precio     |
| ---------------- | ------ | ---- | ----- | ------------- | ---------- |
| Básico           | 512 MB | 1    | 20 GB | 1 TB          | \$3.50/mes |
| Medio            | 1 GB   | 1    | 40 GB | 2 TB          | \$5/mes    |
| Alto rendimiento | 4 GB   | 2    | 80 GB | 4 TB          | \$20/mes   |

### 🔸 **Amazon EC2** (Pago por uso)

Ejemplo con instancia t3.micro (en free tier o uso bajo):

* \$0.0104/hora = \~\$7.50/mes si está encendida 24/7
* Sin contar EBS (almacenamiento), tráfico, IP elástica, etc.

### ✅ ¿Cuál te conviene?

| Necesidad                                               | Recomendación            |
| ------------------------------------------------------- | ------------------------ |
| Quiero lanzar un sitio rápido y barato                  | **Lightsail**            |
| Quiero tener control total y escalar con microservicios | **EC2**                  |
| No quiero pensar en precios variables                   | **Lightsail**            |
| Necesito alta disponibilidad y balanceo de carga        | **EC2** con Auto Scaling |
| Quiero aprender a usar AWS de forma profesional         | **EC2**                  |

### Resumen

Esto es lo que te ofrece Lightsail:

- El costo de los CPUs depende del número que elijas.
- Tienes almacenamiento SSD.
- Te ofrece Networking y transferencia de datos.
- Incluye manejo de DNS.
- Tienes una IP estática asignada a ti.
- Tienes acceso a otros servicios de **AWS**

En una comparativa de costos, el plan más económico de Lightsail ofrece por $3.50 1 TB de transferencia mientras que la misma capacidad en **EC2** puede salir por más de $90. Sigue aprendiendo con [cursos de AWS](https://platzi.com/ruta/aws/ "cursos de AWS").

## Creación y Configuración de un VPS con Ubuntu

¡Perfecto! A continuación te presento una **guía paso a paso para crear y configurar un VPS con Ubuntu en AWS**, usando **Amazon Lightsail**, que es ideal para crear un VPS de manera sencilla y económica.

### ✅ **Guía: Crear y Configurar un VPS con Ubuntu en Amazon Lightsail**

### 🧩 **Paso 1: Accede a AWS y abre Lightsail**

1. Inicia sesión en tu cuenta de AWS: [https://aws.amazon.com/](https://aws.amazon.com/)
2. En el buscador de servicios escribe **Lightsail** y selecciónalo.
3. Haz clic en **“Create Instance” (Crear instancia)**.

### 🛠️ **Paso 2: Configura la instancia VPS**

1. **Ubicación del servidor:** Selecciona la región donde quieres desplegar tu VPS.
2. **Sistema Operativo:**

   * Selecciona **Linux/Unix**
   * Escoge **Ubuntu 22.04 LTS** o la versión que necesites.
3. **Blueprint (imagen base):**

   * Escoge **OS Only** → **Ubuntu**.
4. **Plan de instancia:**

   * Elige un plan, por ejemplo:

     * 512MB RAM, 1 vCPU, 20 GB SSD → \$3.50/mes
     * Puedes cambiarlo después si necesitas más recursos.
5. **Nombre de la instancia:** Asigna un nombre como `mi-vps-ubuntu`.

### 🔐 **Paso 3: Accede por SSH**

1. Una vez creada, entra a tu instancia desde Lightsail:

   * Haz clic en tu VPS → Clic en **“Connect using SSH”**.
   * Se abrirá una terminal en el navegador.
2. Opcional: También puedes conectar usando un cliente SSH como PuTTY o Terminal de Linux/macOS:

   ```bash
   ssh -i ~/.ssh/LightsailDefaultPrivateKey.pem ubuntu@TU_IP_PUBLICA
   ```

### ⚙️ **Paso 4: Configura tu sistema Ubuntu**

1. Actualiza los paquetes:

   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
2. Instala paquetes básicos:

   ```bash
   sudo apt install git curl unzip build-essential -y
   ```

### 🌐 **Paso 5: Instala un servidor web (opcional)**

Por ejemplo, para instalar Apache:

```bash
sudo apt install apache2 -y
sudo systemctl start apache2
sudo systemctl enable apache2
```

Ahora abre tu navegador y visita tu IP pública. Verás la página de bienvenida de Apache.

### 🧱 **Paso 6: Configura un firewall (opcional)**

Puedes permitir solo tráfico esencial:

```bash
sudo ufw allow OpenSSH
sudo ufw allow http
sudo ufw allow https
sudo ufw enable
```

### 📎 **Paso 7: Asocia una IP estática (recomendado)**

1. En Lightsail → pestaña **Networking**.
2. Asigna una **Static IP** a tu instancia para que no cambie.

### 🎯 ¿Y luego qué?

Puedes ahora:

* Instalar tu aplicación web (Node.js, Python, Django, PHP, etc.)
* Montar un WordPress
* Configurar un dominio con DNS
* Subir tu código con Git o FTP

### Resumen

Esto es lo que te ofrece Lightsail:

- El costo de los CPUs depende del número que elijas.
- Tienes almacenamiento SSD.
- Te ofrece Networking y transferencia de datos.
- Incluye manejo de DNS.
- Tienes una IP estática asignada a ti.
- Tienes acceso a otros servicios de **AWS**

En una comparativa de costos, el plan más económico de Lightsail ofrece por $3.50 1 TB de transferencia mientras que la misma capacidad en **EC2** puede salir por más de $90. Sigue aprendiendo con [cursos de AWS](https://platzi.com/ruta/aws/ "cursos de AWS").

## Instalación de Proyecto en Instancia Lysel con Apache y Git

¡Perfecto! Aquí tienes una **guía paso a paso para instalar un proyecto web en una instancia de Amazon Lightsail con Ubuntu, Apache y Git**.

### 🛠️ **Requisitos previos**

* Tener una instancia de **Amazon Lightsail con Ubuntu** creada.
* Acceso SSH a la instancia (puedes usar el navegador o tu terminal).
* Tener un repositorio de Git con tu proyecto web.

### ✅ **Paso 1: Conectarse a la instancia**

Conéctate vía SSH desde la consola de Lightsail:

1. Ve al panel de Lightsail.
2. Haz clic en tu instancia.
3. Haz clic en **"Connect using SSH"**.

### 🧱 **Paso 2: Instalar Apache y Git**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install apache2 git -y
```

Verifica que Apache esté funcionando:

```bash
sudo systemctl status apache2
```

Si no está activo, arráncalo:

```bash
sudo systemctl start apache2
sudo systemctl enable apache2
```

Abre tu navegador y visita la **IP pública de tu instancia**, deberías ver la página de bienvenida de Apache.

### 📁 **Paso 3: Clonar tu proyecto desde Git**

Ve al directorio raíz de Apache (usualmente `/var/www/html`), limpia el contenido, y clona tu repo:

```bash
cd /var/www/html
sudo rm -rf *
sudo git clone https://github.com/tu_usuario/tu_repo.git .
```

> 🔁 Reemplaza `https://github.com/tu_usuario/tu_repo.git` por tu URL real.

### 🔒 **Paso 4: Asignar permisos**

```bash
sudo chown -R www-data:www-data /var/www/html
sudo chmod -R 755 /var/www/html
```

### 🌐 **Paso 5: Verifica tu sitio web**

Abre tu navegador y accede a la IP pública de tu instancia:

```
http://TU_IP_PUBLICA
```

Tu sitio debería mostrarse.

### 🔧 **Opcional: Habilitar un dominio personalizado**

1. Apunta tu dominio a la IP estática de Lightsail (configura DNS).
2. Configura un VirtualHost en Apache para usar el dominio.

### Resumen

Instalar un proyecto en **Lightsail** es muy parecido al procedimiento que se realiza en **EC2**.

Cosas a tener en cuenta al momento de instalar tu proyecto con **Lightsail**:

- Si estás en Windows, deberás usar un software como MobaXterm para tener una terminal que se conecte por SSH.
- Recuerda hacerte administrador con el comando “sudo su”
- Recuerda hacer update con el comando “apt-get update” porque es una instancia nueva y no tiene en memoria caché las direcciones desde dónde debe tomar el software. Si el proyecto se instala sin hacer esto, fallará.
- El comando para instalar el software es “sudo apt install apache2 git libapache2-mod-php -y”.
- La URL del proyecto es “[https://github.com/mauropm/quotes-generator](https://github.com/mauropm/quotes-generator%E2%80%9D "https://github.com/mauropm/quotes-generator")”.

Configurar todo lo que esté en la red de Amazon es súper veloz, dándole más ventajas a la instalación de proyectos en **Lightsail**.

## Instalación de bases de datos en Amazon LightSail

Aquí tienes una guía clara para la **instalación y configuración de bases de datos en Amazon Lightsail**, incluyendo opciones integradas y la configuración manual si prefieres instalar el motor en tu propia instancia.

### ✅ **Opción 1: Usar el servicio administrado de bases de datos de Lightsail (recomendado)**

Amazon Lightsail ofrece instancias de base de datos preconfiguradas y administradas (MySQL, PostgreSQL).

### 🔧 Paso a paso:

1. Ve a [https://lightsail.aws.amazon.com](https://lightsail.aws.amazon.com).
2. Haz clic en **“Bases de datos”** > **“Crear base de datos”**.
3. Elige el motor: **MySQL** o **PostgreSQL**.
4. Selecciona la versión, zona, plan y nombre.
5. Espera a que el estado diga “Disponible”.

### 🔐 Datos que obtendrás:

* **Host** (endpoint).
* **Puerto** (por defecto 3306 MySQL, 5432 PostgreSQL).
* **Usuario y contraseña principal**.

### 🔌 Conexión desde tu instancia:

Ejemplo para MySQL:

```bash
sudo apt install mysql-client -y
mysql -h TU_ENDPOINT -u usuario -p
```

### ✅ **Opción 2: Instalar manualmente una base de datos en una instancia de Lightsail (más flexible)**

Puedes instalar tú mismo **MySQL**, **PostgreSQL** o **MariaDB** en una instancia Ubuntu de Lightsail.

### 🔧 Paso a paso para instalar **MySQL Server** en Ubuntu:

```bash
sudo apt update
sudo apt install mysql-server -y
sudo systemctl enable mysql
sudo systemctl start mysql
```

### 🔐 Configurar acceso remoto (opcional y con precaución):

1. Edita el archivo:

```bash
sudo nano /etc/mysql/mysql.conf.d/mysqld.cnf
```

2. Cambia:

```
bind-address = 127.0.0.1
```

por:

```
bind-address = 0.0.0.0
```

3. Luego, crea un usuario remoto:

```sql
sudo mysql -u root

CREATE USER 'admin'@'%' IDENTIFIED BY 'tu_clave_segura';
GRANT ALL PRIVILEGES ON *.* TO 'admin'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;
```

4. Reinicia MySQL:

```bash
sudo systemctl restart mysql
```

5. Asegúrate de abrir el puerto 3306 en el **firewall de Lightsail**.

### 🔒 Seguridad:

* No uses el usuario root para conexiones remotas.
* Limita IPs que puedan conectarse a tu base de datos.
* Usa contraseñas seguras y actualiza regularmente.

### 🚀 ¿Qué opción deberías elegir?

* **Servicio administrado**: fácil de usar, backups automáticos, ideal para producción.
* **Instancia propia**: mayor control, útil para pruebas o configuraciones personalizadas.

### Resumen

Las bases de datos en **Lightsail** también tienen un costo fijo con la disponibilidad que ofrece Amazon.

Cosas a tener en cuenta al momento de crear tu base de datos:

- **Lightsail** nos ofrece varias versiones de MySQL; si es un proyecto nuevo es recomendado utilizar la más actual. Si es una migración deberemos elegir la versión más cercana a nuestra base de datos existente.
- **Lightsail** nos propone un password seguro, es recomendable usarlo.
- Puedes configurar tu base de datos de dos maneras:
 **Estándar**: Un servidor con una conexión desde afuera.
 **HA**: Alta disponibilidad, donde tienes dos servidores o más con un load balancer.2

**Lecturas recomendadas**

[Databases in Amazon Lightsail | Lightsail Documentation](https://lightsail.aws.amazon.com/ls/docs/en_us/articles/amazon-lightsail-databases)

## Configuración de Servidores y Software en Línea

La **configuración de servidores y software en línea** se refiere al proceso de **preparar un servidor en la nube (como AWS, Azure, DigitalOcean, Lightsail, etc.)** e instalar y configurar el software necesario para que esté listo para alojar aplicaciones web, servicios o bases de datos. Aquí tienes una **guía paso a paso** general para realizar esta configuración:

---

## 🔧 1. Elección del Proveedor y del Servidor

### Opciones comunes:

* **AWS EC2 / Lightsail**
* **DigitalOcean Droplets**
* **Azure Virtual Machines**
* **Google Cloud Compute Engine**

### Recomendación para principiantes:

✅ **Amazon Lightsail** – Interfaz sencilla y precios fijos.

---

## 🖥️ 2. Creación de la Instancia del Servidor

1. Selecciona el sistema operativo (recomendado: **Ubuntu LTS**).
2. Define el tamaño de la instancia (RAM, CPU).
3. Configura el almacenamiento.
4. Genera o carga una **llave SSH** para acceso seguro.
5. Lanza el servidor y anota su IP pública.

---

## 🔐 3. Acceso al Servidor (con SSH)

Desde terminal (Linux/macOS) o PuTTY (Windows):

```bash
ssh -i /ruta/tu_llave.pem ubuntu@IP_DEL_SERVIDOR
```

---

## 🛠️ 4. Instalación del Software Básico

### Actualización del sistema:

```bash
sudo apt update && sudo apt upgrade -y
```

### Instalación de software común:

* **Apache / Nginx** (servidor web)
* **MySQL / PostgreSQL / MariaDB** (base de datos)
* **PHP / Node.js / Python** (lenguaje backend)
* **Git** (control de versiones)

Ejemplo:

```bash
sudo apt install apache2 php libapache2-mod-php mysql-server git -y
```

---

## 🌍 5. Configuración del Servidor Web

Para Apache:

```bash
sudo systemctl enable apache2
sudo systemctl start apache2
```

Coloca tu sitio en `/var/www/html`.

---

## 🔐 6. Configuración de Seguridad

* Instalar y configurar **UFW (firewall)**:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Apache Full'
sudo ufw enable
```

* Cambiar puertos por defecto si es necesario.
* Instalar **certificados SSL con Let's Encrypt**:

```bash
sudo apt install certbot python3-certbot-apache -y
sudo certbot --apache
```

---

## 💾 7. Despliegue del Proyecto

1. Clona desde GitHub:

```bash
git clone https://github.com/tu_usuario/tu_proyecto.git
```

2. Configura permisos y rutas.
3. Reinicia servicios si es necesario:

```bash
sudo systemctl restart apache2
```

---

## 📈 8. Monitoreo y Mantenimiento

* Instala **fail2ban** para proteger contra ataques de fuerza bruta.
* Usa herramientas como **CloudWatch**, **Grafana** o **Netdata** para monitoreo.
* Programa backups automáticos (con `cron` o herramientas externas).

### Resumen

### ¿Cuál es el reto propuesto en la clase?

Iniciar con el pie derecho en el ámbito de la computación en la nube puede ser un desafío apasionante y lleno de aprendizaje. En la clase que nos ocupa, el objetivo es sencillo pero significativo: te enfrentarás a la tarea de crear una instancia desde cero y configurarla de manera adecuada. Esto te permitirá comprender la estructura y complejidad detrás de la puesta en marcha de un sistema en línea. No te preocupes, con práctica y dedicación, podrás dominar esta habilidad esencial.

### ¿Cómo se configura e instala un servidor?

La configuración e instalación de un servidor requiere seguir una serie de pasos que varían según el sistema operativo y las herramientas seleccionadas. Aquí hay una guía básica que puedes seguir:

1. **Selecciona tu proveedor de nube**: Puedes optar por opciones populares como AWS, Azure o Google Cloud.
2. **Crea una instancia**: Elige la imagen del sistema operativo que prefieras para tu servidor. Algunas opciones comunes incluyen Ubuntu, CentOS y Windows Server.
3. **Configura el sistema operativo**:
 - **Actualiza**: Asegúrate de actualizar el sistema operativo para garantizar que todas las aplicaciones y dependencias estén al día.
 - **Configura el nombre del host y la red**: Establece un hostname significativo y configura las interfaces de red necesarias.
4. **Instala el software necesario**: Dependiendo de tus necesidades, podría ser un servidor web como Apache o Nginx, una base de datos como MySQL o PostgreSQL, entre otros.
5. **Configura el software instalado**: Asegúrate de que el software esté correctamente configurado para cumplir con tus requisitos específicos.
6. **Asegura tu servidor**: Implementa medidas de seguridad como firewalls, SSH seguro y actualizaciones regulares del sistema.

### ¿Qué debes tener en cuenta al trabajar con bases de datos?

Trabajar con bases de datos en un entorno de servidor requiere atención a varios aspectos que asegurarán su correcto funcionamiento:

1. **Elección del motor de la base de datos**: Define si necesitas uno relacional como MySQL o PostgreSQL, o uno NoSQL como MongoDB.
2. **Configuración inicial**: Realiza los ajustes necesarios para que la base de datos opere eficientemente, como la asignación de memoria y el número máximo de conexiones.
3. **Seguridad**: Protege los datos realizando copias de seguridad de manera regular y configurando roles de usuario con permisos específicos.
4. **Optimización de consultas**: Asegúrate de que las consultas sean eficientes para minimizar el consumo de recursos y mejorar el rendimiento.
5. **Monitoreo**: Implementa herramientas de monitoreo para detectar cualquier irregularidad en el rendimiento o operación.

Dado que los sistemas de bases de datos son críticos para muchas aplicaciones, la atención a estos detalles garantizará un entorno más seguro y eficiente.

### ¿Por qué es importante practicar la creación de instancias?

Practicar la creación y configuración de instancias en la nube no solamente desarrolla habilidades técnicas, sino que también incrementa tu confianza al enfrentarte a tareas complejas. Esta práctica fomenta una comprensión más profunda de los servicios en la nube, lo cual es invaluable en el campo de la tecnología actual.

1. **Experiencia práctica**: Aprender haciendo es una de las formas más efectivas de adquirir nuevas habilidades.
2. **Adaptación a escenarios del mundo real**: Familiarizarte con entornos de servidor y sus retos te prepara para enfrentar problemas reales.
3. **Mejora continua**: Con cada instancia que configuras, mejoras tu habilidad para optimizar y asegurar servidores eficientemente.

La tecnología no se detiene, y como profesional, es crucial que te mantengas al día con las últimas innovaciones. El futuro de la computación está en la nube; adentrarte en este mundo te abrirá puertas hacia un sinfín de oportunidades. ¡No te rindas y sigue aprendiendo!

## Contenedores en Amazon: Registro, Creación y Kubernetes con EKS

Aquí tienes una **guía paso a paso sobre contenedores en Amazon AWS**, cubriendo **registro, creación y despliegue con Kubernetes (EKS)**:

### 🧱 1. ¿Qué son los contenedores?

Los contenedores (como los creados con Docker) encapsulan una aplicación y sus dependencias en un entorno portable. En AWS puedes:

* **Crear imágenes de contenedores (Docker)**
* **Registrar y almacenar imágenes en ECR (Elastic Container Registry)**
* **Desplegar y orquestar contenedores con EKS (Elastic Kubernetes Service)**

### 📦 2. Registro de Contenedores con **Amazon ECR**

### a. Crear un repositorio en ECR:

1. Entra a la consola de AWS → Servicio **ECR**.
2. Clic en “**Create Repository**”.
3. Define el nombre (ej: `mi-aplicacion`) y crea el repositorio.

### b. Conecta tu CLI a ECR:

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ID>.dkr.ecr.us-east-1.amazonaws.com
```

### c. Construye y sube tu imagen:

```bash
# Construcción de la imagen Docker
docker build -t mi-aplicacion .

# Etiquetado con la URL del repositorio
docker tag mi-aplicacion:latest <ID>.dkr.ecr.us-east-1.amazonaws.com/mi-aplicacion:latest

# Push a ECR
docker push <ID>.dkr.ecr.us-east-1.amazonaws.com/mi-aplicacion:latest
```

### ⚙️ 3. Despliegue con **Amazon EKS (Elastic Kubernetes Service)**

### a. Crear un clúster EKS (puedes usar consola o CLI):

Con `eksctl` (fácil):

```bash
eksctl create cluster \
  --name mi-cluster \
  --version 1.29 \
  --region us-east-1 \
  --nodegroup-name grupo-nodos \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 3 \
  --managed
```

> Necesitas tener `eksctl`, `kubectl`, y `awscli` configurados.

### b. Configura `kubectl`:

```bash
aws eks --region us-east-1 update-kubeconfig --name mi-cluster
```

### c. Despliega tu contenedor con Kubernetes:

1. Crea un archivo `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mi-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mi-app
  template:
    metadata:
      labels:
        app: mi-app
    spec:
      containers:
      - name: mi-app
        image: <ID>.dkr.ecr.us-east-1.amazonaws.com/mi-aplicacion:latest
        ports:
        - containerPort: 80
```

2. Aplica el despliegue:

```bash
kubectl apply -f deployment.yaml
```

3. Exponer el servicio:

```bash
kubectl expose deployment mi-app --type=LoadBalancer --port=80
```

### d. Verifica:

```bash
kubectl get all
```

### 🧩 Extra: Herramientas y Tips

* Usa **IAM roles for service accounts** para limitar el acceso de pods a servicios AWS.
* Configura **Auto Scaling** para nodos con `Cluster Autoscaler`.
* Monitorea con **CloudWatch Container Insights** o Prometheus + Grafana.

### Resumen

**ECR** es el servicio que te permite registrar los contenedores a través de Dockerfiles en Amazon.
Aunque existe **ECR**, no aparece como producto. Es necesario entrar a **ECS** y ya desde ahí encontramos las opciones para entrar al ECR.
Importante antes de registrar contenedores: Tener instalado el **AWS CLI** y **Docker**, adicionalmente es importante tener instalado Git.

**Lecturas recomendadas**

[ubuntu - Got permission denied while trying to connect to the Docker daemon socket while executing docker stop - Stack Overflow](https://stackoverflow.com/questions/46759268/got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket-while "ubuntu - Got permission denied while trying to connect to the Docker daemon socket while executing docker stop - Stack Overflow")

[How To Install and Use Docker on Ubuntu 18.04 | DigitalOcean](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04 "How To Install and Use Docker on Ubuntu 18.04 | DigitalOcean")

## Ejecución de Contenedores Docker con Amazon ECS

Aquí tienes una **guía paso a paso para ejecutar contenedores Docker en Amazon ECS (Elastic Container Service)**, que es la alternativa a EKS cuando quieres usar contenedores sin preocuparte por la administración directa de Kubernetes.

### 🚢 ¿Qué es Amazon ECS?

Amazon ECS (Elastic Container Service) es un **servicio de orquestación de contenedores totalmente administrado**. Permite ejecutar y escalar contenedores Docker fácilmente usando:

* **Fargate** (sin administrar servidores)
* **EC2** (tú manejas los servidores)

### 🧭 Flujo básico para ejecutar contenedores en ECS

1. ✅ Crear un **repositorio en Amazon ECR**
2. 🛠️ Crear una **imagen Docker** y subirla
3. 📦 Definir una **tarea ECS (task definition)**
4. 🚀 Crear un **servicio ECS** para ejecutar y mantener los contenedores
5. 🌍 Opcional: Exponer tu aplicación al público con un Load Balancer

### 🧱 Paso a Paso

### 1. Crear el repositorio en Amazon ECR

```bash
aws ecr create-repository --repository-name mi-aplicacion
```

> Guarda la URI que retorna, por ejemplo:
> `<aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/mi-aplicacion`

### 2. Construir y subir la imagen Docker

```bash
# Login a ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <tu-id>.dkr.ecr.us-east-1.amazonaws.com

# Build y tag
docker build -t mi-aplicacion .
docker tag mi-aplicacion:latest <tu-id>.dkr.ecr.us-east-1.amazonaws.com/mi-aplicacion:latest

# Push
docker push <tu-id>.dkr.ecr.us-east-1.amazonaws.com/mi-aplicacion:latest
```

### 3. Crear una definición de tarea (Task Definition)

Desde la consola AWS o con JSON:

```json
{
  "family": "mi-aplicacion-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "mi-aplicacion",
      "image": "<tu-repo>.dkr.ecr.us-east-1.amazonaws.com/mi-aplicacion:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 80,
          "hostPort": 80
        }
      ]
    }
  ]
}
```

Sube esto usando:

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### 4. Crear un clúster ECS

```bash
aws ecs create-cluster --cluster-name mi-cluster
```

### 5. Crear un servicio para ejecutar la tarea

```bash
aws ecs create-service \
  --cluster mi-cluster \
  --service-name mi-servicio \
  --task-definition mi-aplicacion-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}"
```

> ⚠️ Asegúrate de tener una VPC, subred y grupo de seguridad configurados.

### 6. Exponer al público (opcional)

* Usa **Application Load Balancer (ALB)** en la red de tu ECS.
* Configura una **regla de escucha (listener rule)** para redirigir al servicio ECS.

### 🧩 Tips Adicionales

* Usa **CloudWatch Logs** para ver el output de tus contenedores.
* Usa **autoscaling de servicios ECS** para responder a picos de tráfico.
* Puedes gestionar ECS con la consola, CLI, o infra como código (CDK/Terraform).

### Resumen

**ECS** es toda la infraestructura que te permite correr contenedores de Docker directo en AWS.

- Su ventaja es que no debes poner una máquina con Docker donde encima corran los contenedores. Amazon da la infraestructura pre-hecha y nosotros solo elegimos capacidades.
- Únicamente se paga por la capacidad solicitada (cCPU, memoria, transferencia de datos).
- Puedes escalar tu instancia basada en contenedor de manera manual.

Usos clásicos de **ECS**:

- Microservicios.
- Migración de aplicaciones Legacy al Cloud.

**Lecturas recomendadas**

[Amazon ECR | Amazon Web Services](https://aws.amazon.com/es/ecr/ "Amazon ECR | Amazon Web Services")

[Â¿QuÃ© es Amazon Elastic Container Service? - Amazon Elastic Container Service](https://docs.aws.amazon.com/es_es/AmazonECS/latest/developerguide/Welcome.html "Â¿QuÃ© es Amazon Elastic Container Service? - Amazon Elastic Container Service")

## Implementación de Contenedores en AWS Fargate paso a paso

Aquí tienes una **guía paso a paso para implementar contenedores en AWS Fargate** utilizando ECS (Elastic Container Service), sin necesidad de gestionar servidores. AWS Fargate permite ejecutar contenedores serverless, por lo que es ideal para aplicaciones containerizadas.

### 🔧 **Requisitos Previos**

* Tener una cuenta de AWS
* Tener instalado:

  * AWS CLI
  * Docker
  * `ecs-cli` o utilizar AWS Management Console
* Una imagen de contenedor disponible (puede estar en [Docker Hub](https://hub.docker.com) o en [Amazon ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/what-is-ecr.html))

### 🪜 **Pasos para Implementar Contenedores en AWS Fargate**

### 1. **Crear o subir una imagen de Docker**

```bash
# Crear imagen local
docker build -t mi-app .

# (Opcional) Subir a ECR
aws ecr create-repository --repository-name mi-app
aws ecr get-login-password | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com
docker tag mi-app:latest <aws_account_id>.dkr.ecr.<region>.amazonaws.com/mi-app:latest
docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/mi-app:latest
```

### 2. **Crear una VPC (si no tienes una)**

Puedes usar la VPC por defecto o crear una nueva con subredes públicas y privadas. Esto se hace desde la consola de AWS: **VPC > Launch VPC Wizard**.

### 3. **Crear un Cluster ECS para Fargate**

Desde la consola:

* Ve a **ECS > Clusters**
* Click en **Create Cluster**
* Elige **Networking only (Fargate)**
* Dale un nombre y crea el cluster

### 4. **Definir una Task Definition (Definición de Tarea)**

Desde la consola:

* Ve a **ECS > Task Definitions**
* Click en **Create new Task Definition**
* Elige **FARGATE**
* Configura:

  * Nombre de la tarea
  * Rol de ejecución (`ecsTaskExecutionRole`)
  * Container:

    * Nombre: `mi-app-container`
    * Image: URI del contenedor en ECR o Docker Hub
    * Memory: por ejemplo 512 MiB
    * CPU: por ejemplo 256
    * Port mappings: por ejemplo `80:80`

### 5. **Crear un Servicio ECS**

Desde la consola:

* Ir a tu Cluster
* Click en **Create** > **Create Service**
* Tipo de lanzamiento: **FARGATE**
* Task Definition: la creada en el paso anterior
* Número de tareas: mínimo 1
* Cluster: el que creaste
* VPC y subredes: selecciona las de tu VPC
* Seguridad: crea o selecciona un security group con el puerto necesario abierto (por ejemplo, el 80)

### 6. **Configurar Load Balancer (opcional)**

* Si tu app necesita ser pública o balanceada, puedes:

  * Crear un **Application Load Balancer**
  * Asociarlo a tu servicio
  * Configurar el target group y listener para que redireccione al puerto del contenedor

### 7. **Verifica el despliegue**

* Entra al ECS Cluster > Service > Tareas
* Debe estar en estado **RUNNING**
* Si tienes Load Balancer, ve a la consola de EC2 > Load Balancers y revisa el **DNS name**

### 8. **Accede a tu aplicación**

* Si configuraste un Load Balancer, usa el URL público:

  ```
  http://<load-balancer-dns-name>
  ```
* Si no, puedes usar una IP pública de la subred (si la tarea está en una subred pública)

### 🛡️ Roles necesarios

Asegúrate de que tu rol `ecsTaskExecutionRole` tenga al menos estas políticas:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

### ✅ Ejemplo de archivo `task-definition.json`

```json
{
  "family": "mi-app-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "mi-app",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/mi-app:latest",
      "portMappings": [
        {
          "containerPort": 80,
          "hostPort": 80,
          "protocol": "tcp"
        }
      ]
    }
  ]
}
```

Puedes registrar la tarea con:

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### Resumen

Cosas a tener en cuenta al momento de correr un contenedor:

- **Networking only** está basado en un producto llamado AWS Fargate que nos da la infraestructura de Docker sin tener que preocuparnos por las máquinas base y es el que usaremos en este proyecto.
- Es necesario crear una tarea relacionada con la imagen de Docker que creamos anteriormente.

## Configuración de Docker en EC2 para Crear Imágenes de Contenedores

**Introducción**

Para poder ejecutar comandos como “docker build” necesitamos configurar nuestro ambiente de docker en una instancia EC2 pequeña.

Configuración de Docker
- Crea una instancia de EC2 con Ubuntu.
- Selecciona una instancia de tamaño mínimo (nano, por ejemplo, si tienes una cuenta AWS de mas de un año. - En caso contrario, la t2.micro es la gratuita en tu primer año de servicio (recuerda, únicamente por un año).
- Una vez que este en estado “Running” conectate a ella.
- Teclea: a) sudo su b) apt-get update
- Una vez que termine, corre, como usuario root: a) snap install docker -y b) apt-get install git -y
- Después de esto, ya podrás hacer: a) git clone https://github.com/mauropm/quotes-generator.git b) cd quotes-generator c) dock build

Con esto, ya podrás hacer imágenes de contenedores y siguiendo las instrucciones de la clase, podrás enviarlo a ECR (El registro de contenedores de AWS).

## Implementación de Kubernetes en Amazon EKS

A continuación se presenta una **guía paso a paso para implementar Kubernetes en Amazon EKS** (Elastic Kubernetes Service), la solución administrada de Kubernetes de AWS. Esta guía abarca desde la configuración de los requisitos previos hasta la verificación de la implementación y el despliegue de aplicaciones en el clúster.

### 🔧 Requisitos Previos

Antes de comenzar, asegúrate de contar con:

* **Cuenta de AWS:** Con permisos administrativos o los permisos necesarios para crear recursos como VPCs, roles IAM, clústeres EKS y grupos de nodos.
* **AWS CLI:** Configurada y autenticada con tus credenciales.
* **eksctl:** Una herramienta de línea de comandos que simplifica la creación y gestión de clústeres EKS. Se recomienda [eksctl](https://eksctl.io/), ya que automatiza muchos de los pasos necesarios.
* **kubectl:** La herramienta de línea de comandos para interactuar con Kubernetes.
* **IAM Roles y Políticas:** Permisos adecuados para crear y asociar roles a los recursos EKS.
* (Opcional) **Terraform o CloudFormation:** Si prefieres gestionar la infraestructura como código.

### 🪜 Pasos para Implementar Kubernetes en Amazon EKS

### 1. **Instalar y Configurar Herramientas**

#### a. Instalar AWS CLI

Verifica que tienes instalada la última versión de AWS CLI:

```bash
aws --version
```

Si no la tienes, sigue la [guía oficial de instalación](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

#### b. Instalar eksctl

Instala eksctl siguiendo las instrucciones de su [documentación oficial](https://eksctl.io/introduction/#installation). Por ejemplo, en sistemas basados en Unix:

```bash
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
```

#### c. Instalar kubectl

Asegúrate de tener `kubectl` instalado y actualizado. Puedes instalarlo con:

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
```

### 2. **Crear el Clúster EKS**

Utilizando **eksctl**, crear un clúster es muy sencillo. Puedes personalizar opciones como la región, versión de Kubernetes, y la cantidad/tamaño de nodos en el grupo de trabajadores.

#### Ejemplo de comando para crear un clúster:

```bash
eksctl create cluster \
  --name mi-cluster-eks \
  --version 1.24 \
  --region us-east-1 \
  --nodegroup-name workers \
  --node-type t3.medium \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed
```

Este comando:

* Crea un clúster llamado `mi-cluster-eks` en la región `us-east-1`.
* Utiliza la versión especificada de Kubernetes.
* Configura un grupo de nodos gestionados con instancias `t3.medium` y escalabilidad entre 1 y 4 nodos.

**Nota:** El proceso puede tardar unos 10–20 minutos. Durante la creación, `eksctl` se encargará de crear la VPC, subredes, roles IAM y demás recursos necesarios.

### 3. **Configurar `kubectl` para Conectarse al Clúster**

Una vez completada la creación del clúster, `eksctl` actualiza automáticamente el archivo kubeconfig. Verifica la conexión con:

```bash
kubectl get svc
```

Deberías ver la lista de servicios del namespace `default` del clúster.

### 4. **Verificar el Estado del Clúster y Grupos de Nodos**

Para ver el estado de los nodos, ejecuta:

```bash
kubectl get nodes
```

La salida deberá listar los nodos que se encuentran en estado `Ready`.

Si deseas más detalles sobre los componentes del clúster:

```bash
kubectl get pods -A
```

Esto muestra los pods que se están ejecutando en todos los namespaces, lo cual es útil para revisar los add-ons que EKS instala (por ejemplo, kube-system).

### 5. **Desplegar Aplicaciones en el Clúster**

Con el clúster operativo, puedes desplegar aplicaciones utilizando archivos YAML de Kubernetes.

#### Ejemplo de despliegue de una aplicación NGINX:

1. **Crear un archivo YAML llamado `nginx-deployment.yaml`:**

   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: nginx-deployment
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: nginx
     template:
       metadata:
         labels:
           app: nginx
       spec:
         containers:
         - name: nginx
           image: nginx:latest
           ports:
           - containerPort: 80
   ```

2. **Aplicar la definición:**

   ```bash
   kubectl apply -f nginx-deployment.yaml
   ```

3. **Verificar el despliegue:**

   ```bash
   kubectl get deployments
   kubectl get pods
   ```

#### Exponer la aplicación (opcional)

Para exponer el despliegue a través de un LoadBalancer de AWS:

1. **Crear un servicio tipo LoadBalancer:**

   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: nginx-service
   spec:
     type: LoadBalancer
     selector:
       app: nginx
     ports:
       - protocol: TCP
         port: 80
         targetPort: 80
   ```

2. **Aplicar el servicio:**

   ```bash
   kubectl apply -f nginx-service.yaml
   ```

3. **Obtener la IP o DNS del LoadBalancer:**

   ```bash
   kubectl get svc nginx-service
   ```

Una vez asignada la dirección, podrás acceder a NGINX desde el navegador.

### 6. **Administrar y Escalar el Clúster**

#### Escalado de nodos:

Si necesitas ajustar el tamaño del grupo de nodos, puedes usar **eksctl** nuevamente. Por ejemplo, para escalar a 5 nodos:

```bash
eksctl scale nodegroup --cluster mi-cluster-eks --name workers --nodes 5 --region us-east-1
```

#### Escalado de aplicaciones:

Modifica el número de réplicas en el despliegue y aplica los cambios:

```bash
kubectl scale deployment/nginx-deployment --replicas=4
```

#### Actualizaciones:

* **Actualizar `kubectl`, `eksctl` y versiones de Kubernetes:** Revisa periódicamente las versiones y las notas de lanzamiento de EKS para mantenerte actualizado.
* **Monitoreo y Logging:** EKS se integra con servicios de AWS como CloudWatch para logs y monitoreo, lo cual es útil para la administración del clúster en producción.

### 7. **Limpieza de Recursos (Opcional)**

Si deseas eliminar el clúster para evitar costos innecesarios, usa:

```bash
eksctl delete cluster --name mi-cluster-eks --region us-east-1
```

Esto eliminará el clúster, los grupos de nodos y los recursos asociados creados por eksctl.

### Conclusión

La implementación de Kubernetes en Amazon EKS con la ayuda de **eksctl** y **kubectl** simplifica la creación y gestión del clúster. Al seguir estos pasos podrás:

* Configurar el entorno de EKS de forma automatizada.
* Administrar y escalar tanto el clúster como las aplicaciones desplegadas.
* Aprovechar la integración con otros servicios de AWS para monitoreo y seguridad.

### Resumen

- **EKS** es una implementación de Kubernetes en Amazon que no requiere que coordines nodos maestros y esclavos.
- Te permite crear un ambiente de workers de k8s en **AWS**.
- Podrás correr contenedores con el dashboard de Kubernetes o cualquier orquestador que quieras usar.

**EKS** va desde poner el nodo maestro de Kubernetes, poner los workers y ya podrás conectarte a la API para correr tareas.

**Lecturas recomendadas**

[Amazon EKS – Servicio de Kubernetes administrado](https://aws.amazon.com/es/eks/)

[Production-Grade Container Orchestration - Kubernetes](https://kubernetes.io/)

## Configuración de kops y creación de clúster Kubernetes en AWS

**Introducción**

kops es una herramienta que nos permite crear y administrar kubernetes (también conocido como k8s) en AWS (y otros clouds). En esta lectura pondremos las instrucciones para configurarlo localmente y crear un cluster de k8s en AWS.

Instrucciones
Como root, en alguna instancia EC2 pequeña o en su máquina local (estas instrucciones son para linux).

- sudo apt update
- sudo apt install -y awscli
- sudo snap install kubectl --classic
- curl -LO [https://github.com/kubernetes/kops/releases/download/1.7.0/kops-linux-amd64](https://github.com/kubernetes/kops/releases/download/1.7.0/kops-linux-amd64 "https://github.com/kubernetes/kops/releases/download/1.7.0/kops-linux-amd64")
- chmod +x kops-linux-amd64
- mv ./kops-linux-amd64 /usr/local/bin/kops
- Tienen que crear un usuario llamado kops en IAM.
- Entren en IAM, hagan un nuevo usuario.
- Configuren esto como acceso programatico.
- Apunten el Access Key ID y el password.
- Asígnenle el rol de AdministratorAccess (un rol preconfigurado en AWS IAM).
- Salvar.
- Regresen de la consola de AWS a tu consola / terminal, y continúen con lo siguiente:
- aws config
- aws iam create-group --group-name kops
- aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess --group-name kops
- aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonRoute53FullAccess --group-name kops
- aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess --group-name kops
- aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/IAMFullAccess --group-name kops
- aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonVPCFullAccess --group-name kops
- aws iam add-user-to-group --user-name kops --group-name kops
- aws s3api create-bucket --bucket s3kopstudominiocom --region us-west-2
- Antes de ejecutar el próximo comando, anexen lo siguiente a su archivo ~/.bashrc (al final): export AWS_ACCESS_KEY_ID=tuaccesskey export AWS_SECRET_ACCESS_KEY=tusecret export KOPS_STATE_STORE=s3://s3kopstudominiocom export KOPS_CLUSTER_NAME=kops-cluster-tudominio
- Sálvenlo. Cierren sesión con “exit” y vuelvan a entrar. Ahora si, ejecuta:
- kops create cluster --name=kops-cluster-tudominio --cloud=aws --zones=us-west-2a --state=s3kopstudominiocom
- Esta operación puede tardar 20 minutos.
- Cuando terminen, denle: kubectl apply -f [https://raw.githubusercontent.com/kubernetes/dashboard/master/src/deploy/recommended/kubernetes-dashboard.yaml](https://raw.githubusercontent.com/kubernetes/dashboard/master/src/deploy/recommended/kubernetes-dashboard.yaml "https://raw.githubusercontent.com/kubernetes/dashboard/master/src/deploy/recommended/kubernetes-dashboard.yaml")
- Con eso, se instalará el dashboard de k8s que vieron en el ejemplo.
- Loguearse con user admin, y el password se obtiene con: kops get secrets kube --type secret -oplaintext
- Cuando se conecten, seleccionen anunciarse por token, y el token lo obtienen ejecutando lo siguiente: kops get secrets admin --type secret -oplaintext
- Con eso, ya podrán dar click en “Create” y poder poner su imagen del contenedor en ECR.
- Cuando termine de hacer el deployment, encontrarán la url en la sección en el menú llamada “Services”.

**Nota:**

Si estas instrucciones las llevan a cabo en su máquina local, si tecleas kubectl proxy, tendrán el dashboard en la dirección: [https://localhost:8001](https://localhost:8001/ "https://localhost:8001") - Noten que usa https siempre, y que el certificado no es confiable, por lo que tendrán que autorizar a su browser para poder abrirlo. La url completa para el dashboard, utilizando kubectl proxy, es:

[http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy/#!/login](http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy/#!/login "http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy/#!/login")

**Conclusión:**

Esta actividad no es fácil. Kubernetes es un proyecto en construcción, por lo que está en constante cambio todo el tiempo, y evoluciona tan rápido que estas instrucciones podrían volverse obsoletas pronto, por lo que les pido que no desesperen, y que si hay alguna situación que no esté funcionando, pregunten en la sección de comentarios.

## Creación y Ejecución de Dockerfile para Aplicaciones

Aquí tienes una **guía completa y paso a paso** para la **creación y ejecución de un `Dockerfile`** para contenerizar una aplicación, con ejemplos prácticos.

### 🧱 ¿Qué es un Dockerfile?

Un `Dockerfile` es un archivo de texto con instrucciones para construir una imagen Docker personalizada. Describe paso a paso cómo se debe construir esa imagen: sistema base, dependencias, archivos de tu app, comandos, etc.

### 🔧 Requisitos Previos

* Tener instalado [Docker](https://docs.docker.com/get-docker/) en tu sistema.
* Código fuente de tu aplicación (por ejemplo, una app en Python, Node.js, etc.).
* Terminal o línea de comandos.

### 📁 Estructura Básica del Proyecto

Ejemplo con una app en Python:

```
mi_app/
├── app.py
├── requirements.txt
└── Dockerfile
```

### 📝 Paso 1: Crear el archivo `Dockerfile`

### Ejemplo para una app Python (`app.py`)

#### 📄 `Dockerfile`

```Dockerfile
# Imagen base oficial de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos locales al contenedor
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos de la app
COPY . .

# Puerto expuesto (opcional, solo para apps web)
EXPOSE 5000

# Comando para ejecutar la app
CMD ["python", "app.py"]
```

#### 📄 `requirements.txt`

```txt
flask
```

#### 📄 `app.py` (ejemplo simple con Flask)

```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hola desde Docker!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 🏗️ Paso 2: Construir la imagen Docker

Desde la terminal, ubícate en la carpeta donde está el Dockerfile y ejecuta:

```bash
docker build -t mi_app_python .
```

> Esto crea una imagen llamada `mi_app_python`.

### 🚀 Paso 3: Ejecutar el contenedor

```bash
docker run -d -p 5000:5000 --name contenedor_app mi_app_python
```

* `-d`: modo *detached* (en segundo plano).
* `-p 5000:5000`: mapea el puerto del contenedor al de tu máquina.
* `--name`: asigna un nombre al contenedor.

Luego accede desde el navegador a:
👉 [http://localhost:5000](http://localhost:5000)

### 🔍 Paso 4: Verificar que funciona

Verifica los logs del contenedor:

```bash
docker logs contenedor_app
```

Para entrar al contenedor en ejecución:

```bash
docker exec -it contenedor_app bash
```

### 🧹 Paso 5: Detener y eliminar

```bash
docker stop contenedor_app
docker rm contenedor_app
docker rmi mi_app_python
```

### 🎁 Otros ejemplos

### 🔷 Node.js

```Dockerfile
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

### ✅ Buenas Prácticas

* Usa imágenes oficiales y ligeras (`slim`, `alpine`).
* Evita copiar archivos innecesarios (usa `.dockerignore`).
* Usa capas eficientes: primero dependencias, luego código.
* Etiqueta tus imágenes (`mi_app:v1.0`).

### Resumen

#### ¿Qué es Docker y por qué es importante en el desarrollo de software?

Docker es una herramienta que ha revolucionado el mundo del desarrollo de software al facilitar la creación, despliegue y ejecución de aplicaciones en contenedores. Este enfoque permite que las aplicaciones se ejecuten de manera consistente en cualquier entorno, lo que resuelve el típico problema de “funciona en mi máquina”. Al encapsular una aplicación y todas sus dependencias en un contenedor, Docker garantiza que se pueda ejecutar sin modificaciones en cualquier máquina que tenga Docker instalado.

#### ¿Cómo iniciar con la creación de un Dockerfile?

La creación de un Dockerfile es el primer paso para contenerizar tu aplicación. Un Dockerfile es un archivo de texto que contiene los comandos necesarios para ensamblar una imagen de Docker. Aquí te mostramos un flujo básico para comenzar:

1. **Definir la imagen base**: Selecciona una imagen base que se adecue a las necesidades de tu aplicación. Por ejemplo, si estás trabajando con Node.js, podrías usar `node:14` como tu imagen base.

2. **Copiar el código fuente**: Incluye los archivos de tu aplicación al contenedor usando `COPY` o `ADD`.

3. **Instalar dependencias**: Si tu aplicación necesita bibliotecas externas, asegúrate de instalarlas, normalmente utilizando un sistema de gestión de paquetes como `npm` para `Node.js` o `pip` para Python.

4. **Especificar el comando de ejecución**: Define cuál será el comando que lanzará tu aplicación al ejecutarse en el contenedor. Usualmente, esto se hace con el comando `CMD` o `ENTRYPOINT`.

Aquí tienes un ejemplo básico:

```shell
# Usar la imagen base Node.js
FROM node:14

# Crear el directorio de la aplicación
WORKDIR /usr/src/app

# Copiar archivos necesarios
COPY package*.json ./

# Instalar dependencias
RUN npm install

# Copiar el código fuente
COPY . .

# Exponer el puerto en el que corre la aplicación
EXPOSE 8080

# Comando de ejecución de la aplicación
CMD ["node", "app.js"]
```

#### ¿Cómo publicar y correr tu contenedor en un register?

Una vez que hayas creado tu contenedor Docker, el siguiente paso es subirlo a un registro para manejarlo fácilmente y compartirlo con otros. Los pasos son los siguientes:

1. **Crear una cuenta en un Docker Registry**: Los registros más comunes son Docker Hub o un registry privado.

2. **Etiquetar tu imagen**: Usa d`ocker tag <image_name> <repository>/<image>:<tag>` para etiquetar tu imagen.

3. **Iniciar sesión**: Usa `docker login` para ingresar tus credenciales y conectarte con tu registro.

4. **Subir la imagen**: Con el comando `docker push <repository>/<image>:<tag>` envías tu imagen al registro.

5. **Ejecutar la imagen desde el registro**: En cualquier máquina con Docker, puedes tirar de la imagen usando `docker pull <repository>/<image>:<tag>`. Una vez descargada, la puedes correr con `docker run`.

#### ¿Cuáles son los beneficios de correr aplicaciones con Docker?

**Docker** ofrece una serie de ventajas significativas:

- **Portabilidad mejorada**: Los contenedores se pueden desplegar en cualquier máquina que tenga Docker instalado, sin importar el sistema operativo subyacente.

- **Uso eficiente de recursos**: Los contenedores comparten el kernel del sistema operativo, lo que los hace muy ligeros y fáciles de manejar.

- **Fácil integración continua y despliegue (CI/CD)**: Docker se integra bien con las herramientas de CI/CD, lo que facilita la automatización de pruebas y despliegues.

- **Aislamiento**: Garantiza que una aplicación en un contenedor no afecte a otras aplicaciones en el mismo host.

Empezar con Docker puede ser un poco desalentador al principio, pero con práctica, se convertirá en una herramienta esencial en tu kit de desarrollador. ¡No dudes en experimentar y seguir practicando!

## Introducción a AWS Lambda y la Computación Serverless

Aquí tienes una **introducción clara y completa a AWS Lambda y la computación serverless**, ideal para comprender sus fundamentos, beneficios y cómo empezar a usarla.

### ☁️ ¿Qué es la Computación Serverless?

**Serverless** (sin servidor) no significa que no haya servidores, sino que **tú no administras los servidores**. En lugar de preocuparte por infraestructura, te enfocas únicamente en el código y la lógica de negocio.

Con **serverless**:

* No gestionas servidores ni instancias.
* Pagas solo por el tiempo de ejecución real de tu código.
* La escalabilidad es automática.

### ⚙️ ¿Qué es AWS Lambda?

**AWS Lambda** es el servicio serverless de AWS que te permite ejecutar código **sin aprovisionar ni administrar servidores**.

### 🔑 Características clave:

* **Ejecución basada en eventos:** Tu función se ejecuta al recibir un evento (ej. HTTP, carga en S3, mensaje en SQS).
* **Escala automáticamente** según el tráfico.
* **Cobra por invocación y duración** (ms).
* Compatible con muchos lenguajes: **Python, Node.js, Java, Go, .NET, Ruby**.

### 🔄 ¿Cómo Funciona AWS Lambda?

1. **Subes tu código o lo escribes directamente en la consola.**
2. **Configuras un trigger/evento** (ejemplo: un endpoint API Gateway o una carga en S3).
3. Cuando el evento ocurre, **AWS ejecuta tu función Lambda**.
4. **Lambda se detiene automáticamente** una vez termina la ejecución.

### 🔧 Ejemplo Básico: Función Lambda en Python

```python
def lambda_handler(event, context):
    name = event.get("name", "Mundo")
    return {
        "statusCode": 200,
        "body": f"¡Hola, {name}!"
    }
```

Esta función puede responder a solicitudes HTTP enviadas a través de Amazon API Gateway.

### 🧠 Casos de Uso Comunes

* **APIs sin servidor:** Lambda + API Gateway.
* **Procesamiento de imágenes:** Lambda + S3.
* **Automatización:** Lambda como respuesta a eventos de CloudWatch o DynamoDB.
* **Chatbots, notificaciones, validaciones, etc.**

### 🛠️ Cómo Crear una Función Lambda (pasos básicos)

1. Ve a la consola de AWS → Lambda → “Crear función”.
2. Elige “Autor desde cero”.
3. Define:

   * Nombre de la función
   * Tiempo de ejecución (Node.js, Python, etc.)
   * Rol de ejecución (permisos IAM)
4. Escribe o sube tu código.
5. Configura el **evento trigger** (ejemplo: HTTP mediante API Gateway).
6. Guarda y prueba.

### 📊 Ventajas de AWS Lambda

✅ No gestionas infraestructura
✅ Escalado automático
✅ Paga solo por uso
✅ Alta disponibilidad
✅ Integración nativa con otros servicios de AWS

### ⚠️ Consideraciones

* **Tiempo de ejecución máximo:** 15 minutos por ejecución.
* **Tamaño máximo del paquete:** 50 MB (zipped), 250 MB descomprimido.
* **Estado efímero:** No almacena datos entre ejecuciones. Usa S3, DynamoDB, etc. para persistencia.
* **Tiempo de inicio (cold start):** Algunas funciones pueden tardar más en arrancar si no han sido invocadas recientemente.

### 🧪 Herramientas Útiles

* **AWS SAM** (Serverless Application Model): Framework para definir y desplegar apps serverless.
* **Serverless Framework:** Framework open-source que simplifica el desarrollo y despliegue de funciones serverless (multi-cloud).
* **AWS CloudFormation/Terraform:** Infraestructura como código.

### 📌 Conclusión

**AWS Lambda y el enfoque serverless** te permiten crear aplicaciones escalables, económicas y fáciles de mantener sin preocuparte por los servidores. Ideal para microservicios, automatización, APIs y tareas event-driven.

### Resumen

Lambda es un producto que implementa la filosofía de **Serverless**, lo cual significa no tener un servidor sino tener funciones que hagan cosas muy específicas (sin embargo sí se usan servidores que administra **AWS** sin que tú pienses en ello). Es código que puede conectarse a una base de datos, servicios web, etc.

En el mundo clásico se tenía un servidor o grupo de servidores corriendo software y teniendo microservicios. El software internamente resolvía todo y todo consistía en llamadas al mismo código. Con **Lambda** el enfoque es más de separar las funciones, ponerlas en diferentes servicios y correremos una parte del código en diferentes *endpoints*.

**Lambda escala automáticamente**: Esto quiere decir que si tu microservicio comienza a usarse más, se te brindarán más recursos para que corra siempre correctamente.

El costo de **Lambda** es atractivo porque AWS te da 1 millón de llamadas gratis por mes y cuando te excedas de eso, el costo es muy bajo.

Lenguajes soportados:

- Node.js (JavaScript)
- Python
- Java
- C#
- Go

**Lecturas recomendadas**

[AWS Lambda – Preguntas frecuentes](https://aws.amazon.com/es/lambda/faqs/ "AWS Lambda – Preguntas frecuentes")

## Creación y Configuración de Funciones Lambda en AWS

Aquí tienes una **guía completa paso a paso para la creación y configuración de funciones Lambda en AWS**, ya sea desde la consola, CLI o frameworks como Serverless Framework o AWS SAM.

### 🧠 ¿Qué es una Función Lambda?

Una **función Lambda** es una pieza de código que se ejecuta en respuesta a un **evento**, sin necesidad de administrar servidores. Puede activarse por una API REST, una carga en S3, un cambio en DynamoDB, o un evento programado.

### 🛠️ MÉTODO 1: Crear una Función Lambda desde la Consola de AWS

### 🔹 Paso 1: Ir a la consola

1. Inicia sesión en [AWS Console](https://console.aws.amazon.com/)
2. Ve a **Servicios > Lambda > Crear función**

### 🔹 Paso 2: Configurar la función

* **Nombre de la función**: `miFuncionEjemplo`
* **Tiempo de ejecución**: elige un lenguaje (ej. Python 3.10, Node.js 18, etc.)
* **Permisos**:

  * Crea un rol con permisos básicos de Lambda
  * O usa uno existente con políticas como `AWSLambdaBasicExecutionRole`

### 🔹 Paso 3: Escribir el código

Ejemplo para Python:

```python
def lambda_handler(event, context):
    return {
        'statusCode': 200,
        'body': '¡Hola desde Lambda!'
    }
```

Haz clic en **Deploy (Implementar)**.

### 🔹 Paso 4: Probar la función

1. Haz clic en **"Probar"**
2. Crea un evento de prueba (puedes dejar el JSON por defecto)
3. Ejecuta y revisa el resultado

### 🔁 MÉTODO 2: Crear función Lambda desde la CLI (AWS CLI)

### Requisitos:

* Tener [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) configurado con `aws configure`

### Pasos:

1. Crear un archivo `lambda_function.py`:

```python
def lambda_handler(event, context):
    return {
        "statusCode": 200,
        "body": "¡Hola desde AWS CLI!"
    }
```

2. Empaquetar el código en un ZIP:

```bash
zip function.zip lambda_function.py
```

3. Crear la función:

```bash
aws lambda create-function \
  --function-name miFuncionCLI \
  --runtime python3.10 \
  --role arn:aws:iam::<tu-id-cuenta>:role/<nombre-del-rol> \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip
```

### ⚙️ MÉTODO 3: Usar Serverless Framework

1. Instalar:

```bash
npm install -g serverless
```

2. Crear proyecto:

```bash
serverless create --template aws-python --path mi-lambda
cd mi-lambda
```

3. Editar `handler.py` y `serverless.yml`

4. Desplegar:

```bash
sls deploy
```

### 🔁 EVENTOS DISPONIBLES PARA DISPARAR UNA FUNCIÓN LAMBDA

* **API Gateway**: para exponer la función como una API REST/HTTP.
* **S3**: ejecución cuando se sube un archivo.
* **DynamoDB Streams**: para cambios en tablas.
* **EventBridge / CloudWatch Events**: tareas programadas.
* **SNS / SQS**: mensajes de cola o notificación.

### 🔒 CONFIGURACIONES ADICIONALES

* **Timeout máximo**: hasta 15 minutos
* **Memoria asignable**: entre 128 MB y 10,240 MB
* **Variables de entorno**
* **VPC**: si necesita acceso a RDS u otros recursos internos

### 📦 DEPENDENCIAS

Si necesitas instalar paquetes externos (Python):

```bash
mkdir python
pip install requests -t python/
zip -r layer.zip python
```

Y lo subes como **Lambda Layer**.

### 🧪 Prueba, Monitoreo y Logs

* **AWS CloudWatch Logs**: cada ejecución genera logs automáticamente.
* Puedes ver los logs directamente desde la consola Lambda o en CloudWatch.

### ✅ Conclusión

Crear y configurar funciones Lambda es fácil y escalable. Puedes hacerlo desde la consola, CLI o frameworks, y vincularlas a muchos servicios de AWS para construir flujos serverless potentes y económicos.

### Resumen

Cosas a tener en cuenta al momento de configurar una función Lambda:

- La opción de **Blueprints** se refiere a un documento o plano de lo que vas a hacer.
- Para cada una de las funciones hay que crear un rol que dé acceso a los recursos de **AWS** que nosotros elijamos. Por default, sólo trae unos permisos básicos. La idea es dar el menor número de permisos posibles.
- Puedes aprender a manejar el CLI [https://aws.amazon.com/es/cli/](https://aws.amazon.com/es/cli/ "https://aws.amazon.com/es/cli/") para otorgar permisos por línea de comandos.
- La entrada de un **Lambda** es un **API Gateway**.

**Lecturas recomendadas**

[AWS Serverless Application Repository - Amazon Web Services](https://aws.amazon.com/serverless/serverlessrepo/ "AWS Serverless Application Repository - Amazon Web Services")

## Creación de funciones Lambda en AWS con API Gateway

Aquí tienes una **guía paso a paso para crear una función Lambda conectada a Amazon API Gateway**, ideal para exponer una **API REST** usando arquitectura serverless.

### 🌐 ¿Qué vas a lograr?

Crear una función Lambda que responde a solicitudes HTTP a través de una API pública usando **API Gateway**. Por ejemplo:

```http
GET https://abcd1234.execute-api.us-east-1.amazonaws.com/prod/saludo
```

### 🧰 Requisitos Previos

* Cuenta de AWS activa.
* Permisos para crear funciones Lambda, roles IAM y API Gateway.
* Tener instalada la [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) o usar la consola web.

### 🚀 PASO A PASO EN LA CONSOLA

### 🔹 1. Crear Función Lambda

1. Ir a [AWS Lambda](https://console.aws.amazon.com/lambda/)

2. Haz clic en **“Crear función”**

3. Selecciona **“Autor desde cero”**

   * Nombre: `saludoLambda`
   * Tiempo de ejecución: `Python 3.10` o `Node.js 18.x`
   * Permisos: crear un rol nuevo con políticas básicas

4. Código de ejemplo (`Python`):

```python
def lambda_handler(event, context):
    nombre = event.get('queryStringParameters', {}).get('nombre', 'mundo')
    return {
        'statusCode': 200,
        'body': f'¡Hola, {nombre}!'
    }
```

Haz clic en **“Implementar”** para guardar los cambios.

### 🔹 2. Crear API Gateway

1. Ir a [Amazon API Gateway](https://console.aws.amazon.com/apigateway)

2. Selecciona **“Crear API” > REST API (antigua)** > **“Crear”**

3. Configura:

   * Nombre: `apiSaludo`
   * Tipo: **Pública**
   * Seguridad: abierta por ahora (puedes agregar auth después)

4. Crear recurso:

   * Click en `/` → **“Crear recurso”**
   * Nombre: `saludo`
   * Ruta: `/saludo`

5. Crear método:

   * Selecciona el recurso `/saludo`
   * Haz clic en **“Crear método” → GET**
   * Integración: selecciona **Lambda Function**
   * Escribe el nombre: `saludoLambda`
   * Marca la opción de usar la región correcta
   * Autoriza el acceso si se solicita

### 🔹 3. Desplegar la API

1. Haz clic en **“Acciones” > “Implementar API”**
2. Etapa: `prod`
3. Anota el **endpoint URL** generado, como:

```
https://abcd1234.execute-api.us-east-1.amazonaws.com/prod/saludo
```

### 🔹 4. Probar la API

Puedes hacer una solicitud en el navegador o con `curl`:

```bash
curl "https://abcd1234.execute-api.us-east-1.amazonaws.com/prod/saludo?nombre=Mario"
```

Respuesta esperada:

```json
¡Hola, Mario!
```

### ⚙️ OPCIONAL: Crear con Serverless Framework

1. Instala:

```bash
npm install -g serverless
```

2. Crea un proyecto:

```bash
serverless create --template aws-python --path saludo-lambda-api
cd saludo-lambda-api
```

3. Edita `handler.py`:

```python
def hello(event, context):
    nombre = event.get('queryStringParameters', {}).get('nombre', 'mundo')
    return {
        "statusCode": 200,
        "body": f"¡Hola, {nombre}!"
    }
```

4. Edita `serverless.yml`:

```yaml
service: saludo-lambda-api

provider:
  name: aws
  runtime: python3.10
  region: us-east-1

functions:
  saludo:
    handler: handler.hello
    events:
      - http:
          path: saludo
          method: get
```

5. Desplega:

```bash
sls deploy
```

6. Probar el endpoint generado en consola al final del despliegue.

### 🧪 Consejos

* Usa **CloudWatch Logs** para depurar errores.
* Usa **API keys o Cognito** si quieres autenticar la API.
* Usa **variables de entorno** para almacenar secretos o configuración.

### ✅ Conclusión

Conectar Lambda a API Gateway te permite construir microservicios y APIs REST completamente **serverless**, escalables y económicas. Puedes extenderlo con múltiples rutas, autenticación y validaciones.

### Resumen

El reto de esta clase consiste en crear una función **Lambda** con su **API Gateway** probando con diferentes lenguajes y diferentes códigos. Juega con los **Blueprints** y todas las opciones que tienes para crear funciones.

**Lecturas recomendadas**

[AWS | Lambda - Gestión de recursos informáticos](https://aws.amazon.com/es/lambda/ "AWS | Lambda - Gestión de recursos informáticos")

## Despliegue y Gestión de Aplicaciones con Elastic Beanstalk

Aquí tienes una **guía completa paso a paso** para el **despliegue y gestión de aplicaciones con AWS Elastic Beanstalk**, una de las formas más simples de desplegar aplicaciones web en AWS sin preocuparte demasiado por la infraestructura.

### 🧠 ¿Qué es AWS Elastic Beanstalk?

Elastic Beanstalk (EB) es un **servicio de orquestación de aplicaciones** que te permite desplegar automáticamente aplicaciones web en servicios como **EC2, S3, RDS, Load Balancer, Auto Scaling**, etc., sin tener que configurarlos uno por uno.

### ✅ ¿Qué puedes desplegar?

* Aplicaciones en: **Node.js, Python, Java, .NET, PHP, Ruby, Go, Docker**
* Aplicaciones monolíticas o con múltiples servicios
* Con o sin base de datos

### 🧰 Requisitos Previos

1. Tener una **cuenta AWS**
2. Instalar el **AWS CLI**:
   [Instrucciones](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
3. Instalar la **Elastic Beanstalk CLI (EB CLI)**:
   [Instrucciones](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html)
4. Configurar tus credenciales:

```bash
aws configure
```

### 🛠️ PASO 1: Preparar tu Aplicación

### Ejemplo en Python (Flask):

```bash
mkdir app-eb && cd app-eb
```

1. Crea un archivo `application.py`:

```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "¡Hola desde Beanstalk!"
```

2. Crea un archivo `requirements.txt`:

```
Flask==2.2.5
```

3. Crea el archivo `wsgi.py` (requerido por EB):

```python
from application import app

if __name__ == "__main__":
    app.run()
```

4. Estructura final:

```
app-eb/
├── application.py
├── requirements.txt
└── wsgi.py
```

### 🛠️ PASO 2: Inicializar Elastic Beanstalk

```bash
eb init -p python-3.10 app-eb --region us-east-1
```

* Elige una clave SSH si deseas acceder a la instancia
* Esto crea el archivo `.elasticbeanstalk/config.yml`

### 🛠️ PASO 3: Crear un Entorno y Desplegar

```bash
eb create app-eb-env
```

Esto:

* Lanza una instancia EC2
* Configura Load Balancer
* Instala los paquetes de Python
* Sube el código

Después de unos minutos:

```bash
eb open
```

👉 Te abrirá la aplicación desplegada en tu navegador.

### 🛠️ PASO 4: Actualizar la Aplicación

Haz cambios en tu código y ejecuta:

```bash
eb deploy
```

¡Y listo! Se actualiza automáticamente.

### 🔒 PASO 5 (Opcional): Añadir Base de Datos RDS

Puedes añadir una base de datos desde la consola:

1. Ve al entorno en la consola de EB
2. Selecciona **Configuration > Database**
3. Elige el tipo de motor (MySQL, PostgreSQL, etc.)
4. Elastic Beanstalk gestionará el RDS junto con tu aplicación

⚠️ Nota: Si eliminas el entorno, la base de datos se elimina también a menos que la marques como persistente.

### ⚙️ Gestión desde la Consola EB

Desde la consola de Elastic Beanstalk puedes:

* Ver logs
* Escalar instancias
* Reiniciar el entorno
* Configurar variables de entorno
* Agregar monitoreo (CloudWatch)

### 🔁 Comandos útiles con EB CLI

| Comando        | Acción                    |
| -------------- | ------------------------- |
| `eb init`      | Inicializar configuración |
| `eb create`    | Crear entorno             |
| `eb deploy`    | Desplegar cambios         |
| `eb open`      | Abrir en el navegador     |
| `eb status`    | Ver estado del entorno    |
| `eb terminate` | Eliminar entorno          |
| `eb logs`      | Ver logs                  |

### 🧪 Buenas Prácticas

* Usa `.ebextensions` para configuraciones adicionales (por ejemplo, instalar paquetes, crear usuarios, configurar Nginx).
* Usa Elastic Load Balancer + Auto Scaling para alta disponibilidad.
* Usa entornos separados para producción y pruebas (`eb create app-env-dev`, `app-env-prod`).

### 📦 Extra: Ejemplo de `.ebextensions` para RDS

```yaml
# .ebextensions/db.config
option_settings:
  aws:elasticbeanstalk:application:environment:
    DB_HOST: mydb.<region>.rds.amazonaws.com
    DB_PORT: '3306'
    DB_USER: admin
    DB_PASS: mypassword
```

### ✅ Conclusión

Elastic Beanstalk te permite desplegar, escalar y gestionar tu aplicación sin preocuparte por configurar servidores, balanceadores o escalado manual. Es ideal para desarrolladores que quieren enfocarse en su aplicación y no en la infraestructura.

### Resumen

**Elastic Beanstalk** es una arquitectura para cuando vas a hacer una entrega a producción de un proyecto web que tengas. Su ventaja es que incluye todo lo que necesitas en un sólo paquete:

- Tienes un Endpoint donde puedes a través de Route 53 editar tu dominio.
- Puedes tener un **Load Balancer**
- Tienes instancias **EC2** Linux o Windows con soporte a muchos lenguajes.

Maneja las siguientes plataformas:

- Docker
- Go
- Java SE
- Java / Tomcat
- .NET (sobre Windows)
- NodeJS
- PHP
- Otros

**Elastic Beanstalk** te permite de manera muy fácil hacer un rollback, teniendo una gran flexibilidad para hacer un arreglo.
Esta arquitectura es auto-escalable dependiendo del tráfico o necesidades.

## Creación de Ambientes en Elastic Beanstalk con PHP

Aquí tienes una guía paso a paso para la **creación de ambientes en AWS Elastic Beanstalk con PHP**, ideal para desplegar tus sitios o aplicaciones web rápidamente en la nube sin gestionar infraestructura manualmente.

### 🧠 ¿Qué es un Ambiente en Elastic Beanstalk?

Un **ambiente** en Elastic Beanstalk es un entorno de ejecución completo que incluye:

* Una instancia EC2 con tu aplicación desplegada
* Un servidor web configurado automáticamente (Apache para PHP)
* Un balanceador de carga (si es necesario)
* Escalabilidad automática y monitoreo

### ✅ Requisitos Previos

* Cuenta de AWS activa
* Tener la [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
* Tener la [EB CLI](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html)
* Configurar AWS CLI con:

  ```bash
  aws configure
  ```

### 🛠️ Paso 1: Crear un Proyecto PHP

1. Crea un directorio para tu app:

```bash
mkdir app-php-eb && cd app-php-eb
```

2. Crea un archivo `index.php`:

```php
<?php
echo "<h1>¡Hola desde AWS Elastic Beanstalk con PHP!</h1>";
?>
```

3. Estructura final:

```
app-php-eb/
└── index.php
```

### 🛠️ Paso 2: Inicializar el Proyecto con EB CLI

```bash
eb init -p php app-php-eb --region us-east-1
```

Responde las preguntas:

* Selecciona la región
* Crea o selecciona una clave SSH (opcional)

Esto genera un archivo `.elasticbeanstalk/config.yml`.

### 🛠️ Paso 3: Crear un Ambiente

```bash
eb create ambiente-php
```

Esto:

* Crea el entorno EC2 con Apache y PHP
* Despliega tu app
* Configura Elastic Load Balancer si es necesario

Espera unos minutos hasta que diga "Environment is ready".

### 🛠️ Paso 4: Ver tu Aplicación

```bash
eb open
```

Se abrirá tu sitio en el navegador.
Verás:

```html
¡Hola desde AWS Elastic Beanstalk con PHP!
```

### 🛠️ Paso 5: Desplegar Cambios

Haz cambios en tus archivos `.php` y luego usa:

```bash
eb deploy
```

Elastic Beanstalk empaquetará y redeplegará tu aplicación.

### 🔄 Comandos útiles EB CLI

| Comando        | Acción                         |
| -------------- | ------------------------------ |
| `eb init`      | Inicializar proyecto Beanstalk |
| `eb create`    | Crear ambiente                 |
| `eb deploy`    | Subir código                   |
| `eb open`      | Abrir la app en navegador      |
| `eb status`    | Ver estado del entorno         |
| `eb logs`      | Ver logs                       |
| `eb terminate` | Eliminar el entorno            |

### 🧰 Extra: Variables de Entorno

Puedes definir variables de entorno PHP desde la EB CLI:

```bash
eb setenv DB_HOST=mi-db.rds.amazonaws.com DB_USER=admin DB_PASS=secreto
```

Y acceder a ellas en PHP así:

```php
$host = getenv('DB_HOST');
```

### 🧩 Extra: Archivos `.ebextensions`

Puedes incluir configuraciones adicionales creando un directorio `.ebextensions` con archivos `.config`. Por ejemplo, para instalar extensiones PHP:

```yaml
# .ebextensions/php.config
packages:
  yum:
    php-mbstring: []
```

### ✅ Conclusión

Elastic Beanstalk con PHP es una manera **sencilla y profesional** de desplegar sitios PHP escalables en AWS. No necesitas configurar servidores ni manejar despliegues manualmente.

### Resumen

Cosas a tener en cuenta al momento de crear un ambiente:

- Debemos tener nuestra aplicación en un archivo .zip. Si es la primera vez que usas el comando para crear archivos .zip, debes poner esto en la línea de comandos “sudo apt-get install zip -y”.
- El comando para crear el archivo .zip es “zip -r nombredelzipfile.zip archivos”. Muchos archivos deberán ponerse de forma explícita como los .env
- En “Version label” es recomendado poner el número de la versión que estamos manejando que nos permite recordar cuando tenemos más archivos y podamos devolvernos en el tiempo a alguna versión en específico si lo requerimos.

## Actualización de Aplicaciones en Elastic Beanstalk

Aquí tienes una guía clara y paso a paso para realizar la **actualización de aplicaciones en AWS Elastic Beanstalk**, ya sea mediante la **EB CLI**, la **consola web**, o incluso **CI/CD automatizado**.

### 🧠 ¿Qué significa "actualizar una aplicación" en Elastic Beanstalk?

Actualizar una aplicación en Elastic Beanstalk implica:

* Subir una nueva versión del código
* Desplegarlo sobre un entorno existente
* Aplicar cambios de configuración si es necesario

### ✅ 1. Requisitos Previos

Antes de actualizar:

* Debes tener una aplicación ya desplegada en Elastic Beanstalk.
* Tener configurada la [EB CLI](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html).
* Tener el código fuente actualizado en tu máquina local.

### 🛠️ 2. Método 1: Actualización desde EB CLI

### 📝 Paso 1: Modifica tu aplicación local

Haz los cambios necesarios en tu código fuente.

### 🚀 Paso 2: Desplegar con `eb deploy`

```bash
eb deploy
```

Este comando:

* Empaqueta tu código
* Crea una nueva versión de la aplicación
* La despliega en el entorno actual

### ✅ Paso 3: Verifica que todo funcione

```bash
eb open
```

### 🖥️ 3. Método 2: Actualización desde la Consola Web de AWS

### Paso 1: Empaqueta tu código

* Crea un archivo `.zip` con el contenido de tu aplicación (sin incluir carpetas como `.git`, `node_modules`, etc.)

```bash
zip -r app-v2.zip .
```

### Paso 2: Ve a la Consola de Elastic Beanstalk

1. Abre: [https://console.aws.amazon.com/elasticbeanstalk](https://console.aws.amazon.com/elasticbeanstalk)
2. Selecciona tu aplicación
3. Ve a **Application versions**
4. Haz clic en **Upload**
5. Carga el archivo `.zip`
6. Haz clic en **Deploy** y selecciona el entorno

### 🔄 4. Método 3: Automatización con CI/CD (opcional)

Puedes configurar un flujo de CI/CD con GitHub Actions, CodePipeline o GitLab CI que:

* Detecta cambios en tu repositorio
* Empaqueta y despliega automáticamente a Elastic Beanstalk

Ejemplo básico de GitHub Actions:

```yaml
name: Deploy to Elastic Beanstalk

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to EB
        uses: einaregilsson/beanstalk-deploy@v20
        with:
          aws_access_key: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws_secret_key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          application_name: "mi-app"
          environment_name: "mi-entorno"
          region: "us-east-1"
          version_label: "v1-${{ github.run_number }}"
          deployment_package: "app.zip"
```

### 🧰 Extras útiles

### Ver historial de versiones:

```bash
eb appversion
```

### Cambiar a una versión anterior:

```bash
eb deploy --version label-antiguo
```

### Ver logs:

```bash
eb logs
```

### ✅ Conclusión

Actualizar aplicaciones en Elastic Beanstalk es sencillo, y puedes hacerlo:

* Con un solo comando (`eb deploy`)
* Desde la consola web
* O automáticamente con CI/CD

Esto te da **velocidad**, **control de versiones** y **facilidad para revertir** si algo sale mal.

### Resumen

#### ¿Cómo actualizar una versión en Elastic Beanstalk?

Actualizar la versión de una aplicación en Elastic Beanstalk es un proceso clave para asegurar que se tenga siempre la infraestructura más eficiente y segura. En Elastic Beanstalk, se pueden crear varios ambientes para una aplicación, como desarrollo, pruebas, calidad y producción. Tener configuraciones específicas para cada entorno optimiza el flujo de trabajo y minimiza errores.

#### ¿Cómo crear un archivo ZIP para la nueva versión?

Antes de proceder con la actualización, es esencial crear un archivo ZIP con los cambios efectuados en la aplicación. Este archivo contendrá todos los archivos necesarios, como el `index.php` y `Quotes.txt`. A continuación, se muestra cómo hacerlo:

1. Abre el proyecto en tu consola.
2. Realiza los cambios necesarios en el código, como por ejemplo, actualizar el texto o agregar nuevos autores.
3. Crea un archivo ZIP que contenga la nueva versión, por ejemplo, Quotes versión 2.
`zip -r Quotes_v2.zip index.php Quotes.txt`

#### ¿Cómo desplegar la nueva versión?

Una vez que el archivo ZIP está listo, se debe subir y desplegar en Elastic Beanstalk.

1. En la consola de Elastic Beanstalk, haz clic en el botón "Upload and Deploy".
2. Selecciona el archivo de la nueva versión ZIP y asigna un número de versión nuevo, por ejemplo, 2.0.

El proceso de despliegue puede hacerse de distintas maneras, cada una con sus ventajas y desventajas.

#### ¿Qué tipos de despliegues existen?

Se pueden seleccionar entre despliegues simultáneos o en etapas (Rolling Deployments):

- **Simultáneo**: Actualiza todos los servidores al mismo tiempo, minimizando el tiempo de actualización pero con un mayor riesgo si algo falla.
- **Rolling**: Actualiza un tercio de los servidores al principio, seguido por los siguientes tercios, hasta completar la actualización. Esto reduce la posibilidad de un fallo total, pero puede afectar a los usuarios si hay un desajuste entre versiones.

#### ¿Cuál es la mejor estrategia de despliegue?

Decidir la mejor estrategia depende de factores como:

1. Horarios de menos tráfico (para minimizar la afectación al usuario).
2. Configuración y programación de la aplicación.

Recomendamos realizar las actualizaciones fuera de las horas pico o cuando haya menos usuarios activos, por ejemplo, temprano en la mañana o tarde en la noche.

#### ¿Cómo verificar que la actualización fue exitosa?

Una vez que el despliegue ha comenzado, es vital monitorizar los eventos para garantizar que todo se haya actualizado correctamente. En el apartado de "Eventos recientes" verás las notificaciones sobre el estado del despliegue. Si se presenta algún problema, esto queda registrado y puedes tomar acciones correctivas.

Además, al hacer clic en "Health", puedes ver detalles sobre las instancias de EC2, como el tiempo que tomaron en actualizarse y la versión actual implementada. Si todo está en verde y muestra la nueva versión, entonces la actualización fue exitosa.

Elastic Beanstalk facilitan el despliegue al permitir una revisión detallada del estado de tu aplicación. Eso sí, si aún deseas aprender más sobre AWS y Elastic Beanstalk, un curso como el de Introducción a AWS de Platzi podría enriquecer aún más tu conocimiento.

Continúa explorando y aprendiendo más sobre cómo mejorar tus habilidades en manejo de servidores y despliegues. ¡Buena suerte en tus futuras actualizaciones!

**Lecturas recomendadas**

[AWS | Elastic beanstalk para aplicaciones web desarrolladas con Java](https://aws.amazon.com/es/elasticbeanstalk/ "AWS | Elastic beanstalk para aplicaciones web desarrolladas con Java")

## Creación de Aplicaciones en Elastic Beanstalk

Aquí tienes una guía completa para la **creación de aplicaciones en AWS Elastic Beanstalk**, paso a paso, ideal para quienes desean desplegar rápidamente aplicaciones web sin preocuparse por la infraestructura.

### 🧠 ¿Qué es una "aplicación" en Elastic Beanstalk?

En Elastic Beanstalk, una **aplicación** es una colección lógica que contiene:

* Una o más **versiones de código**
* Uno o más **ambientes de ejecución** (entornos con EC2, balanceadores, RDS, etc.)

### ✅ Requisitos Previos

Antes de comenzar:

* Cuenta de AWS activa
* AWS CLI y EB CLI instaladas
* Ejecutar `aws configure` para configurar tus credenciales

### 🛠️ Paso 1: Crear tu proyecto

Ejemplo con una aplicación sencilla (Node.js, PHP, Python, etc.)

### 📁 Estructura básica:

```bash
mkdir mi-app && cd mi-app
```

Ejemplo con Node.js:

```javascript
// index.js
const http = require('http');
const port = process.env.PORT || 3000;
const server = http.createServer((req, res) => {
  res.end('¡Hola desde Elastic Beanstalk!');
});
server.listen(port);
```

Y su `package.json`:

```json
{
  "name": "mi-app",
  "version": "1.0.0",
  "main": "index.js",
  "scripts": {
    "start": "node index.js"
  }
}
```

### 🛠️ Paso 2: Inicializar Elastic Beanstalk

Ejecuta:

```bash
eb init
```

### Preguntas comunes:

* Nombre de la aplicación: `mi-app`
* Plataforma: Elige según tu lenguaje (Node.js, PHP, Python, etc.)
* Región: por ejemplo `us-east-1`
* ¿Deseas usar SSH? (opcional, útil para debug)

Esto crea un archivo `.elasticbeanstalk/config.yml`.

### 🛠️ Paso 3: Crear el ambiente de ejecución

```bash
eb create mi-entorno
```

Elastic Beanstalk creará:

* Instancia EC2
* Balanceador de carga (si aplica)
* Configuración de red, seguridad, monitoreo

Este proceso tarda entre 3–7 minutos.

### 🛠️ Paso 4: Abrir la aplicación

Cuando el entorno esté listo, ejecuta:

```bash
eb open
```

Verás tu aplicación corriendo en la URL pública proporcionada por Elastic Beanstalk.

### 🚀 Paso 5: Actualizar tu aplicación (despliegue continuo)

Haz cambios en tu código y usa:

```bash
eb deploy
```

Beanstalk empaquetará el código, creará una nueva versión y la aplicará al entorno.

### 🔧 Extras opcionales

### 🧩 Variables de entorno

```bash
eb setenv DB_USER=admin DB_PASS=secreto
```

Y luego en tu código:

```js
const user = process.env.DB_USER;
```

### 🧰 Archivos `.ebextensions`

Permiten configurar el entorno automáticamente (por ejemplo, instalar paquetes o activar logs).

Ejemplo:

```yaml
# .ebextensions/01_packages.config
packages:
  yum:
    git: []
```

### 📌 Comandos útiles EB CLI

| Comando        | Descripción                   |
| -------------- | ----------------------------- |
| `eb init`      | Inicializa proyecto Beanstalk |
| `eb create`    | Crea un nuevo entorno         |
| `eb open`      | Abre la app en el navegador   |
| `eb deploy`    | Despliega nueva versión       |
| `eb logs`      | Muestra logs del entorno      |
| `eb terminate` | Elimina un entorno            |

### ✅ Conclusión

Elastic Beanstalk te permite crear, desplegar y escalar tus aplicaciones en la nube de forma **rápida y sin gestionar servidores manualmente**. Es ideal para aplicaciones web, APIs y prototipos listos para producción.

### Resumen

##### ¿Qué es Elastic Beanstalk y por qué es importante?

Elastic Beanstalk es un servicio de Amazon Web Services (AWS) que facilita la implementación y el manejo de aplicaciones en la nube. Imagina que quieres poner en marcha tu aplicación web rápidamente, pero sin complicarte con la infraestructura subyacente. Aquí es donde Elastic Beanstalk entra en juego: te permite centrarte en el desarrollo de la aplicación mientras automatiza tareas como la provisión de servidores, balanceo de carga, escalado y monitoreo.

#### ¿Cuáles son los beneficios de usar Elastic Beanstalk?

Usar Elastic Beanstalk ofrece varios beneficios que pueden impactar positivamente en la gestión de tus aplicaciones:

- **Simplicidad y rapidez**: Proporciona un entorno rápidamente desplegable, lo que acelera el lanzamiento de aplicaciones sin tener que preocuparse por la configuración manual de la infraestructura.
- **Escalabilidad automática**: Elastic Beanstalk ajusta automáticamente la capacidad de las instancias según la demanda.
- **Integración con otros servicios de AWS**: Se integra fácilmente con otros servicios de AWS, lo que potencia aún más tus aplicaciones a través de bases de datos, almacenamiento, seguridad, etc.
- **Soporte para múltiples lenguajes y entorno**s: Compatible con una variedad de lenguajes de programación, como Java, .NET, PHP, Node.js, Python, Ruby, Go y Docker.

#### ¿Cuáles son los pasos para crear una aplicación en Elastic Beanstalk?

Implementar una aplicación en Elastic Beanstalk es un proceso relativamente directo gracias a su interfaz amigable. Aquí te presentamos una guía básica para comenzar:

1. **Prepara tu aplicación**: Asegúrate de que está lista para desplegarse, con todos los archivos y configuraciones necesarias.
2. **Crea un entorno en Elastic Beanstalk**: Inicia sesión en tu consola de AWS, navega hasta Elastic Beanstalk y crea un nuevo entorno. Elige la plataforma adecuada para tu aplicación.
3. **Despliega tu aplicación**: Sube tu código fuente a Elastic Beanstalk. Puedes hacerlo directamente desde la consola de AWS o usar la CLI de Elastic Beanstalk.
4. **Configura tu entorno**: Ajusta las configuraciones según tus necesidades, tales como el escalado automático, balanceo de carga y las variables de entorno.
5. **Monitorea y ajusta**: Utiliza las herramientas de monitoreo integradas para ajustar y optimizar el rendimiento de tus aplicaciones.

#### ¿Cómo pueden los desarrolladores aprovechar al máximo Elastic Beanstalk?

Elastic Beanstalk no solo facilita el despliegue de aplicaciones, sino que también ofrece herramientas para maximizar su potencial. Aquí algunas recomendaciones:

- **Automatización**: Aprovecha las capacidades de escritura de scripts para automatizar procesos repetitivos, lo que disminuye el tiempo gastado en tareas manuales.
- **Monitoreo proactiv**o: Utiliza CloudWatch para monitorear el rendimiento y establecer alertas que te permita actuar antes de que surja un problema significativo.
- **Revisar registros**: Access logs y otros registros se pueden utilizar para diagnosticar y solucionar problemas rápidamente, optimizando el tiempo de funcionamiento.
- **Pruebas y política de rollback**: Implementa pruebas automatizadas y establece políticas de retroceso para asegurarte de que las actualizaciones no interrumpan servicios críticos.

#### ¿Qué desafíos podrías encontrar al usar Elastic Beanstalk?

Aunque Elastic Beanstalk simplifica mucho la gestión de aplicaciones, existen algunos desafíos para tener en cuenta:

- **Configuraciones avanzadas**: Si deseas configuraciones extremadamente detalladas, puede que necesites ir más allá de lo que Elastic Beanstalk ofrece de manera predeterminada.
- **Costos**: A medida que la aplicación escala, los costos también pueden aumentar rápidamente. Se debe supervisar el uso de recursos para optimizar los gastos.
- **Dependencia en AWS**: Usar Elastic Beanstalk significa depender del ecosistema AWS, lo cual puede ser una limitación si deseas flexibilidad multicloud en el futuro.

Aunque puede haber obstáculos, con una buena planificación y práctica, Elastic Beanstalk puede transformar significativamente la forma en que gestionas y despliegas aplicaciones en la nube.

Finalmente, te animamos no solo a implementar aplicaciones en Elastic Beanstalk, sino también a experimentar y compartir tu experiencia con otros desarrolladores para obtener retroalimentación valiosa. La comunidad siempre es una gran fuente de aprendizaje y apoyo.

## Servicios y Funcionalidades Actuales de AWS

AWS (Amazon Web Services) ofrece **más de 200 servicios totalmente integrados**, utilizados por millones de empresas y desarrolladores en todo el mundo. A continuación te presento un resumen organizado por **categorías clave** con sus principales servicios y funcionalidades **actualizadas al 2025**:

### ☁️ **1. Cómputo**

Servicios que permiten ejecutar código o instancias de manera escalable.

| Servicio          | Funcionalidad principal                                    |
| ----------------- | ---------------------------------------------------------- |
| **EC2**           | Instancias virtuales para cualquier sistema operativo      |
| **Lambda**        | Ejecuta código sin servidores (serverless)                 |
| **ECS / Fargate** | Contenedores orquestados (con o sin gestión de servidores) |
| **EKS**           | Kubernetes administrado por AWS                            |
| **Lightsail**     | VPS fácil de usar para sitios web y apps pequeñas          |
| **Batch**         | Procesamiento de trabajos en lotes                         |

### 🗃️ **2. Almacenamiento**

Servicios para guardar y recuperar datos de forma segura.

| Servicio    | Funcionalidad principal                        |
| ----------- | ---------------------------------------------- |
| **S3**      | Almacenamiento de objetos escalable y duradero |
| **EBS**     | Volúmenes de disco para EC2                    |
| **EFS**     | Sistema de archivos compartido (NFS) para EC2  |
| **Glacier** | Almacenamiento de archivo de bajo costo        |
| **FSx**     | File systems compatibles con Windows y Lustre  |

### 🧠 **3. Inteligencia Artificial y Machine Learning**

| Servicio        | Funcionalidad principal                             |
| --------------- | --------------------------------------------------- |
| **SageMaker**   | Desarrollo y despliegue de modelos ML               |
| **Bedrock**     | Acceso a modelos fundacionales (como Claude, Titan) |
| **Rekognition** | Análisis de imágenes y video                        |
| **Transcribe**  | Transcripción de audio a texto                      |
| **Polly**       | Conversión de texto a voz                           |
| **Comprehend**  | Análisis de texto (sentimientos, entidades, etc.)   |

### 🗄️ **4. Bases de Datos**

| Servicio        | Tipo de base de datos                              |
| --------------- | -------------------------------------------------- |
| **RDS**         | Relacional (MySQL, PostgreSQL, Oracle, SQL Server) |
| **Aurora**      | Motor relacional compatible con MySQL/PostgreSQL   |
| **DynamoDB**    | Base de datos NoSQL altamente escalable            |
| **DocumentDB**  | NoSQL tipo MongoDB                                 |
| **ElastiCache** | Caché en memoria (Redis y Memcached)               |
| **Neptune**     | Base de datos de grafos                            |
| **Timestream**  | Base de datos para series temporales               |

### 🌐 **5. Redes y Entrega de Contenido**

| Servicio        | Funcionalidad principal                     |
| --------------- | ------------------------------------------- |
| **VPC**         | Redes privadas virtuales                    |
| **CloudFront**  | CDN (entrega global de contenido)           |
| **Route 53**    | DNS escalable y balanceo de tráfico         |
| **API Gateway** | Creación y gestión de APIs REST y WebSocket |
| **PrivateLink** | Conexión privada entre servicios y VPCs     |

### 🔐 **6. Seguridad e Identidad**

| Servicio                          | Funcionalidad principal                              |
| --------------------------------- | ---------------------------------------------------- |
| **IAM**                           | Control de acceso basado en roles                    |
| **Cognito**                       | Autenticación de usuarios y federación (OAuth, SAML) |
| **Secrets Manager**               | Gestión segura de secretos y credenciales            |
| **KMS**                           | Gestión de claves de cifrado                         |
| **Inspector / GuardDuty / Macie** | Seguridad automatizada y detección de amenazas       |

### 🧰 **7. DevOps e Integración Continua**

| Servicio           | Funcionalidad principal                               |
| ------------------ | ----------------------------------------------------- |
| **CodePipeline**   | Orquestación de CI/CD                                 |
| **CodeBuild**      | Compilación de código                                 |
| **CodeDeploy**     | Despliegue automático de aplicaciones                 |
| **CloudFormation** | Infraestructura como código                           |
| **CDK**            | Infraestructura como código usando lenguajes modernos |

### 📊 **8. Monitoreo, Logging y Observabilidad**

| Servicio       | Funcionalidad principal                 |
| -------------- | --------------------------------------- |
| **CloudWatch** | Logs, métricas, alarmas                 |
| **X-Ray**      | Trazabilidad distribuida para debugging |
| **AWS Config** | Auditoría de configuraciones            |
| **CloudTrail** | Registro de actividades en AWS          |

### 📈 **9. Análisis y Big Data**

| Servicio       | Funcionalidad principal                        |
| -------------- | ---------------------------------------------- |
| **Athena**     | Consultas SQL sobre datos en S3                |
| **Redshift**   | Almacenamiento de datos analíticos             |
| **Kinesis**    | Ingesta y análisis de streaming en tiempo real |
| **QuickSight** | BI y visualización de datos                    |
| **Glue**       | ETL (extracción, transformación y carga)       |

### 🏗️ **10. Migración e Híbrido**

| Servicio                | Funcionalidad principal                            |
| ----------------------- | -------------------------------------------------- |
| **DMS**                 | Migración de bases de datos                        |
| **Snowball / Snowcone** | Transferencia física de grandes volúmenes de datos |
| **Storage Gateway**     | Integración de almacenamiento local y en la nube   |
| **Outposts**            | Extensión de AWS en centros de datos locales       |

### 🧩 Extras recientes y relevantes (2024-2025)

| Servicio            | Novedades/Funcionalidad                       |
| ------------------- | --------------------------------------------- |
| **Amazon Q**        | Asistente de IA para desarrollo y operaciones |
| **AWS AppComposer** | Entorno visual para construir apps serverless |
| **PartyRock**       | Plataforma creativa no-code con IA generativa |
| **Amazon Titan**    | Familia de modelos de IA propia de AWS        |

### ✅ Conclusión

Amazon Web Services es una plataforma **integral, escalable y segura**, ideal para todo tipo de proyectos: desde **startups** hasta **corporaciones globales**. Ofrece servicios para **infraestructura**, **desarrollo de aplicaciones**, **IA**, **análisis de datos**, **DevOps**, y más.

### Resumen

**AWS** está en constante crecimiento, siempre tendrá nuevos servicios o features. No dejes de estar aprendiendo nuevas cosas y capacitandote cada vez más.

Sigue estudiando con [cursos de AWS](https://platzi.com/ruta/aws/ "cursos de AWS").