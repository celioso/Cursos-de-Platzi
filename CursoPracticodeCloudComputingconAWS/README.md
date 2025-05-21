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