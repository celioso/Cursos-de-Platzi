# Curso Práctico de AWS Cómputo con EC2

## Configuración de Budget en AWS

La configuración de **AWS Budget** te permite establecer límites de gasto y recibir alertas cuando tu consumo de recursos en AWS alcanza ciertos umbrales. Esto es fundamental para evitar costos inesperados.

### 🔹 **Pasos para configurar un Budget en AWS**
#### 🛠️ **1. Acceder a AWS Budgets**
1️⃣ Inicia sesión en tu cuenta de **AWS Management Console**.  
2️⃣ Ve a **Billing Dashboard** (Panel de Facturación).  
3️⃣ En el menú lateral, selecciona **Budgets**.

#### 🔹 **2. Crear un nuevo presupuesto**
1️⃣ Haz clic en **Create a budget**.  
2️⃣ Elige el tipo de presupuesto según tu necesidad:  
   - **Cost Budget** (*Presupuesto basado en costos*): Define un límite de gasto en dólares.  
   - **Usage Budget** (*Presupuesto basado en uso*): Controla el uso de un recurso específico (como GB almacenados o horas de cómputo).  
   - **Reservation Budget** (*Presupuesto de reservas*): Monitorea instancias reservadas.

#### 🔹 **3. Configurar los detalles**
1️⃣ Asigna un **nombre** al presupuesto.  
2️⃣ Selecciona si el **presupuesto será mensual, trimestral o anual**.  
3️⃣ Define el **monto o nivel de uso permitido**.

#### 🔹 **4. Configurar alertas**
1️⃣ Habilita alertas para recibir notificaciones cuando tu consumo supere un porcentaje del presupuesto.  
   - Ejemplo: Recibir una alerta al **80% del presupuesto**.  
2️⃣ Ingresa una dirección de **correo electrónico** o **SNS Topic** donde recibirás las notificaciones.

#### 🔹 **5. Revisar y crear**
✅ Verifica la configuración y haz clic en **Create budget**.  
✅ Recibirás alertas cuando se alcance el umbral definido.

### 🎯 **Beneficios de configurar un AWS Budget**
✔️ **Evitas costos inesperados.**  
✔️ **Controlas el gasto en servicios específicos.**  
✔️ **Automatizas alertas para reaccionar a tiempo.**  
✔️ **Optimización del uso de recursos en AWS.**  

🚀 **Con esta configuración, puedes asegurarte de mantener tus costos bajo control en AWS.**

**Lecturas recomendadas**

[https://console.aws.amazon.com/billing/home#/account](https://console.aws.amazon.com/billing/home#/account)

## Fundamentos de EC2

Amazon **EC2 (Elastic Compute Cloud)** es un servicio de computación en la nube que permite ejecutar máquinas virtuales (instancias) en AWS, brindando escalabilidad y flexibilidad para distintas cargas de trabajo.

### 🔹 **1. Características Principales**  

✅ **Escalabilidad**: Aumenta o reduce la cantidad de instancias según la demanda.  
✅ **Pago por uso**: Solo pagas por el tiempo que usas la instancia.  
✅ **Diversos tipos de instancias**: Optimizadas para CPU, memoria, almacenamiento o GPU.  
✅ **Elección de SO**: Puedes usar Linux, Windows, macOS, etc.  
✅ **Seguridad**: Integra grupos de seguridad, IAM y cifrado para proteger las instancias.

### 🔹 **2. Tipos de Instancias EC2**  
EC2 ofrece distintos tipos de instancias optimizadas para diferentes usos:  

📌 **General Purpose** (Equilibradas) – Ejemplo: `t3.micro`, `m5.large`.  
📌 **Compute Optimized** (Procesamiento intensivo) – Ejemplo: `c5.large`, `c6g.2xlarge`.  
📌 **Memory Optimized** (Carga de memoria alta) – Ejemplo: `r5.large`, `x1.32xlarge`.  
📌 **Storage Optimized** (Alto rendimiento de disco) – Ejemplo: `i3.large`, `d2.8xlarge`.  
📌 **Accelerated Computing** (GPU o FPGAs) – Ejemplo: `p3.2xlarge`, `g4dn.xlarge`.

### 🔹 **3. Modelos de Pago**  

💲 **On-Demand**: Pago por hora/segundo sin compromisos.  
💲 **Reserved Instances**: Descuento a cambio de un compromiso a 1-3 años.  
💲 **Spot Instances**: Hasta 90% de descuento, pero pueden ser terminadas si hay mayor demanda.  
💲 **Dedicated Hosts**: Servidores físicos exclusivos para cumplimiento de normativas.

### 🔹 **4. Componentes Claves de EC2**  

✅ **AMI (Amazon Machine Image)**: Imagen del sistema operativo y software preinstalado.  
✅ **Instancias**: Máquinas virtuales en la nube.  
✅ **EBS (Elastic Block Store)**: Almacenamiento persistente para las instancias.  
✅ **Security Groups**: Firewall para controlar el tráfico entrante y saliente.  
✅ **Key Pairs**: Claves SSH para acceso seguro a las instancias.  
✅ **Elastic IPs**: Dirección IP fija para una instancia.  
✅ **Auto Scaling**: Ajusta automáticamente el número de instancias según la demanda.

### 🔹 **5. Pasos para Crear una Instancia EC2**  

1️⃣ **Acceder a AWS EC2**: Inicia sesión en AWS y ve a **EC2 Dashboard**.  
2️⃣ **Elegir una AMI**: Selecciona el sistema operativo y configuración base.  
3️⃣ **Seleccionar Tipo de Instancia**: Escoge una según tus necesidades.  
4️⃣ **Configurar Instancia**: Define red, almacenamiento y otras opciones.  
5️⃣ **Agregar Almacenamiento (EBS)**: Define el tamaño del disco.  
6️⃣ **Configurar Seguridad**: Configura reglas de firewall en el Security Group.  
7️⃣ **Seleccionar Clave SSH**: Descarga el par de claves para acceso seguro.  
8️⃣ **Lanzar la Instancia** 🚀

### 🎯 **Conclusión**  

Amazon EC2 es un servicio flexible y potente para ejecutar servidores en la nube. Su escalabilidad, opciones de pago y variedad de instancias lo hacen ideal para cualquier tipo de carga de trabajo, desde aplicaciones web hasta procesamiento de datos intensivo.  

💡 **¿Necesitas optimizar costos o rendimiento en EC2? Podemos analizar juntos la mejor opción.** 😃

**scriptEC2.sh**

```
#!/bin/bash

# Use this for your user data (script from top to bottom)

# install httpd (Linux 2 version)

yum update -y

yum install -y httpd

systemctl start httpd

systemctl enable httpd

echo "<h1>Hello World from $(hostname -f)</h1>" > /var/www/html/index.html
```

## Lab: Configura tu grupo de Seguridad

### 🔐 **Configuración de un Grupo de Seguridad en Amazon EC2**  

Un **Grupo de Seguridad (Security Group)** en AWS EC2 actúa como un firewall que controla el tráfico de red hacia y desde una instancia.

### 📌 **Pasos para Configurar un Grupo de Seguridad en EC2**  

1️⃣ **Acceder a EC2 en AWS:**  
   - Inicia sesión en la consola de AWS.  
   - Ve al servicio **EC2**.  
   - En el menú lateral, selecciona **Grupos de seguridad**.  

2️⃣ **Crear un Nuevo Grupo de Seguridad:**  
   - Haz clic en **Crear grupo de seguridad**.  
   - Asigna un **nombre** y una **descripción**.  
   - Selecciona la **VPC** donde aplicará el grupo de seguridad.  

3️⃣ **Configurar las Reglas de Entrada (Inbound Rules):**  
   - **SSH (22/tcp)**: Para acceso remoto vía terminal (*solo permite tu IP*).  
   - **HTTP (80/tcp)**: Para tráfico web si usas un servidor web.  
   - **HTTPS (443/tcp)**: Para tráfico seguro en aplicaciones web.  
   - **RDP (3389/tcp)**: Si usas Windows Server en la instancia.  

4️⃣ **Configurar las Reglas de Salida (Outbound Rules):**  
   - Por defecto, todas las conexiones salientes están permitidas.  
   - Puedes restringir puertos si es necesario.  

5️⃣ **Guardar y Asociar el Grupo de Seguridad:**  
   - Guarda el grupo de seguridad.  
   - Ve a tu instancia EC2 y asígnalo en la configuración de **Red y Seguridad**.

### ✅ **Mejores Prácticas**  
🔹 **Restringe accesos**: Evita abrir **SSH (22) o RDP (3389) a “0.0.0.0/0”** (todo el mundo).  
🔹 **Usa IPs específicas**: Limita el acceso SSH solo a tu IP pública.  
🔹 **Grupos separados**: Usa diferentes grupos para cada tipo de aplicación.  
🔹 **Monitorea actividad**: Revisa logs en **AWS CloudWatch** para detectar accesos sospechosos.  

Con esto, tu **instancia EC2 estará protegida** y solo permitirá el tráfico necesario. 🚀

## Tipos de instancias en EC2

Amazon EC2 ofrece diferentes tipos de instancias optimizadas para diversos casos de uso. Cada tipo de instancia tiene características específicas en términos de CPU, memoria, almacenamiento y capacidad de red. 

### 🔹 **1. Instancias de Propósito General**  
📌 **Uso:** Aplicaciones web, servidores pequeños, bases de datos de tamaño moderado.  
📌 **Ejemplos:**  
- **t4g, t3, t2** → Bajo costo, uso flexible (ideal para pruebas o aplicaciones pequeñas).  
- **m7g, m6i, m5, m4** → Equilibrio entre CPU, RAM y almacenamiento.

### 🔹 **2. Instancias Optimizadas para Cómputo**  
📌 **Uso:** Aplicaciones con alta carga de procesamiento, simulaciones científicas, gaming.  
📌 **Ejemplos:**  
- **c7g, c6i, c5, c4** → CPU de alto rendimiento, menos memoria.

### 🔹 **3. Instancias Optimizadas para Memoria**  
📌 **Uso:** Bases de datos en memoria, análisis de datos, grandes aplicaciones empresariales.  
📌 **Ejemplos:**  
- **r7g, r6i, r5, r4** → Alta capacidad de RAM para procesamiento intensivo de datos.  
- **x2idn, x1e, x1** → Aún más memoria para cargas extremas.

### 🔹 **4. Instancias Optimizadas para Almacenamiento**  
📌 **Uso:** Big Data, bases de datos NoSQL, sistemas de archivos distribuidos.  
📌 **Ejemplos:**  
- **i4i, i3, i2** → Almacenamiento SSD de baja latencia.  
- **d2, h1** → Alta capacidad en discos duros (HDD).

### 🔹 **5. Instancias Optimizadas para GPU (Machine Learning y Videojuegos)**  
📌 **Uso:** Machine Learning, IA, Renderizado 3D, Streaming de juegos.  
📌 **Ejemplos:**  
- **p4d, p3, p2** → GPU NVIDIA para Deep Learning y AI.  
- **g5, g4dn** → GPU NVIDIA para renderizado y streaming.

### 🔹 **6. Instancias de Alto Rendimiento en Red**  
📌 **Uso:** Sistemas financieros, trading de alta frecuencia, redes 5G.  
📌 **Ejemplos:**  
- **u-6tb1, u-9tb1, u-12tb1** → Instancias con hasta **12 TB de RAM**.  
- **m6idn, c6gn** → Alta capacidad de red y almacenamiento.

### ✅ **¿Cómo elegir la mejor instancia EC2?**  
1️⃣ **Si buscas un balance entre rendimiento y costo:** *m5, t3.*  
2️⃣ **Si necesitas más CPU:** *c6i, c5.*  
3️⃣ **Si trabajas con grandes bases de datos:** *r6i, x1e.*  
4️⃣ **Si usas Machine Learning o gráficos avanzados:** *g5, p4d.*  
5️⃣ **Si almacenas grandes volúmenes de datos:** *i3, d2.*  

Cada tipo de instancia está diseñado para diferentes necesidades.

**Lecturas recomendadas**

[Tipos de instancias de Amazon EC2 - Amazon Web Services](https://aws.amazon.com/es/ec2/instance-types/)

## Grupos de seguridad y puertos clásicos

Los **grupos de seguridad** en AWS EC2 actúan como un firewall virtual que controla el tráfico de entrada y salida de las instancias. Para configurar correctamente un grupo de seguridad, es fundamental conocer los **puertos clásicos** utilizados en diferentes aplicaciones.

### 🔹 **¿Qué es un Grupo de Seguridad en AWS?**  
- Es un conjunto de reglas que permiten o bloquean tráfico basado en **direcciones IP, protocolos y puertos**.  
- Se pueden aplicar a una o varias instancias EC2.  
- Controlan tanto **entrada (Inbound)** como **salida (Outbound)**.  
- Son **stateful**, lo que significa que si se permite un tráfico de entrada, la respuesta se permite automáticamente.

### 🔹 **Puertos Clásicos y su Uso**  

| **Puerto** | **Protocolo** | **Uso Común** |
|------------|--------------|---------------|
| **22** | TCP | SSH (Acceso remoto a servidores Linux) |
| **80** | TCP | HTTP (Tráfico web sin cifrar) |
| **443** | TCP | HTTPS (Tráfico web cifrado con SSL/TLS) |
| **3306** | TCP | MySQL (Base de datos relacional) |
| **5432** | TCP | PostgreSQL (Base de datos relacional) |
| **1433** | TCP | Microsoft SQL Server (Base de datos relacional) |
| **3389** | TCP | RDP (Acceso remoto a Windows) |
| **6379** | TCP | Redis (Base de datos en memoria) |
| **9200** | TCP | Elasticsearch (Búsquedas y analítica) |
| **27017** | TCP | MongoDB (Base de datos NoSQL) |

### 🔹 **Mejores Prácticas de Seguridad**  
✅ **Regla del Mínimo Privilegio**: Solo abrir los puertos estrictamente necesarios.  
✅ **Restringir IPs**: No permitir acceso global (`0.0.0.0/0`) a puertos críticos.  
✅ **Usar VPN o Bastion Host**: Para evitar exposición directa de servicios sensibles como SSH o RDP.  
✅ **Habilitar HTTPS en lugar de HTTP**: Para proteger la comunicación web.

🔐 **¡Configura tus grupos de seguridad con cuidado para mantener tus instancias protegidas!** 🚀

## Lab: Crea nuevos grupos de seguridad para tu instancia

Para proteger tu instancia de EC2, debes crear y configurar un **Grupo de Seguridad** con las reglas adecuadas.

### 🔹 **Pasos para Crear un Grupo de Seguridad**  

### **1️⃣ Accede a la Consola de AWS**  
- Inicia sesión en [AWS Management Console](https://aws.amazon.com/console/).  
- Ve al servicio **EC2**.  
- En el menú lateral, selecciona **Security Groups**.  

### **2️⃣ Crear un Nuevo Grupo de Seguridad**  
- Haz clic en **Create Security Group**.  
- **Asigna un nombre** (Ejemplo: `sg-web-server`).  
- **Agrega una descripción** (Ejemplo: "Grupo de seguridad para servidores web").  
- **Selecciona la VPC** donde se aplicará el grupo.  

### **3️⃣ Configurar Reglas de Entrada (Inbound Rules)**  
Aquí defines qué tráfico puede entrar a tu instancia.  

Ejemplo para un servidor web:  

| **Tipo** | **Protocolo** | **Puerto** | **Origen** | **Descripción** |
|----------|--------------|------------|------------|-----------------|
| SSH | TCP | 22 | Tu IP (`xx.xx.xx.xx/32`) | Acceso seguro vía SSH |
| HTTP | TCP | 80 | `0.0.0.0/0` | Permitir tráfico web sin cifrar |
| HTTPS | TCP | 443 | `0.0.0.0/0` | Permitir tráfico web cifrado |

🔹 **Recomendación:** Restringe el acceso SSH solo a tu IP en lugar de abrirlo a todos (`0.0.0.0/0`).  

### **4️⃣ Configurar Reglas de Salida (Outbound Rules)**  
Por defecto, AWS permite **todo el tráfico saliente**. Puedes dejar la configuración predeterminada.  

### **5️⃣ Asociar el Grupo de Seguridad a tu Instancia**  
- Ve a **EC2 > Instances**.  
- Selecciona tu instancia.  
- Haz clic en **Actions > Networking > Change Security Groups**.  
- Selecciona el nuevo grupo de seguridad y confirma los cambios.

## ¿Qué es SSH?

**SSH (Secure Shell)** es un protocolo de red que permite la comunicación segura entre dos dispositivos a través de una red no confiable, como Internet. Se usa principalmente para administrar servidores y computadoras de forma remota mediante una conexión encriptada.

### 🔑 **Características principales de SSH**  

✅ **Cifrado seguro**: Protege la comunicación para evitar ataques como "Man-in-the-Middle".  
✅ **Autenticación con claves**: Permite iniciar sesión sin contraseña mediante **claves SSH**.  
✅ **Túneles SSH**: Puedes redirigir tráfico de otras aplicaciones de forma segura.  
✅ **Transferencia de archivos**: Usa comandos como `scp` o `sftp` para enviar archivos de manera segura.

### 📌 **Cómo conectar a un servidor con SSH**  

Si tienes una instancia en AWS EC2 o cualquier servidor remoto, puedes acceder con:  

🔹 **Desde Linux/macOS**:  
```bash
ssh -i "tu-clave.pem" usuario@IP-del-servidor
```
Ejemplo para AWS:  
```bash
ssh -i "mi-clave-aws.pem" ec2-user@54.123.45.67
```

🔹 **Desde Windows**:  
- Puedes usar **PuTTY** o el terminal de Windows con `ssh`.

### 🔐 **Autenticación con Claves SSH**  

En lugar de usar contraseñas, puedes autenticarte con claves públicas y privadas:  
1️⃣ **Generar claves SSH** (en tu máquina local):  
```bash
ssh-keygen -t rsa -b 4096 -C "tu-email@example.com"
```
2️⃣ **Copiar la clave pública al servidor**:  
```bash
ssh-copy-id usuario@IP-del-servidor
```
3️⃣ **Conectar sin contraseña**:  
```bash
ssh usuario@IP-del-servidor
```

### 📌 **Comandos básicos en SSH**  

| **Comando** | **Descripción** |
|------------|----------------|
| `ssh usuario@servidor` | Conectarse a un servidor remoto. |
| `exit` | Cerrar la sesión SSH. |
| `scp archivo.txt usuario@servidor:/ruta/` | Enviar un archivo con SSH. |
| `sftp usuario@servidor` | Transferir archivos con SFTP. |

### 🚀 **Resumen**  
🔹 SSH permite acceso remoto seguro a servidores.  
🔹 Usa claves SSH para evitar contraseñas.  
🔹 Puedes transferir archivos y ejecutar comandos de forma segura.  

📌 **¡Esencial para administrar servidores en la nube como AWS, GCP y Azure!** 💻🔒

## Lab: Cómo usar ssh en mac/linux

SSH (Secure Shell) en **Mac y Linux** está integrado en la terminal, lo que facilita la conexión remota a servidores o instancias en la nube.

### 🔑 **1️⃣ Conectar a un servidor con SSH**  

🔹 **Sintaxis básica**:  
```bash
ssh usuario@IP-del-servidor
```
Ejemplo:  
```bash
ssh ec2-user@54.123.45.67
```

🔹 **Si usas una clave privada (.pem o .ppk)**:  
```bash
ssh -i "mi-clave.pem" usuario@IP-del-servidor
```
Ejemplo para AWS:  
```bash
ssh -i "mi-clave-aws.pem" ec2-user@54.123.45.67
```

### 🔐 **2️⃣ Usar autenticación con claves SSH**  

Para evitar el uso de contraseñas en cada conexión, puedes configurar una **clave SSH** en tu máquina local y copiarla al servidor.  

🔹 **Generar una clave SSH (si no tienes una)**  
```bash
ssh-keygen -t rsa -b 4096 -C "tu-email@example.com"
```
🔹 **Copiar la clave al servidor (si tienes acceso con contraseña)**  
```bash
ssh-copy-id usuario@IP-del-servidor
```

🔹 **Si no puedes usar `ssh-copy-id`, hazlo manualmente**  
```bash
cat ~/.ssh/id_rsa.pub | ssh usuario@IP-del-servidor "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

Ahora puedes conectarte sin contraseña con:  
```bash
ssh usuario@IP-del-servidor
```

### 📂 **3️⃣ Transferir archivos con SSH (SCP y SFTP)**  

🔹 **Enviar un archivo con `scp`**:  
```bash
scp -i "mi-clave.pem" archivo.txt usuario@IP-del-servidor:/ruta/destino/
```

🔹 **Descargar un archivo desde el servidor**:  
```bash
scp usuario@IP-del-servidor:/ruta/archivo.txt .
```

🔹 **Usar SFTP para administrar archivos**:  
```bash
sftp usuario@IP-del-servidor
```
Dentro de SFTP:  
```bash
put archivo.txt  # Subir archivo  
get archivo.txt  # Descargar archivo  
```

### ⚙ **4️⃣ Configurar SSH para evitar repetir comandos**  

Si te conectas con frecuencia, puedes agregar la configuración en `~/.ssh/config`:  

```bash
Host mi-servidor
    HostName 54.123.45.67
    User ec2-user
    IdentityFile ~/.ssh/mi-clave.pem
```

Ahora solo necesitas ejecutar:  
```bash
ssh mi-servidor
```

Para dar permisos:

```shell
chmod prueba.pem
```

y luego

```shell
ssh -i "prueba.pem" ec2-user@ec2-54-236-237-225.compute-1.amazonaws.com
```

### 🚀 **Resumen**  
✅ **SSH** permite conexión remota segura en Mac/Linux.  
✅ **Usa claves SSH** para evitar contraseñas repetitivas.  
✅ **Transfiere archivos con SCP o SFTP** fácilmente.  
✅ **Configura `~/.ssh/config`** para simplificar conexiones.  

📌 **¡Fundamental para administrar servidores en la nube como AWS, GCP y Azure!** 💻🔒

## Cómo usar ssh utilizando windows

En **Windows**, puedes usar SSH para conectarte a servidores remotos de varias formas. Aquí te explico cómo hacerlo usando:  

1️⃣ **PowerShell o Símbolo del sistema (CMD)** 🖥️  
2️⃣ **PuTTY (para configuraciones avanzadas)** 🛠️ 

### 🔑 **1️⃣ Conectar a un servidor SSH desde PowerShell o CMD**  

Desde **Windows 10/11**, SSH ya está integrado en **PowerShell** y **CMD**, por lo que puedes conectarte fácilmente:  

🔹 **Abrir PowerShell o CMD y ejecutar:**  
```powershell
ssh usuario@IP-del-servidor
```
Ejemplo:  
```powershell
ssh ec2-user@54.123.45.67
```

🔹 **Si necesitas una clave privada (.pem):**  
```powershell
ssh -i "C:\ruta\mi-clave.pem" usuario@IP-del-servidor
```
Ejemplo en AWS:  
```powershell
ssh -i "C:\Users\TuUsuario\Downloads\mi-clave.pem" ec2-user@54.123.45.67
```

### 🔐 **2️⃣ Configurar autenticación con claves SSH**  

Si usas claves SSH, guárdalas en `C:\Users\TuUsuario\.ssh\` y luego agrégala manualmente al **agente SSH** con:  
```powershell
ssh-add C:\Users\TuUsuario\.ssh\mi-clave.pem
```
Así, no tendrás que escribir la ruta cada vez.

### 📂 **3️⃣ Transferir archivos con SSH (SCP y SFTP)**  

🔹 **Subir un archivo al servidor con SCP:**  
```powershell
scp -i "C:\ruta\mi-clave.pem" archivo.txt usuario@IP:/ruta/destino/
```

🔹 **Descargar un archivo del servidor:**  
```powershell
scp usuario@IP:/ruta/archivo.txt C:\ruta\destino\
```

🔹 **Conectar con SFTP:**  
```powershell
sftp usuario@IP
```
Y luego usar comandos como:  
```powershell
put archivo.txt  # Subir archivo  
get archivo.txt  # Descargar archivo  
```

### 🔷 **4️⃣ Usar PuTTY (Alternativa con interfaz gráfica)**  

Si prefieres una interfaz gráfica, usa **PuTTY**:  

🔹 **Descargar**: [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)  

🔹 **Abrir PuTTY y configurar**:  
1️⃣ En **Host Name**, ingresa `usuario@IP-del-servidor`.  
2️⃣ En **Connection type**, selecciona `SSH`.  
3️⃣ Si usas una clave `.pem`, conviértela a `.ppk` con `PuTTYgen`.  
4️⃣ Ve a **SSH > Auth** y carga la clave `.ppk`.  
5️⃣ **Haz clic en Open** para conectarte.  

### ⚙ **5️⃣ Configurar SSH en Windows para facilitar conexiones**  

Si te conectas frecuentemente, agrega esta configuración en:  
🔹 **`C:\Users\TuUsuario\.ssh\config`**  

```plaintext
Host mi-servidor
    HostName 54.123.45.67
    User ec2-user
    IdentityFile C:\Users\TuUsuario\.ssh\mi-clave.pem
```

Ahora solo necesitas escribir:  
```powershell
ssh mi-servidor
```

### 🚀 **Resumen**  
✅ **Windows 10/11 ya tiene SSH en PowerShell y CMD.**  
✅ **Puedes conectarte con `ssh usuario@IP`.**  
✅ **Usa SCP o SFTP para transferir archivos.**  
✅ **PuTTY es una alternativa con interfaz gráfica.**  
✅ **Configura `~/.ssh/config` para conexiones rápidas.**  

📌 **¡Con esto puedes administrar servidores en AWS, Azure y más desde Windows!** 💻🔒

**Lecturas recomendadas**

[Download PuTTY - a free SSH and telnet client for Windows](https://putty.org/)

## Lab: Cómo usar ssh utilizando power shell

Para usar **SSH en PowerShell** en **Windows**, sigue estos pasos:

### ✅ **1. Verifica que SSH está instalado**
Desde PowerShell, ejecuta:
```powershell
Get-Service -Name ssh-agent
```
Si aparece el servicio, significa que **SSH está instalado**. Si no, instala **OpenSSH** desde "Características opcionales" de Windows.

### ✅ **2. Conectarse a un servidor remoto**  
Usa el siguiente comando:  
```powershell
ssh usuario@ip_o_hostname
```
🔹 **Ejemplo:**  
```powershell
ssh admin@192.168.1.100
```
Esto intentará conectarte al servidor con el usuario `admin`.

### ✅ **3. Usar una clave SSH en lugar de contraseña**  
Si tienes una **clave privada** en tu máquina local, puedes conectarte sin escribir la contraseña:  
```powershell
ssh -i C:\ruta\clave.pem usuario@ip_o_hostname
```
🔹 **Ejemplo:**  
```powershell
ssh -i C:\Users\Usuario\.ssh\id_rsa admin@192.168.1.100
```

### ✅ **4. Cerrar sesión SSH**
Para desconectarte, usa:
```powershell
exit
```
O presiona **`Ctrl + D`**.

### 🎯 **Resumen rápido**  
✅ **Conectar:** `ssh usuario@ip`  
✅ **Usar clave SSH:** `ssh -i C:\ruta\clave.pem usuario@ip`  
✅ **Salir:** `exit` o `Ctrl + D`  

📌 ¡Ahora puedes usar **SSH en PowerShell** como un pro! 🚀

## Lab: EC2 Instance Connect

### ✅ **¿Qué es EC2 Instance Connect?**  
EC2 Instance Connect es una función de AWS que te permite acceder a **instancias EC2** de Amazon Linux o Ubuntu **directamente desde la consola web de AWS**, sin necesidad de una clave SSH o cliente externo.

### 🔹 **¿Cuándo usar EC2 Instance Connect?**  
✔️ Cuando no tienes una clave SSH configurada.  
✔️ Para acceder rápidamente a una instancia sin instalar un cliente SSH.  
✔️ Para solucionar problemas de conectividad en instancias sin acceso remoto.

### 🚀 **Cómo usar EC2 Instance Connect**  
Sigue estos pasos para conectarte a una instancia EC2:

#### **1️⃣ Accede a la consola de AWS**  
- Ve a [AWS Console](https://aws.amazon.com/console/)
- Dirígete a **EC2** > **Instancias**.

#### **2️⃣ Selecciona tu instancia**  
- Busca la instancia EC2 a la que quieres conectarte.
- Asegúrate de que ejecuta **Amazon Linux** o **Ubuntu** (Instance Connect **no funciona en Windows ni en otras distros**).

#### **3️⃣ Conéctate desde la consola**  
- Haz clic en **Connect**.
- Ve a la pestaña **EC2 Instance Connect**.
- Haz clic en **Connect** y se abrirá una terminal en el navegador.

### 🎯 **Ventajas de EC2 Instance Connect**  
✅ No necesitas configurar claves SSH.  
✅ Acceso rápido y seguro desde el navegador.  
✅ No requiere instalación de software adicional.  
✅ Permite acceso temporal sin modificar la configuración de seguridad.

### 📌 **Alternativa: Conectarse con SSH**  
Si tu instancia **no es compatible** con Instance Connect, usa:  
```sh
ssh -i "clave.pem" usuario@ip_publica
```
🔹 Ejemplo para Amazon Linux:  
```sh
ssh -i "mi-clave.pem" ec2-user@34.215.10.123
```

📌 **¡Ahora ya sabes cómo acceder a tu instancia EC2 de forma rápida y sencilla! 🚀**

## Lab: EC2 Instance Roles

### ✅ **¿Qué son los EC2 Instance Roles?**  
Los **EC2 Instance Roles** en AWS permiten asignar **permisos temporales** a una instancia EC2 **sin necesidad de credenciales estáticas**. Esto se hace mediante **AWS Identity and Access Management (IAM)**.

### 🔹 **¿Por qué usar EC2 Instance Roles?**  
✔️ **Evita almacenar claves de acceso** en la instancia.  
✔️ **Automatiza la autenticación** con otros servicios de AWS.  
✔️ **Mejora la seguridad** al gestionar permisos centralmente en IAM.  
✔️ **Facilita el acceso** a servicios como S3, DynamoDB, CloudWatch, etc.

### 🚀 **Cómo crear y asignar un EC2 Instance Role**  

#### **1️⃣ Crear el IAM Role**  
1. En la **consola de AWS**, ve a **IAM** > **Roles**.  
2. Haz clic en **Crear rol**.  
3. En "Tipo de entidad de confianza", elige **AWS Service**.  
4. Selecciona **EC2** como servicio que usará el rol.  
5. **Adjunta permisos** según las necesidades de la instancia:  
   - Para acceder a S3: **AmazonS3ReadOnlyAccess**  
   - Para acceder a DynamoDB: **AmazonDynamoDBFullAccess**  
   - Para escribir en CloudWatch: **CloudWatchAgentServerPolicy**  
6. Asigna un **nombre al rol** y crea el rol.

#### **2️⃣ Asignar el Role a una Instancia EC2**  
1. Ve a **EC2** > **Instancias**.  
2. Selecciona la instancia.  
3. En la pestaña **Acciones**, selecciona:  
   `Security > Modify IAM Role`.  
4. Elige el **rol IAM** creado y guárdalo.

### 🎯 **Cómo verificar que el rol funciona**  
📌 **Desde la instancia EC2, usa este comando:**  
```sh
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
```
🔹 Si todo está bien, verás el nombre del **rol IAM** asignado.  

📌 **Para listar las credenciales temporales:**  
```sh
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/NOMBRE_DEL_ROL
```

### 📌 **Ejemplo de uso con AWS CLI**  
Si tu instancia tiene un rol con acceso a S3, puedes ejecutar:  
```sh
aws s3 ls
```
✅ Si el rol tiene permisos correctos, listará los buckets S3 sin necesidad de configurar credenciales.

### 🔥 **Conclusión**  
EC2 Instance Roles son **clave para mejorar la seguridad** y **automatizar el acceso** a otros servicios de AWS **sin manejar credenciales manualmente**. 🚀

## Lab: Limpieza de recursos

### 🧹 **Limpieza de Recursos en AWS**  
Para evitar **costos innecesarios** en AWS, es fundamental eliminar los recursos no utilizados. Aquí tienes una guía para limpiar los más comunes. 🚀 

### 🔹 **1️⃣ Revisar Recursos Activos**  
Antes de eliminar, revisa qué servicios están consumiendo recursos:  

📌 **Con AWS Console:**  
- Ve a **Billing Dashboard** > **Cost & Usage Reports**.  
- Usa **AWS Resource Groups** para ver recursos activos.  

📌 **Con AWS CLI:**  
```sh
aws resourcegroupstaggingapi get-resources
```

### 🔹 **2️⃣ Eliminar Instancias EC2**  
Para evitar cargos por instancias en ejecución:  

📌 **Desde la Consola AWS:**  
1. Ve a **EC2** > **Instancias**.  
2. Selecciona la instancia y haz clic en **Actions** > **Terminate Instance**.  
3. Confirma la eliminación.  

📌 **Desde AWS CLI:**  
```sh
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxx
```

### 🔹 **3️⃣ Eliminar Volúmenes EBS**  
Después de eliminar EC2, los volúmenes pueden quedar **huérfanos**.  

📌 **Desde AWS Console:**  
- Ve a **EC2** > **Elastic Block Store (EBS)**.  
- Busca volúmenes **no adjuntos** y elimínalos.  

📌 **Desde AWS CLI:**  
```sh
aws ec2 delete-volume --volume-id vol-xxxxxxxxxxxx
```

### 🔹 **4️⃣ Limpiar S3 Buckets**  
Si tienes almacenamiento en **S3**, vacíalo o elimínalo.  

📌 **Desde AWS Console:**  
1. Ve a **S3** > Selecciona el bucket.  
2. Vacía su contenido antes de eliminarlo.  

📌 **Desde AWS CLI:**  
```sh
aws s3 rb s3://nombre-del-bucket --force
```
(El `--force` elimina el contenido antes de borrar el bucket).

### 🔹 **5️⃣ Eliminar Bases de Datos RDS**  
Las bases de datos siguen generando costos si no se eliminan.  

📌 **Desde AWS Console:**  
1. Ve a **RDS** > **Bases de datos**.  
2. Selecciona la instancia y haz clic en **Eliminar**.  
3. Decide si quieres **hacer backup** antes de borrarla.  

📌 **Desde AWS CLI:**  
```sh
aws rds delete-db-instance --db-instance-identifier nombre-db
```

### 🔹 **6️⃣ Eliminar Clústeres Redshift**  
📌 **Desde AWS Console:**  
- Ve a **Redshift** > **Clústeres** > **Eliminar**.  

📌 **Desde AWS CLI:**  
```sh
aws redshift delete-cluster --cluster-identifier nombre-cluster --skip-final-cluster-snapshot
```

### 🔹 **7️⃣ Liberar Direcciones IP elásticas**  
Si no liberas IPs estáticas, AWS las sigue cobrando.  

📌 **Desde AWS Console:**  
1. Ve a **EC2** > **Elastic IPs**.  
2. Selecciona y haz clic en **Release Address**.  

📌 **Desde AWS CLI:**  
```sh
aws ec2 release-address --allocation-id eipalloc-xxxxxxxx
```

### 🔹 **8️⃣ Revisar Cargos en AWS Billing**  
- Ve a **Billing Dashboard** y revisa **cargos pendientes**.  
- Configura un **presupuesto en AWS Budgets** para evitar sorpresas.  

### 🚀 **Conclusión**  
✅ **Elimina recursos innecesarios** regularmente para evitar costos.  
✅ **Usa AWS Budgets** para monitorear gastos.  
✅ **Automatiza la limpieza** con scripts AWS Lambda o AWS CLI.  

**¡Así optimizas costos y evitas sorpresas en tu factura de AWS!** 🔥

## Opciones de compra de instancias

### 🛒 **Opciones de Compra de Instancias EC2 en AWS**  
AWS ofrece diferentes opciones de compra para adaptarse a distintas necesidades y presupuestos. Aquí tienes un resumen de cada opción:  

### 🔹 **1️⃣ On-Demand Instances (Instancias Bajo Demanda)**
**📌 Ideal para:** Uso flexible sin compromisos a largo plazo.  
**💰 Costo:** Tarifa por segundo o por hora, según el tipo de instancia.  

✅ **Ventajas:**  
✔ Sin compromiso a largo plazo.  
✔ Escalabilidad instantánea.  
✔ Pago solo por lo que usas.  

❌ **Desventajas:**  
❌ Más costoso en comparación con otras opciones a largo plazo.  

📌 **Ejemplo de uso:** Aplicaciones con tráfico variable o pruebas.  

```sh
aws ec2 run-instances --image-id ami-xxxxxxxx --instance-type t2.micro --count 1
```

### 🔹 **2️⃣ Reserved Instances (Instancias Reservadas)**
**📌 Ideal para:** Cargas de trabajo constantes y previsibles.  
**💰 Costo:** Hasta **75% de descuento** en comparación con On-Demand.  

✅ **Ventajas:**  
✔ Costos más bajos con compromisos de 1 a 3 años.  
✔ Opción de pago total, parcial o mensual.  

❌ **Desventajas:**  
❌ Falta de flexibilidad, ya que requiere un compromiso de tiempo.  

📌 **Ejemplo de uso:** Servidores de bases de datos o aplicaciones de producción.  

```sh
aws ec2 purchase-reserved-instances-offering --reserved-instances-offering-id xxxx --instance-count 1
```

### 🔹 **3️⃣ Spot Instances (Instancias Spot)**
**📌 Ideal para:** Procesos no críticos y tareas escalables.  
**💰 Costo:** Hasta **90% más barato** que On-Demand.  

✅ **Ventajas:**  
✔ Súper económico para cargas de trabajo flexibles.  
✔ Escalabilidad masiva a bajo costo.  

❌ **Desventajas:**  
❌ AWS puede recuperar la instancia si el precio sube.  
❌ No recomendable para cargas críticas o de larga duración.  

📌 **Ejemplo de uso:** Procesamiento por lotes, Big Data, Machine Learning.  

```sh
aws ec2 request-spot-instances --spot-price "0.03" --instance-count 2 --launch-specification file://config.json
```

### 🔹 **4️⃣ Savings Plans (Planes de Ahorro)**
**📌 Ideal para:** Empresas que desean descuentos sin restricciones de instancia.  
**💰 Costo:** Hasta **72% de ahorro** comparado con On-Demand.  

✅ **Ventajas:**  
✔ Más flexibilidad que Reserved Instances.  
✔ Aplica a cualquier instancia en la misma familia.  
✔ Opción de 1 o 3 años de compromiso.  

❌ **Desventajas:**  
❌ Requiere compromiso de pago a largo plazo.  

📌 **Ejemplo de uso:** Empresas con uso constante de instancias EC2.  

### 🔹 **5️⃣ Dedicated Hosts (Hosts Dedicados)**
**📌 Ideal para:** Cumplimiento de normativas y licencias específicas.  
**💰 Costo:** Más caro, pero permite el uso de licencias propias.  

✅ **Ventajas:**  
✔ Servidor físico dedicado solo para ti.  
✔ Cumple con regulaciones de seguridad y auditoría.  
✔ Optimización de licencias de software (BYOL - Bring Your Own License).  

❌ **Desventajas:**  
❌ Costo elevado comparado con otras opciones.  
❌ No es escalable dinámicamente como las otras opciones.  

📌 **Ejemplo de uso:** Entornos financieros, gubernamentales o con requisitos de seguridad estrictos.  

```sh
aws ec2 allocate-hosts --instance-type c5.large --host-recovery on --quantity 1
```

### 🚀 **Conclusión**  
| **Opción**             | **Costo**    | **Compromiso**  | **Casos de uso** |
|------------------------|-------------|----------------|------------------|
| **On-Demand**         | Alto        | Ninguno       | Aplicaciones flexibles |
| **Reserved**          | Bajo        | 1-3 años      | Producción estable |
| **Spot**              | Muy bajo    | Sin garantía  | Procesos no críticos |
| **Savings Plan**      | Bajo        | 1-3 años      | Empresas con uso predecible |
| **Dedicated Host**    | Alto        | Largo plazo   | Cumplimiento de normativas |

Si **necesitas flexibilidad**, usa **On-Demand**.  
Si **tienes cargas predecibles**, opta por **Reserved Instances o Savings Plans**.  
Si buscas **la opción más económica**, considera **Spot Instances**.  

## Modelo de responsabilidad compartida para EC2

AWS utiliza un **modelo de responsabilidad compartida**, donde AWS y el cliente tienen roles específicos para garantizar la seguridad y administración de los recursos.

### 🏢 **Responsabilidad de AWS (Seguridad de la Nube)**  
AWS es responsable de la infraestructura subyacente que soporta EC2, asegurando su disponibilidad y seguridad física.  

✅ **Lo que AWS gestiona:**  
✔ Seguridad física de los centros de datos.  
✔ Mantenimiento del hardware de servidores.  
✔ Red y virtualización de instancias.  
✔ Parches y actualizaciones de la infraestructura de AWS.  

💡 **Ejemplo:** Si hay una falla en el hardware de un servidor, AWS se encarga de solucionarla.

### 🧑‍💻 **Responsabilidad del Cliente (Seguridad en la Nube)**  
El cliente es responsable de la configuración y gestión de sus instancias EC2.  

✅ **Lo que el cliente gestiona:**  
✔ Configuración del sistema operativo en EC2.  
✔ Administración de accesos y credenciales.  
✔ Configuración de firewalls y reglas de seguridad (Grupos de Seguridad).  
✔ Cifrado de datos en tránsito y en reposo.  
✔ Instalación de parches y actualizaciones en el sistema operativo.  

💡 **Ejemplo:** Si configuras un grupo de seguridad que permite acceso público por SSH (puerto 22), es tu responsabilidad asegurarte de que esté correctamente protegido.

### 🔍 **Ejemplo Práctico de Responsabilidad Compartida en EC2**  

### 🛠 **Responsabilidad de AWS:**  
✅ AWS mantiene la infraestructura subyacente, como los servidores físicos.  
✅ AWS garantiza que la red y los hipervisores funcionen correctamente.  

### 👨‍💻 **Responsabilidad del Cliente:**  
✅ Configurar correctamente el acceso SSH para evitar vulnerabilidades.  
✅ Aplicar actualizaciones de seguridad al sistema operativo de la instancia.  
✅ Definir políticas de cifrado para los volúmenes EBS y los datos almacenados.

### 🔄 **Resumen: ¿Quién es responsable de qué?**  

| **Categoría**                     | **Responsabilidad de AWS**          | **Responsabilidad del Cliente**  |
|-----------------------------------|---------------------------------|--------------------------------|
| **Infraestructura Física**        | ✅ Seguridad de los Data Centers | ❌ No aplica |
| **Hardware de Servidores**        | ✅ Mantenimiento y actualizaciones | ❌ No aplica |
| **Red y Virtualización**          | ✅ Configuración y seguridad | ❌ No aplica |
| **Sistema Operativo en EC2**      | ❌ No aplica | ✅ Parches y actualizaciones |
| **Grupos de Seguridad y Firewalls** | ❌ No aplica | ✅ Configuración adecuada |
| **Cifrado de Datos**              | ✅ Opciones de cifrado | ✅ Implementación y gestión |
| **Control de Accesos**            | ❌ No aplica | ✅ IAM y permisos correctos |

### 🚀 **Buenas Prácticas de Seguridad en EC2**  
🔹 **Usar claves SSH seguras** y evitar accesos abiertos al público.  
🔹 **Configurar Grupos de Seguridad** para restringir el acceso por IP.  
🔹 **Mantener las instancias actualizadas** con parches de seguridad.  
🔹 **Activar logs y monitoreo** con CloudWatch y GuardDuty.  
🔹 **Cifrar datos sensibles** en volúmenes EBS y en tránsito.  

⚡ **Recuerda:** AWS proporciona las herramientas, pero tú eres responsable de configurarlas correctamente. 