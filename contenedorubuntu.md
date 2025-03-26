Para crear un **contenedor Docker** basado en **Ubuntu** con **Python 3**, sigue estos pasos:  

---

## **1️⃣ Crear y Ejecutar un Contenedor Ubuntu con Python 3**
Ejecuta el siguiente comando para lanzar un contenedor con **Ubuntu** e instalar **Python 3**:  
```bash
docker run -it --rm ubuntu bash
```
Esto abrirá una terminal dentro del contenedor. Ahora, instala Python:  
```bash
apt update && apt install -y python3 python3-pip
python3 --version
```

Si quieres **crear un contenedor preconfigurado** con Python 3, usa un `Dockerfile`.  

---

## **2️⃣ Crear un `Dockerfile` para un Contenedor Ubuntu con Python 3**
📌 Crea un archivo llamado `Dockerfile`:  
```dockerfile
# Usar Ubuntu como base
FROM ubuntu:latest

# Instalar dependencias y Python
RUN apt update && apt install -y python3 python3-pip

# Definir el comando por defecto
CMD ["python3"]
```

---

## **3️⃣ Construir la Imagen Docker**
Ejecuta este comando en la carpeta donde creaste el `Dockerfile`:  
```bash
docker build -t ubuntu-python3 .
```

---

## **4️⃣ Ejecutar un Contenedor desde la Imagen**
```bash
docker run -it --rm ubuntu-python3 bash
```

✅ ¡Listo! Ahora tienes un **contenedor Ubuntu con Python 3**. 🚀  
Si necesitas agregar más paquetes, dime. 😊


Para instalar **MySQL** en **Ubuntu**, sigue estos pasos:

---

### **1️⃣ Actualizar el sistema**
Abre una terminal y ejecuta:  
```bash
sudo apt update && sudo apt upgrade -y
```

---

### **2️⃣ Instalar MySQL Server**
```bash
sudo apt install -y mysql-server
```

---

### **3️⃣ Verificar el Estado del Servicio**
Después de la instalación, MySQL debería estar ejecutándose automáticamente. Confírmalo con:  
```bash
sudo systemctl status mysql
```
Si no está activo, inicia el servicio manualmente:  
```bash
sudo systemctl start mysql
```

---

### **4️⃣ Configurar MySQL (Opcional, pero Recomendado)**
Para mejorar la seguridad de MySQL, ejecuta:  
```bash
sudo mysql_secure_installation
```
Aquí podrás:
- Configurar una contraseña de root.
- Eliminar usuarios anónimos.
- Deshabilitar el acceso remoto a root.
- Eliminar la base de datos de prueba.

---

### **5️⃣ Acceder a MySQL**
```bash
sudo mysql -u root -p
```
Ingresa tu contraseña y ya podrás ejecutar comandos SQL.

---

### **6️⃣ Habilitar el Inicio Automático (Opcional)**
Si quieres que MySQL inicie automáticamente al arrancar Ubuntu:
```bash
sudo systemctl enable mysql
```

✅ **¡Listo!** Ahora tienes **MySQL instalado y listo para usar** en Ubuntu. 🚀

Para instalar **GitHub CLI (gh) o Git** en **Ubuntu**, sigue estos pasos:

---

## 🔹 **Opción 1: Instalar Git (Cliente de GitHub)**
Si solo necesitas usar Git para manejar repositorios, instala **Git**:

### **1️⃣ Actualizar el sistema**
```bash
sudo apt update && sudo apt upgrade -y
```

### **2️⃣ Instalar Git**
```bash
sudo apt install git -y
```

### **3️⃣ Verificar instalación**
```bash
git --version
```
Salida esperada:  
```
git version X.Y.Z
```

### **4️⃣ Configurar Git**
Reemplaza con tus datos de GitHub:
```bash
git config --global user.name "TuNombre"
git config --global user.email "tuemail@ejemplo.com"
```
Verifica la configuración con:
```bash
git config --list
```

---

## 🔹 **Opción 2: Instalar GitHub CLI (`gh`)**
Si deseas administrar repositorios, hacer PRs y gestionar GitHub desde la terminal:

### **1️⃣ Instalar GitHub CLI**
```bash
type -p curl >/dev/null || sudo apt install curl -y
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh -y
```

### **2️⃣ Verificar instalación**
```bash
gh --version
```

### **3️⃣ Iniciar sesión en GitHub**
```bash
gh auth login
```
Sigue las instrucciones en pantalla.

---

✅ **¡Listo!** Ahora tienes **Git** y **GitHub CLI** instalados en tu Ubuntu. 🚀

Dockerfile

```Dockerfile
# Usa la última imagen oficial de Ubuntu
FROM ubuntu:latest

# Evita prompts interactivos durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Actualiza el sistema y las dependencias necesarias
RUN apt update && apt upgrade -y \
    && apt install -y \
        python3 \
        python3-pip \
        mysql-server \
        mysql-client \
        git \
        curl \
        unzip \
        nano \
        vim \
    && rm -rf /var/lib/apt/lists/*

# Instalar GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Configuración para MySQL (opcional, solo si deseas iniciar MySQL en el contenedor)
RUN mkdir -p /var/run/mysqld && chown -R mysql:mysql /var/run/mysqld

# Copia un archivo de configuración de MySQL personalizado (opcional)
# COPY my.cnf /etc/mysql/my.cnf

# Definir el comando de inicio (útil si se quiere iniciar MySQL)
CMD ["/bin/bash"]
```

Aquí tienes un **Dockerfile** para crear un contenedor Ubuntu con **Python 3, MySQL, Git y GitHub CLI** instalado. También incluye las configuraciones necesarias para que todo funcione correctamente.

Este **Dockerfile** hace lo siguiente:

1. Usa Ubuntu como base.
2. Instala:
   - **Python 3** y **pip**.
   - **MySQL Server y Cliente**.
   - **Git** y **GitHub CLI** (`gh`).
   - Herramientas adicionales como `curl`, `unzip`, `nano`, y `vim`.
3. Configura MySQL creando el directorio `/var/run/mysqld` (necesario para que corra dentro del contenedor).
4. Incluye un comando de inicio (`CMD`) que mantiene la sesión abierta con `/bin/bash`.

### **Construcción y Ejecución**
Para construir la imagen:
```sh
docker build -t ubuntu-python-mysql-git .
```

Para ejecutar el contenedor en modo interactivo:
```sh
docker run -it --rm ubuntu-python-mysql-git
```
Si deseas mantener el contenedor en ejecución:
```sh
docker run -it -d ubuntu-python-mysql-git
```

Déjame saber si necesitas alguna modificación o ajustes. 🚀

Para crear un contenedor de **Kali Linux** con Docker, sigue estos pasos:

### **1️⃣ Instalar Docker** (si no lo tienes ya)  
En **Ubuntu/Debian**, ejecuta:
```sh
sudo apt update
sudo apt install docker.io -y
```
Para **Windows/Mac**, descarga **Docker Desktop** desde [aquí](https://www.docker.com/products/docker-desktop/).

---

### **2️⃣ Descargar y ejecutar Kali Linux en Docker**
Ejecuta este comando para descargar y correr un contenedor de Kali Linux:
```sh
docker run -it --rm kalilinux/kali-rolling /bin/bash
```
📌 **Explicación**:
- `docker run` → Ejecuta un nuevo contenedor.
- `-it` → Permite interacción con el terminal.
- `--rm` → Borra el contenedor cuando se detiene.
- `kalilinux/kali-rolling` → Imagen oficial de Kali Linux.
- `/bin/bash` → Inicia el contenedor con la shell de Bash.

---

### **3️⃣ Mantener el contenedor después de cerrar**
Si deseas que el contenedor **persista**, usa:
```sh
docker run -it --name kali_persist kalilinux/kali-rolling /bin/bash
```
Luego, para **reanudar** el contenedor:
```sh
docker start -ai kali_persist
```

---

### **4️⃣ Instalar herramientas adicionales en Kali**
Dentro del contenedor, puedes instalar herramientas como:
```sh
apt update && apt install -y metasploit-framework nmap
```
## Apache zepellin 

`docker run -p 8080:8080 --rm --name zeppelin apache/zeppelin:0.12.0`


## contenedor de kali linux

¡Claro! Vamos a desglosar el comando:  

```bash
docker run -it --rm --name kali_persist kalilinux/kali-rolling /bin/bash
```

### Explicación de cada parte:

1. **`docker run`**  
   - Ejecuta un nuevo contenedor.

2. **`-it`**  
   - `-i` (interactivo): Mantiene la entrada estándar abierta para que puedas interactuar con el contenedor.  
   - `-t` (pseudo-TTY): Asigna una terminal virtual para que puedas ver el shell como si fuera una terminal normal.  

3. **`--rm`**  
   - **Elimina automáticamente el contenedor cuando se detiene.**  
   - Esto significa que cualquier cambio realizado dentro del contenedor se perderá a menos que uses volúmenes o montes un directorio persistente.  

4. **`--name kali_persist`**  
   - Asigna el nombre `kali_persist` al contenedor para facilitar su identificación.  

5. **`kalilinux/kali-rolling`**  
   - Usa la imagen `kalilinux/kali-rolling`, que es la versión rolling de Kali Linux disponible en Docker Hub.  

6. **`/bin/bash`**  
   - Ejecuta el shell Bash dentro del contenedor, permitiéndote interactuar con el sistema.  

### **¿Qué hace este comando en resumen?**  
Crea y ejecuta un contenedor de Kali Linux en modo interactivo con una terminal (`bash`). Además, el contenedor se eliminará automáticamente (`--rm`) cuando lo detengas.  

### **Si deseas que el contenedor sea persistente**  
Si no quieres que se elimine al cerrarlo, elimina `--rm` y usa `-d` (modo en segundo plano):  

```bash
docker run -it -d --name kali_persist kalilinux/kali-rolling /bin/bash
```

Luego puedes conectarte con:  
```bash
docker exec -it kali_persist /bin/bash
```

### INstalar python 

Para instalar Python en Linux, sigue estos pasos según tu distribución:  

---

### **1. Verificar si Python ya está instalado**  
Antes de instalar, verifica qué versión de Python tienes con:  
```bash
python3 --version
```
Si ya está instalado, verás algo como:  
```
Python 3.x.x
```
Si no, sigue los pasos de instalación según tu sistema operativo.

---

### **2. Instalar Python según tu distribución**  

#### **En Debian, Ubuntu y derivados**  
Ejecuta:  
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv
```
Verifica la instalación con:  
```bash
python3 --version
pip3 --version
```

---

#### **En CentOS, RHEL, Fedora**  
Primero, asegúrate de tener `dnf`:  
```bash
sudo dnf install -y python3 python3-pip
```
Verifica con:  
```bash
python3 --version
pip3 --version
```

---

#### **En Arch Linux y Manjaro**  
Usa `pacman`:  
```bash
sudo pacman -Syu python python-pip
```
Verifica:  
```bash
python --version
pip --version
```

---

### **3. Configurar Python como predeterminado (Opcional)**  
Si deseas que `python3` sea el comando predeterminado en lugar de `python`:  
```bash
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
```

---

### **4. Instalar versiones específicas con `pyenv` (Opcional)**  
Si necesitas instalar y administrar múltiples versiones de Python:  
```bash
curl https://pyenv.run | bash
exec $SHELL
pyenv install 3.10.12  # Instalar una versión específica
pyenv global 3.10.12   # Establecerla como predeterminada
```

---

Con estos pasos, tendrás Python instalado y listo para usarse en Linux. 🚀

Para ver el estado de un contenedor en Docker, puedes utilizar los siguientes comandos:  

### 🔹 **1. Ver todos los contenedores en ejecución**
```sh
docker ps
```
✅ Muestra solo los contenedores que están en ejecución.  

---

### 🔹 **2. Ver el estado de todos los contenedores (incluyendo los detenidos)**
```sh
docker ps -a
```
✅ Muestra los contenedores en ejecución, detenidos y creados.  

---

### 🔹 **3. Ver el estado específico de un contenedor**
Si conoces el nombre o ID del contenedor, usa:  
```sh
docker inspect -f '{{.State.Status}}' <container_id>
```
📌 **Ejemplo:**
```sh
docker inspect -f '{{.State.Status}}' my_container
```
👀 **Posibles resultados:**  
- `"running"` → Contenedor en ejecución  
- `"exited"` → Contenedor detenido  
- `"paused"` → Contenedor en pausa  

---

### 🔹 **4. Ver más detalles del contenedor**
Para ver más información sobre su estado y configuración:  
```sh
docker inspect <container_id>
```
✅ Muestra detalles como estado, puertos, volúmenes y más.  

---

💡 **Resumen:**
- 🚀 **`docker ps`** → Ver contenedores en ejecución.  
- 🛑 **`docker ps -a`** → Ver todos los contenedores, incluso los detenidos.  
- 🔍 **`docker inspect -f '{{.State.Status}}' <container_id>`** → Ver estado exacto.  
- 📄 **`docker inspect <container_id>`** → Ver detalles completos del contenedor.  
