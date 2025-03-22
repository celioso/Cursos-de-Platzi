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