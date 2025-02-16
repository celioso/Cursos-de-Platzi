# Curso de Introducción a AWS Redes, Gobernanza y Machine Learning

Si todavía no tienes conocimientos de AWS, recuerda tomar primero los cursos de [introducción al cloud computing](https://platzi.com/cursos/aws-fundamentos "introducción al cloud computing") y [cómputo, almacenamiento y bases de datos](https://platzi.com/cursos/aws-computo/ "cómputo, almacenamiento y bases de datos").

### ¿Qué aprenderás en este curso?

En este curso aprenderemos

- Componentes principales de redes en AWS: VPC, CloudFront, Route 53
- Creación de los componentes básicos de una VPC
- Gobernanza
- Servicios de machine learning

### ¿Quién es el profesor?

Alexis Araujo, Data Architect dentro del team Platzi. Apasionado por el uso de tecnología cloud, Alexis es un Cloud Practicioner certificado por AWS.

**Lecturas recomendadas**

[Curso de Introducción a AWS: Cómputo, Almacenamiento y Bases de Datos - Platzi](https://platzi.com/cursos/aws-computo/)

[Curso de Introducción a AWS: Fundamentos de Cloud Computing - Platzi](https://platzi.com/cursos/aws-fundamentos)

## Qué son las redes

AWS ofrece varios servicios para gestionar redes en la nube. Aquí te explico los principales:

### **📌 1. Amazon Virtual Private Cloud (Amazon VPC)**  

Amazon VPC permite crear una **red privada y aislada** en AWS donde puedes desplegar recursos como instancias **EC2, bases de datos y servicios**.  

✅ **Definir rangos de IP privadas**  
✅ **Crear subredes públicas y privadas**  
✅ **Configurar reglas de tráfico con Security Groups y NACLs**  
✅ **Conectar con Internet o redes locales** mediante **VPN o Direct Connect**  

🔹 **Ejemplo de uso:**  
- Crear una VPC con subredes privadas para bases de datos y subredes públicas para servidores web.  
- Conectar tu oficina con AWS usando **VPN o Direct Connect**.

### **📌 2. AWS Transit Gateway**  

AWS Transit Gateway permite **conectar múltiples VPC y redes locales** en una arquitectura centralizada.  

✅ **Reduce la complejidad de conexiones** entre múltiples VPC y redes on-premises.  
✅ **Mejora el rendimiento y la seguridad** en comparación con conexiones punto a punto (VPC Peering).  
✅ **Soporta conectividad con VPN y Direct Connect**.  

🔹 **Ejemplo de uso:**  
- Una empresa con múltiples VPC en diferentes regiones puede conectar todas usando un **Transit Gateway** en vez de múltiples conexiones **VPC Peering**.

### **📌 3. AWS PrivateLink**  

AWS PrivateLink permite acceder a **servicios de AWS o aplicaciones de terceros de forma privada** dentro de tu VPC, sin pasar por Internet.  

✅ **Mayor seguridad** al evitar tráfico público.  
✅ **Baja latencia y menor exposición a amenazas externas**.  
✅ **Ideal para conectar VPC a servicios de AWS como S3, DynamoDB, RDS** de forma privada.  

🔹 **Ejemplo de uso:**  
- Conectar tu VPC a **Amazon S3 o DynamoDB** sin exponer el tráfico a Internet.  
- Usar un servicio de un proveedor externo a través de **PrivateLink** sin abrir puertos públicos.

### **📌 4. Amazon Route 53**  

Amazon Route 53 es el servicio de **DNS escalable y altamente disponible** de AWS.  

✅ **Gestión de nombres de dominio** y registros DNS.  
✅ **Balanceo de carga con geolocalización y failover**.  
✅ **Integración con AWS para enrutar tráfico entre servicios**.  

🔹 **Ejemplo de uso:**  
- Crear un dominio personalizado para tu aplicación en AWS.  
- Configurar **balanceo de carga global** redirigiendo tráfico según la ubicación del usuario.

### **🚀 Resumen y Comparación**  

| Servicio        | Función Principal | Casos de Uso |
|---------------|----------------|--------------|
| **Amazon VPC** | Crea una red privada en AWS | Alojar aplicaciones con control de red |
| **AWS Transit Gateway** | Conecta múltiples VPC y redes locales | Empresas con muchas VPC o conexiones híbridas |
| **AWS PrivateLink** | Acceso privado a servicios de AWS o terceros | Conectar servicios sin exponerlos a Internet |
| **Amazon Route 53** | Servicio DNS escalable | Dominios personalizados, balanceo de carga global |

📌 **Conclusión:**  
AWS ofrece herramientas poderosas para la conectividad en la nube. **VPC** es la base de la red, **Transit Gateway** facilita la conectividad entre múltiples redes, **PrivateLink** asegura accesos privados y **Route 53** gestiona dominios y balanceo de carga.

### Resumen

Las **redes** son cómo están conectadas las computadoras (y otros dispositivos tecnológicos) entre sí, y los servicios que permiten esto.

Una muy conocida es el Internet, que consiste en una red de computadoras abierta al mundo. Para que Internet funcione es necesario contar con direcciones IP, enrutadores, DNS y seguridad. AWS provee servicios que permiten la creación de redes y la entrega de contenido a los usuarios de manera rápida.

### Redes en la nube

Entre los servicios de AWS para implementar redes en la nube encontramos:

- [Amazon Virtual Private Cloud (Amazon VPC):](https://platzi.com/clases/2733-aws-redes/48886-que-es-una-vpc/ "Amazon Virtual Private Cloud (Amazon VPC): permite definir y aprovisionar una red privada para nuestros recursos de AWS
- [AWS Transit Gateway](https://aws.amazon.com/transit-gateway/?whats-new-cards.sort-by=item.additionalFields.postDateTime&whats-new-cards.sort-order=desc "AWS Transit Gateway"): Permite conectar VPC con los recursos locales (on-premises) mediante un hub central
- [AWS PrivateLink](https://aws.amazon.com/privatelink/?privatelink-blogs.sort-by=item.additionalFields.createdDate&privatelink-blogs.sort-order=desc "AWS PrivateLink"): proporciona conectividad privada entre las VPC y aplicaciones locales, sin exponer el tráfico al Internet público
- [Amazon Route 53](https://platzi.com/clases/2733-aws-redes/48888-route-53/ "Amazon Route 53"): permite alojar nuestro propio DNS administrado

### Redes a escala

Estos servicios nos permiten escalar el tráfico de red según las necesidades:

- **Elastic Load Balancing**: permite distribuir automáticamente el tráfico de red a través de un grupo de recursos, con el fin de mejorar la escalabilidad
- **AWS Global Accelerator**: redirige el tráfico a través de la red global de AWS para mejorar el rendimiento de las aplicaciones globales
- **Amazon CloudFront**: entrega de forma segura datos, videos y aplicaciones a clientes de todo el mundo con baja latencia.

**Lecturas recomendadas**

[Curso de Redes Informáticas de Internet - Platzi](https://platzi.com/cursos/redes/)

## Qué es una VPC

Una **VPC (Virtual Private Cloud)** es una red virtual en la nube de AWS que permite a los usuarios lanzar recursos de AWS en un entorno aislado lógicamente. Básicamente, es una red privada definida por software dentro de AWS, en la que se pueden configurar subredes, tablas de enrutamiento, puertas de enlace de Internet y reglas de seguridad para controlar el tráfico de red.

### **Características clave de una VPC en AWS**:
1. **Aislamiento lógico**: Permite que los recursos se ejecuten en una red virtual separada de otras VPCs y clientes de AWS.
2. **Subredes personalizadas**: Se pueden definir subredes públicas y privadas dentro de la VPC.
3. **Control del tráfico**: Uso de **Security Groups** y **Network ACLs** para administrar el tráfico entrante y saliente.
4. **Opciones de conectividad**: Se puede conectar a Internet, otras VPCs, centros de datos locales o servicios como AWS Direct Connect y VPN.
5. **Alta disponibilidad y escalabilidad**: Compatible con balanceo de carga y autoescalado.

En resumen, una **VPC** proporciona un entorno seguro y flexible para alojar servidores, bases de datos y otras aplicaciones en la nube de AWS.

### Resumen

Una VPC es una red virtual privada. Cada computadora que está conectada a otra computadora por medio de un cable, enrutador o antena de wifi, requiere de una interfaz de red para ser conectada. La interfaz de red es el puente entre nuestra computadora y la tecnología ya utilizada para conectarse a la otra computadora.

Una vez que conectamos las computadoras, debemos configurar la red, para lo cual necesitamos un [rango de direcciones IP](https://platzi.com/clases/2225-redes/35586-clases-de-redes/ "rango de direcciones IP").

### Qué es el rango de direcciones IP

El rango de direcciones IP es como una comunidad cerrada local, donde los equipos se podrán comunicar solo con otros equipos dentro de la misma red. A cada equipo se le asigna una dirección IPv4. Es decir, se le dan 4 números que varían del 0 al 255 separados por un punto. Para redes privadas ya se tienen especificados los rangos de IP:

- 10.0.0.1
- 172.16.0.1
- 192.168.0.1

### Para qué sirve Amazon VPC

Amazon VPC permite crear una red virtual para poder conectarnos a todos los servicios de AWS que existan en un rango de direcciones IP locales (por ejemplo, 10.0.0.0/24, que representa del rango de IP entre 10.0.0.0 y 10.0.0.255). Esta red virtual será como una pequeña comunidad cerrada para nuestras máquinas virtuales y todos los servicios que tengamos dentro de AWS.

### Componentes de Amazon VPC

Amazon VPC posee los siguientes componentes para controlar el tráfico interno y externo de nuestras VPC

- **Nat Gateway**: si deseamos que nuestras máquinas virtuales puedan acceder a internet, debemos utilizar este componente
- **Internet Gateway**: permite que Internet pueda acceder a nuestra instancia de EC2
- **ACL Control List**: controla el tráfico que vamos a permitir dentro y fuera de la VPC

## Escogiendo CloudFront

### **¿Cuándo elegir Amazon CloudFront?**  

Amazon **CloudFront** es un **servicio de red de entrega de contenido (CDN)** que acelera la distribución de contenido estático y dinámico a usuarios de todo el mundo. Es ideal para aplicaciones con requisitos de baja latencia y alto rendimiento.  

### **Casos en los que se recomienda usar CloudFront**  

✅ **Distribución de contenido estático** (imágenes, videos, archivos CSS/JS)  
✅ **Streaming de video** (HLS, DASH, Smooth Streaming)  
✅ **Aceleración de sitios web y APIs**  
✅ **Protección contra ataques DDoS** (integración con AWS Shield)  
✅ **Optimización de costos** (reduce tráfico a servidores backend)  

Si necesitas **entregar contenido rápido, seguro y eficiente a nivel global**, **Amazon CloudFront** es la opción ideal.

### Resumen

Antes de hablar de CloudFront, recordemos cómo funciona [AWS ElastiCache](https://platzi.com/clases/2732-aws-computo/47026-evaluando-elasticache/ "AWS ElastiCache"). **ElastiCache es un servicio que almacena en memoria caché las solicitudes a la base de datos, para evitar el consultar la base de datos cada vez que se necesite acceder a información**. Este servicio se ubica entre el sitio web y la base de datos

**CloudFront funciona de manera similar, solo que este es un servicio intermedio entre el navegador (o el cliente) y el sitio web**. El propósito de CloudFront es entregar datos, aplicaciones y sitios web en todos el mundo con baja latencia. Para esto, AWS cuenta con **edge locations** (o ubicaciones de borde), es decir, múltiples ubicaciones en el mundo desde las cuales CloudFront puede servir contenido.

**Casos de uso de CloudFront**

Supongamos que un cliente accede a nuestro sitio web. En realidad, el cliente primero accede a CloudFront. Entonces CloudFront redirige automáticamente la solicitud de archivo desde el edge location más cercano. Los archivos se almacenan en la caché de la ubicación de borde primero, durante un periodo de tiempo limitado que nosotros necesitemos.

Si un cliente solicita el contenido que está almacenado en caché por más tiempo que el vencimiento especificado, CloudFront verifica en el servidor de origen para ver si hay una nueva versión del archivo disponible. Si el archivo ha sido modificado, se retorna la nueva versión del archivo. En caso contrario, se entrega la versión que estaba en caché.

Cualquier cambio que realicemos en los archivos se replicará en las ubicaciones de borde a medida que sus visitantes están entrando y solicitando el contenido. Esto es lo que mantiene a los sitios web rápidos sin importar la ubicación del usuario.

### Características de CloudFront

- CloudFront es seguro: ofrece protección contra ataques DDOS, ya que los primeros servidores en recibir estos ataques serán los de CloudFront y no los tuyos. Además, CloudFront está protegido ante picos de tráfico.
- CloudFront también permite ejecutar funciones de [AWS Lambda](https://platzi.com/clases/2732-aws-computo/47016-aprendiendo-sobre-lambda/ "AWS Lambda") en las ubicaciones de borde.
- CloudFront ofrece múltiples métricas en tiempo real, y es rentable.

## Qué es Route 53

**Amazon Route 53** es un **servicio de DNS (Sistema de Nombres de Dominio) escalable y altamente disponible** que permite administrar nombres de dominio y enrutar el tráfico de Internet de manera eficiente.

### **¿Para qué sirve Route 53?**

✔ **Registrar dominios**: Puedes comprar y administrar nombres de dominio directamente en AWS.  
✔ **Resolver nombres de dominio**: Convierte nombres de dominio (como `miweb.com`) en direcciones IP (`192.168.1.1`).  
✔ **Rutas inteligentes de tráfico**: Distribuye tráfico basado en latencia, geolocalización o salud de servidores.  
✔ **Alta disponibilidad**: Integra con AWS servicios como **CloudFront, S3 y EC2** para mejorar rendimiento.  
✔ **Monitoreo y failover**: Detecta fallas y redirige el tráfico automáticamente a una instancia saludable.

### **¿Cuándo usar Route 53?**

✅ **Si necesitas un DNS rápido, seguro y confiable**.  
✅ **Para administrar tráfico global y mejorar disponibilidad**.  
✅ **Si quieres combinarlo con otros servicios de AWS**.  

🔹 **Conclusión**: Amazon Route 53 es ideal para gestionar dominios y enrutar tráfico con escalabilidad y alta disponibilidad.

### Resumen

[DNS](https://platzi.com/clases/2053-introweb/32966-dns/ "DNS") es un sistema que asigna direcciones IP a nombres de dominio. **Route 53 es un servicio de alojamiento de DNS, que cuesta tan solo $0.5 por nombre de dominio por mes. Route 53 cuenta con distintas opciones de política de enrutamiento**.

### Políticas de enrutamiento

Las políticas de enrutamiento nos permiten determinar a dónde se dirigirá un usuario cuando acceda a nuestro dominio. Estas políticas son:

### Ruteo simple

El ruteo simple utiliza el servicio de DNS estándar. Es decir, el tráfico en un dominio se enruta hacia un recurso muy específico.

### Política ponderada

La **política ponderada** (o *weighted routing*) te permite asociar múltiples recursos con un solo nombre de dominio, y ver qué tanto tráfico es dirigido a cada recurso. Esto se determina con un número del 0 al 255, donde el cero representa que el recurso no recibe ningún tráfico, y el 255 indica que el recurso recibe todo el tráfico.

Mediante la política ponderada podemos probar distintas versiones de nuestro sitio web con un número reducido de usuarios. Luego podemos realizar una transición lenta de nuestros usuarios hacia la nueva versión del sitio.

### Política de geolocalización

Usando la **política de geolocalización** podemos escoger qué recursos servir en función de la ubicación geográfica de nuestros usuarios. Esto permite servir contenido específico según la región, así como restringir la distribución del mismo solo a las regiones permitidas.

### Política de latencia

La política de latencia se trata de entregar los recursos desde la región de AWS que esté más cercana a la ubicación del usuario, a fin de reducir el tiempo de respuesta.

### Política de conmutación por error

La política de conmutación por error redirige el tráfico a un recurso cuando este está en buen estado, o a uno diferente cuando el primer recurso no está en buen estado.

### Política de respuesta de múltiples valores

La **respuesta de múltiples valores** permite devolver varios valores, como direcciones IP a los servidores web, en respuesta a las consultas de DNS. Se pueden especificar varios valores para casi cualquier registro, pero este direccionamiento también permite verificar el estado de cada recurso, por lo que Route 53 devuelve los valores únicamente para los recursos en buen estado.

**Esta política no es sustituto de un balanceador de carga**, pero la capacidad de devolver varias direcciones IP (cuyo estado sea comprobable) permite usar el DNS para mejorar la disponibilidad y el equilibrio de la carga.

**Conclusión**

**Route 53 es un servicio complejo, pero útil para mantener nuestros sitios web rápidos y altamente disponibles. Es rentable, seguro, escalable, y posee distintas opciones de enrutamiento para distintos casos**.

## Cómo crear el diagrama de una VPC

Para crear un diagrama de una **VPC en AWS**, sigue estos pasos:

### **1️⃣ Definir la arquitectura de la VPC**  
Tu VPC debe incluir elementos clave como:  
✅ **CIDR Block**: El rango de direcciones IP privadas (Ej: `10.0.0.0/16`).  
✅ **Subredes**: Al menos una pública y una privada.  
✅ **Internet Gateway (IGW)**: Permite el acceso a Internet.  
✅ **Route Tables**: Define cómo se enruta el tráfico entre subredes.  
✅ **NAT Gateway**: Permite que las subredes privadas accedan a Internet sin ser accesibles desde el exterior.  
✅ **Security Groups & Network ACLs**: Controlan el tráfico entrante y saliente.

### **2️⃣ Elegir una herramienta para diagramar**  
Puedes usar:  
🔹 **AWS Diagramming Tool (en AWS Architecture Center)**  
🔹 **Lucidchart**  
🔹 **Draw.io** (gratuito y fácil de usar)  
🔹 **Visio**  
🔹 **Excalidraw**

### **3️⃣ Construcción del diagrama**  
Aquí te dejo una estructura típica:  

📌 **Ejemplo de diagrama de una VPC con 2 subredes**  
```
                 ┌──────────────────────────────────────┐
                 │              AWS VPC                │  (10.0.0.0/16)
                 ├──────────────────────────────────────┤
                 │                                      │
    ┌───────────┴───────────┐      ┌───────────┴───────────┐
    │   Subred Pública      │      │   Subred Privada      │
    │  (10.0.1.0/24)        │      │  (10.0.2.0/24)        │
    │   Internet Gateway    │      │  NAT Gateway (Opcional) │
    │  EC2 (Servidor Web)   │      │  EC2 (Base de Datos)  │
    └───────────┬───────────┘      └───────────┬───────────┘
                 │                              │
                 └─────── Route Table ─────────┘
```
---
### **4️⃣ Validación y Mejoras**  
🔹 Usa **AWS Well-Architected Tool** para validar la arquitectura.  
🔹 Considera **VPC Peering** si necesitas comunicarte con otras VPCs.  
🔹 Añade **VPN o AWS Direct Connect** si integras con tu red local.

### Resumen

Aprendámos a crear los componentes básicos de una VPC desde cero. **Primero necesitamos hacer un diagrama para entender cómo están divididos estos componentes básicos**.

Para originar el diagrama nos dirigimos [a esta herramienta de diagramas de flujo](https://app.diagrams.net/ "a esta herramienta de diagramas de flujo") y escogemos dónde guardaremos el diagrama (en esta clase se escoge Google Drive, pero puedes guardarlo donde prefieras). Entonces le damos a **“Create New Diagram”** -> **“Blank Diagram”**.

### Creando el diagrama de la VPC

En el recuadro de búsqueda podemos poner “AWS VPC”. Escogemos la siguiente figura.

![VPC shape](images/VPC_shape.png)

Luego, buscamos las siguientes figuras: “AWS Internet Gateway”, “User”, “network access control”, “router” y “subnet”. Entonces las ordenamos de la siguiente manera

![Diagrama de VPC](images/Diagrama_de_VPC.png)

Este es el diagrama final. Muestra que cuando un usuario intenta acceder al VPC se encontrará con el **[Internet Gateway](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_Internet_Gateway.html "Internet Gateway")**. Luego, el tráfico será dirigido al **router**, que se encargará de redirigirlo a una de las dos subnets las cuales contienen un **Network Access Control List**. Este se encargará de validar que el usuario pueda acceder al contenido.

**Lecturas recomendadas**

[Cloud Computing Services - Amazon Web Services (AWS)](https://aws.amazon.com/)

[Flowchart Maker & Online Diagram Software](https://app.diagrams.net/)

## Cómo crear la VPC y el internet gateway

Puedes crear una VPC y un Internet Gateway desde la **Consola de AWS** o usando la **AWS CLI**. Aquí te explico ambos métodos.

### **📌 Opción 1: Crear VPC desde la Consola de AWS**  

1️⃣ **Ir a la Consola de AWS** → **VPC** → **Crear VPC**  
2️⃣ **Configurar la VPC**:  
   - **Nombre**: `MiVPC`  
   - **Rango de IP (CIDR)**: `10.0.0.0/16`  
   - **Tenancy**: `Predeterminado`  
3️⃣ **Crear y guardar**  

4️⃣ **Crear un Internet Gateway (IGW)**:  
   - Ir a **Internet Gateways** → **Crear Internet Gateway**  
   - **Nombre**: `MiInternetGateway`  
   - Hacer clic en **Crear**  

5️⃣ **Adjuntar el IGW a la VPC**:  
   - Seleccionar `MiInternetGateway`  
   - Hacer clic en **Acciones → Adjuntar a VPC**  
   - Seleccionar `MiVPC` y **Confirmar**

## **📌 Opción 2: Crear VPC e IGW con AWS CLI**  

📌 **Crear la VPC:**  
```sh
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=MiVPC}]'
```

📌 **Crear el Internet Gateway:**  
```sh
aws ec2 create-internet-gateway --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=MiInternetGateway}]'
```

📌 **Adjuntar el IGW a la VPC:**  
```sh
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=cidr-block,Values=10.0.0.0/16" --query "Vpcs[0].VpcId" --output text)
IGW_ID=$(aws ec2 describe-internet-gateways --filters "Name=tag:Name,Values=MiInternetGateway" --query "InternetGateways[0].InternetGatewayId" --output text)

aws ec2 attach-internet-gateway --internet-gateway-id $IGW_ID --vpc-id $VPC_ID
```

### **📌 Pasos Siguientes**  
✅ **Crear subredes** (pública y privada)  
✅ **Configurar una tabla de enrutamiento** para permitir tráfico a Internet  
✅ **Configurar reglas de seguridad** en los Security Groups

### Resumen

Una vez creado nuestro [diagrama de vpc](https://platzi.com/clases/2733-aws-redes/48889-crea-el-diagrama/ "diagrama de vpc"), [iniciamos sesión en AWS](https://console.aws.amazon.com/console/home?nc2=h_ct&src=header-signin "iniciamos sesión en AWS") para crear los primeros componentes de nuestra VPC.

### Pasos para crear la VPC

1. En la caja de búsqueda de AWS buscamos VPC y seleccionamos el primer resultado.
2. Nos dirigimos a “**Sus VPC**” y le damos a “**Crear VPC**”.
3. Colocamos las siguientes opciones, y dejamos el resto de valores por defecto:

- **Etiqueta de nombre - opcional**: DemoVPCLaboratorio.
- **Bloque de CIDR IPv4**: Entrada manual de CIDR IPv4.
- **CIDR IPv4**: 10.0.0.0/24.

![Configuración de la VPC](images/Configuracion_de_la_VPC.png)

Entonces le damos a **Crear VPC**.

### Pasos para crear el Internet Gateway

1. Nos dirigimos a **“Gateways de Internet” -> “Crear gateway de Internet”**.
2. En “**Etiqueta de nombre**”, colocamos “**DemoIGWLaboratorio**”, y le damos a “Crear gateway de Internet”.
3. Nos aparecerá nuestro nuevo Internet Gateway con un estado “Detached”, ya que no está ligado a ninguna VPC.
4. Para conectar el Intenet Gateway a nuestra VPC, simplemente le damos clic en “**Acciones**” -> “**Conectar a la VPC**”.
5. Aquí seleccionamos nuestra VPC, y le damos clic a “**Concetar gateway de Internet**”. Ojo, **el Internet Gatway y la VPC deben estar en la misma región.**

![diagrama de la VPC](images/diagramaVPC.png)

Ya con esto creamos dos de los componentes de nuestra VPC.

## Cómo crear la tabla de enrutamiento y otros componentes 

Después de crear la **VPC y el Internet Gateway (IGW)**, necesitas:  
✔ **Tabla de enrutamiento** para dirigir el tráfico.  
✔ **Subredes** (pública y privada).  
✔ **Asociar la tabla de enrutamiento** con las subredes.  
✔ **Configurar un grupo de seguridad** para controlar el tráfico.

### **📌 Paso 1: Crear Subredes**

Debes crear al menos **una subred pública** y **una privada** dentro de la VPC.

### **Desde la Consola AWS**

1️⃣ Ir a **VPC** → **Subredes** → **Crear Subred**  
2️⃣ **Seleccionar la VPC creada (MiVPC)**  
3️⃣ **Crear subred pública**  
   - Nombre: `SubredPublica`  
   - CIDR: `10.0.1.0/24`  
   - Zona de disponibilidad: (Ej: `us-east-1a`)  
   - **Habilitar la asignación automática de IPs públicas**  
4️⃣ **Crear subred privada**  
   - Nombre: `SubredPrivada`  
   - CIDR: `10.0.2.0/24`  
   - Zona de disponibilidad: (Ej: `us-east-1b`)  

### **Desde AWS CLI**

```sh
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=tag:Name,Values=MiVPC" --query "Vpcs[0].VpcId" --output text)

aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.0.1.0/24 --availability-zone us-east-1a --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=SubredPublica}]'

aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.0.2.0/24 --availability-zone us-east-1b --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=SubredPrivada}]'
```

## **📌 Paso 2: Crear la Tabla de Enrutamiento y Asociarla**

La tabla de enrutamiento define cómo se dirige el tráfico dentro de la VPC.

### **Desde la Consola AWS**

1️⃣ Ir a **VPC** → **Tablas de Enrutamiento** → **Crear Tabla de Enrutamiento**  
2️⃣ **Nombre**: `TablaPublica`  
3️⃣ **Seleccionar la VPC (MiVPC)**  
4️⃣ **Agregar Ruta**:  
   - Destino: `0.0.0.0/0` (todo el tráfico)  
   - Target: `Internet Gateway (MiInternetGateway)`  
5️⃣ **Asociar con la Subred Pública**  
   - Ir a **Asociaciones de Subredes** → Seleccionar `SubredPublica`  

### **Desde AWS CLI**

```sh
# Crear tabla de enrutamiento
RT_ID=$(aws ec2 create-route-table --vpc-id $VPC_ID --query "RouteTable.RouteTableId" --output text)
aws ec2 create-tags --resources $RT_ID --tags Key=Name,Value=TablaPublica

# Agregar ruta a Internet Gateway
IGW_ID=$(aws ec2 describe-internet-gateways --filters "Name=tag:Name,Values=MiInternetGateway" --query "InternetGateways[0].InternetGatewayId" --output text)
aws ec2 create-route --route-table-id $RT_ID --destination-cidr-block 0.0.0.0/0 --gateway-id $IGW_ID

# Asociar con la Subred Pública
SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=tag:Name,Values=SubredPublica" --query "Subnets[0].SubnetId" --output text)
aws ec2 associate-route-table --route-table-id $RT_ID --subnet-id $SUBNET_ID
```

## **📌 Paso 3: Configurar un Grupo de Seguridad (Firewall de AWS)**  
Los grupos de seguridad controlan el tráfico entrante y saliente de instancias EC2.

### **Desde la Consola AWS**  
1️⃣ Ir a **VPC** → **Grupos de Seguridad** → **Crear Grupo de Seguridad**  
2️⃣ **Nombre**: `SG-WebServer`  
3️⃣ **Seleccionar VPC (MiVPC)**  
4️⃣ **Reglas Entrantes**:  
   - **Permitir tráfico HTTP (80)**:  
     - Tipo: HTTP  
     - Protocolo: TCP  
     - Puerto: 80  
     - Origen: `0.0.0.0/0`  
   - **Permitir tráfico SSH (22)** _(solo si necesitas administrar el servidor)_  
     - Tipo: SSH  
     - Protocolo: TCP  
     - Puerto: 22  
     - Origen: `Tu IP` (`X.X.X.X/32`)  

### **Desde AWS CLI**  
```sh
SG_ID=$(aws ec2 create-security-group --group-name SG-WebServer --description "Grupo de Seguridad para Web" --vpc-id $VPC_ID --query "GroupId" --output text)

# Permitir HTTP (80)
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 80 --cidr 0.0.0.0/0

# Permitir SSH (22) - Cambia "X.X.X.X/32" por tu IP
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 22 --cidr X.X.X.X/32
```

### **📌 Paso 4: Crear una Instancia EC2 (Opcional)**
Si quieres probar la conectividad, puedes lanzar una instancia EC2 en la **Subred Pública** con el **Grupo de Seguridad** creado.

```sh
aws ec2 run-instances --image-id ami-0abcdef1234567890 --count 1 --instance-type t2.micro --subnet-id $SUBNET_ID --security-group-ids $SG_ID --key-name MiClave --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ServidorWeb}]'
```

### **🎯 Conclusión**
✅ Ya tienes tu VPC configurada con una subred pública conectada a Internet.  
✅ Puedes lanzar servidores en la **Subred Pública** con acceso a Internet.  
✅ Si necesitas una **Subred Privada**, puedes crear una con un **NAT Gateway**.

### Resumen

Una vez que [creamos nuestra VPC y el Internet Gateway y los conectamos](https://platzi.com/clases/2733-aws-redes/49130-crear-la-vpc-y-crear-el-internet-gateway/ "creamos nuestra VPC y el Internet Gateway y los conectamos"), procedemos a crear la **tabla de enrutamiento**, las **listas de acceso de control** y **las subredes**.

### Pasos para crear la tabla de enrutamiento

1. Desde la [página del servicio de VPC](https://console.aws.amazon.com/vpc/home "página del servicio de VPC"), nos dirigimos a “**Tablas de ruteo**”.
2. Notamos que ya existe una tabla de ruteo asociada a nuestra VPC, que se creó automáticamente junto con la VPC.
3. La seleccionamos, nos dirigimos a la sección de rutas, y hacemos clic en “**Editar rutas**”.

![Editar rutas](images/Editar_rutas.png)

4. Hacemos clic en “**Agregar ruta**”, colocamos **0.0.0.0/0** y “**Puerta de enlace de internet**”, y seleccionamos el [Internet Gateway](https://platzi.com/clases/2733-aws-redes/49130-crear-la-vpc-y-crear-el-internet-gateway/ "Internet Gateway") que creamos en la clase pasada.
5. Le damos en “**Guardar cambios**”. De esta manera, todo el mundo podrá acceder a nuestra VPC mediante el Internet Gateway.

![Agregar ruta](images/Agregar_ruta.png)

### Pasos para crear Access Control List

1. En el apartado de “**Seguridad**” del servicio de VPC, nos dirigimos a “**ACL de red**”.
2. Le damos clic a “**Crear ACL de red**”. Crearemos dos ACL de red, uno para cada subred. Le damos los nombres **NACLA** y **NACLB**, y en VPC escogemos nuestra VPC.
3. Le damos clic en “**Crear ACL de red**”.

![Crear ACL de red](images/Crear_ACL_de_red.png)

### Pasos para añadir una regla de entrada y una de salida

Ahora, para cada ACL de red creado debemos añadir una regla de entrada y una de salida, con el fin de permitir el tráfico HTTP en el puerto 80. Para esto:

1. Seleccionamos una ACL de red
2. Nos vamos a “**Reglas de entrada**” -> “**Editar reglas de entrada**”.

![Editar reglas de entrada](Editar_reglas_de_entrada.png)

3. Le damos clic en “**Añadir una nueva regla**”. Y colocamos los siguientes parámetros

- **Número de regla**: 100 (las reglas se evalúan comenzando por la regla de número más bajo).
- **Tipo**: HTTP (80).
- **Origen**: 0.0.0.0/0.
- **Permitir/denegar**: Permitir.

4. Le damos a “**Guardar cambios**”.
5. Repetimos el proceso con la regla de salida y con el otro ACL (NACLB), colocando los mismos parámetros anteriores. Ahora solo falta añadir estos ACL a nuestras subredes, las cuales crearemos a continuación.

![Añadir una nueva regla de entrada](Añadir_una_nueva_regla_de_entrada.png)
Añadir una nueva regla de entrada

### Pasos para crear subredes

1. En la sección de “**Subredes**” vamos al botón “**Crear subred**”.
2. Escogemos nuestra VPC, y colocamos los siguientes parámetros:
- **Nombre de la subred**: DemoSubredA.
- **Zona de dispinibilidad**: la primera que te aparezca en el menú de selección, que termine en “a”.
- **Bloque de CIDR IPv4**: 10.0.0.0/25 (asumiendo que tu VPC tiene el bloque de CIDR 10.0.0.0/24)

3. Le damos clic en “**Crear subred**”
4. Repetimos el procedimiento para la otra subred con los siguientes parámetros:
- **Nombre de la subred**: DemoSubredB.
- **Zona de dispinibilidad**: la segunda que te aparezca en el menú de selección, que termine en “b”.
- **Bloque de CIDR IPv4**: 10.0.0.128/25.

![Crear subred](Crear_subred.png)

Ahora solo falta **asociar los ACL que creamos con las subredes**. Para esto simplemente hacemos clic derecho en DemoSubredA y clic en “**Editar la asociación de ACL de red**”, y seleccionamos la ACL correspondiente (NACLA). Entonces le damos en Guardar, y repetimos el procedimiento con *DemoSubredB*.

**Recapitulación**

Ya creamos todos los componentes de nuestra VPC: el Internet Gateway, la tabla de enrutamiento, las Access Control List y las subredes. Además, dimos acceso público a dichas subredes mediante HTTP en el puerto 80.