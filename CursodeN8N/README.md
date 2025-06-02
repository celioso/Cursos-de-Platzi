# Curso de N8N

## Automatización inteligente con N8N

## ¿Qué es n8n?

* **n8n** es una herramienta de automatización de flujos de trabajo (workflow automation) de código abierto.
* Permite conectar diferentes aplicaciones y servicios para automatizar tareas sin necesidad de programar mucho.
* Tiene una interfaz visual para diseñar flujos con nodos que representan acciones, triggers, y condiciones.

### ¿Qué es la automatización inteligente con n8n?

* Se refiere a la capacidad de crear flujos automatizados que no solo ejecutan tareas simples, sino que también pueden **tomar decisiones** basadas en datos.
* Por ejemplo, un flujo que:

  * Recibe un formulario web,
  * Valida la información,
  * Decide enviar un email o notificación solo si cierta condición se cumple,
  * Y guarda datos en una base de datos o sistema CRM automáticamente.

### Beneficios de usar n8n para automatización inteligente

* **Flexibilidad**: Puedes conectar más de 200 aplicaciones (Slack, Gmail, Trello, GitHub, bases de datos, etc.).
* **Control total**: Al ser open source, puedes desplegarlo en tu propio servidor o usar la nube.
* **Flujos condicionales**: Crear decisiones dentro del flujo con nodos condicionales.
* **Extensible**: Puedes escribir funciones personalizadas en JavaScript para lógica compleja.
* **Automatización sin código**: Ideal para usuarios no desarrolladores que quieren automatizar tareas.

### Ejemplo sencillo de flujo en n8n

1. **Trigger:** Cada vez que se recibe un correo con asunto "Soporte".
2. **Condición:** Si el correo contiene la palabra "urgente".
3. **Acción:** Crear un ticket en el sistema de soporte y enviar una notificación a Slack.
4. **Si no es urgente:** Solo crear el ticket sin notificación.

### Cómo empezar con n8n

* Puedes probarlo en [https://n8n.io](https://n8n.io) con la versión cloud.
* También puedes instalarlo localmente o en un servidor con Docker.
* La documentación oficial es muy completa para guiarte en la creación de flujos personalizados.

### Resumen

La automatización de procesos alcanza un nuevo nivel con N8N, una plataforma low-code que centraliza y simplifica la integración de Inteligencia Artificial. En sólo segundos, puedes validar información de formularios, recibir alertas instantáneas a través de Telegram, WhatsApp o Gmail y actualizar automáticamente los dashboards de rendimiento a la vista del equipo directivo.

#### ¿Qué tareas puedes automatizar con agentes IA en N8N?

Con esta plataforma puedes integrar fácilmente tus herramientas cotidianas como Google Sheets, Gmail y Telegram con aplicaciones especializadas como OpenAI y Pinecone. Algunos ejemplos prácticos incluyen:

- Bots que extraen texto de imágenes y lo registran en Google Sheets.
- Agentes RAG generando respuestas precisas desde bases de datos vectoriales.
- Agentes MCP procesando y organizando información desde múltiples canales.

#### ¿Cómo se ejecutan estos procesos sin escribir código?

La creación de flujos inteligentes en N8N se realiza mediante la conexión visual de nodos. De esta manera, facilita que usuarios con diferentes perfiles, desde makers, marketers, desarrolladores y emprendedores, puedan adoptar fácilmente la automatización sin necesidad de escribir código.

#### ¿Quién puede beneficiarse de automatizar procesos con N8N?

La versatilidad del sistema permite que distintos profesionales logren más con menos esfuerzo, optimizando diariamente la productividad en actividades cotidianas y especializadas. Es especialmente útil para:

- Profesionales de marketing buscando una mejor gestión y respuestas rápidas.
- Desarrolladores que quieren reducir tareas repetitivas.
- Emprendedores optimizando presupuestos y tiempos de ejecución.

¿Estás listo para comenzar tu primer flujo y explorar cómo la IA puede optimizar tu trabajo diario?

**Archivos de la clase**

[mi-primer-flujo-n8n-1.xlsx](https://static.platzi.com/media/public/uploads/mi-primer-flujo-n8n-1_e5c15db2-6bdc-4fc9-858c-5a5ecf26786c.xlsx)

**Lecturas recomendadas**

[Community edition features | n8n Docs](https://docs.n8n.io/hosting/community-edition-features/)

## Diferencias entre N8N Cloud y N8N Community

### 🟦 n8n Cloud

**Versión comercial alojada por el equipo de n8n.**

### ✅ Características:

* **Alojamiento gestionado** por n8n (no te preocupas por servidores, backups ni actualizaciones).
* **Alta disponibilidad** y escalabilidad automática.
* **Seguridad** empresarial (cifrado, acceso mediante SSO, backups automáticos).
* **Acceso prioritario** a nuevas funciones y soporte técnico.
* **Autenticación y control de acceso** (RBAC, usuarios y roles).
* **Planes de pago** (mensuales/anuales) según volumen de ejecución y usuarios.

### 🧠 Ideal para:

* Empresas y equipos que quieren evitar la gestión de infraestructura.
* Casos donde se requiere confiabilidad y soporte profesional.

### 🟩 n8n Community Edition (CE)

**Versión gratuita y de código abierto, autoalojada.**

### ✅ Características:

* **100% gratis y open source** (bajo licencia [Sustainable Use License](https://github.com/n8n-io/n8n/blob/master/LICENSE.md)).
* **Control total**: Tú lo instalas, lo ejecutas y lo mantienes.
* **Altamente personalizable**: Puedes modificar el código fuente.
* **Acceso completo a funciones principales** (triggers, nodos, lógica, etc.).

### 🛠️ Requiere:

* Gestión del servidor (Docker, VPS, NGINX, certificados, backups, etc.).
* Actualizaciones manuales y configuración propia de seguridad.

### 🧠 Ideal para:

* Desarrolladores, hackers y equipos técnicos que desean control total.
* Proyectos personales o startups con presupuesto limitado.

### 📊 Comparación rápida

| Característica                  | n8n Cloud             | n8n Community Edition (CE) |
| ------------------------------- | --------------------- | -------------------------- |
| Hosting                         | Gestionado por n8n    | Tú mismo (autoalojado)     |
| Precio                          | Pago mensual/anual    | Gratis                     |
| Escalabilidad                   | Automática            | Manual                     |
| Seguridad empresarial           | Sí                    | Configurable               |
| Personalización profunda        | Limitada              | Total (es open source)     |
| Requiere conocimientos técnicos | No                    | Sí                         |
| Soporte oficial                 | Incluido (según plan) | Comunidad (foros, GitHub)  |

### Resumen

N8n es una plataforma open source que facilita la automatización mediante dos opciones de despliegue: n8n Cloud y n8n Community. Cada modalidad presenta características específicas y factores relevantes al momento de seleccionar cuál conviene según tus necesidades y presupuesto.

#### ¿Qué ventajas ofrece n8n Cloud?

La licencia Enterprise n8n Cloud permite gestionar flujos de trabajo directamente desde la nube, simplificando considerablemente el proceso de configuración y mantenimiento. Sus principales ventajas son:

- Facturación sencilla por ejecución, independiente del número de nodos o datos procesados.
- Mayor control presupuestal, al facilitar la optimización de tareas dentro de cada ejecución.
- Experiencia más directa y simple, sin necesidad de infraestructura propia.

Además, n8n Cloud incluye una prueba gratuita de 14 días para conocer y evaluar la herramienta sin invertir en el inicio.

#### ¿En qué consiste n8n Community?

La licencia Community consiste en alojar personalmente la plataforma n8n, aprovechando servidores ya contratados o incluso implementándolo en el propio computador. Esta opción:

- No implica gastos adicionales en licencias, siendo completamente gratuita.
- Requiere más conocimientos técnicos para configuración y mantenimiento.
- Ofrece menos soporte en comparación con la versión Cloud.

Para disminuir la curva de aprendizaje con esta modalidad, se recomienda consultar detalladamente la documentación oficial disponible.

#### ¿Qué diferencias clave existen entre ambas versiones?

Al comparar n8n Cloud y n8n Community, las diferencias significativas giran en torno a:

- Costos: Cargo por ejecución en Cloud frente a gratuidad en Community.
- Complejidad técnica: Mayor simplicidad con Cloud, pero mayor flexibilidad y dificultad técnica con Community.
- Soporte técnico: Superior en modalidad Cloud respecto a la opción Community, que ofrece principalmente soporte comunitario e informativo.

Antes de elegir, considera cuidadosamente tu nivel técnico, infraestructura actual y necesidades de soporte para seleccionar la opción más efectiva para tus flujos de trabajo.

Si deseas ver funcionalidades específicas no disponibles en la licencia Community, consulta la sección de recursos para obtener un listado actualizado y completo.

**Lecturas recomendadas**

[Community edition features | n8n Docs](https://docs.n8n.io/hosting/community-edition-features/)

## N8N Selfhosted

**n8n Self-hosted** se refiere a la instalación de **n8n Community Edition (CE)** en tu propio servidor o infraestructura. Es ideal si buscas **control total**, **ahorro de costos** y **flexibilidad** para personalizar la herramienta según tus necesidades.

### 🚀 ¿Qué es n8n Self-hosted?

Es la versión gratuita y de código abierto de n8n que puedes instalar en:

* Tu computadora local (para pruebas)
* Un servidor privado (como VPS, EC2, DigitalOcean, etc.)
* Docker o Docker Compose
* Kubernetes (para producción a gran escala)

### ✅ Ventajas de n8n Self-hosted

* **Cero costos de licencias** (gratis).
* **Control total** sobre flujos, datos, privacidad y seguridad.
* **Personalización** avanzada (puedes editar código o crear nodos personalizados).
* **Integración local** con sistemas internos (ERP, bases de datos privadas, etc.).

### 🛠️ Requisitos previos

* Node.js ≥ 18 (si no usas Docker)
* Docker (recomendado)
* Base de datos opcional: SQLite (por defecto), Postgres, MySQL
* Conocimientos básicos de línea de comandos, red y configuración de servidores

### 🐳 Instalación rápida con Docker

La forma más simple y robusta es con Docker Compose:

### 1. Crea un archivo `docker-compose.yml`:

```yaml
version: '3.7'

services:
  n8n:
    image: n8nio/n8n
    restart: always
    ports:
      - 5678:5678
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=tu_contraseña_segura
      - N8N_HOST=tu_dominio_o_ip
      - N8N_PORT=5678
      - WEBHOOK_URL=https://tu_dominio_o_ip/
    volumes:
      - ./n8n_data:/home/node/.n8n
```

### 2. Ejecuta:

```bash
docker-compose up -d
```

### 🌐 Accede a la interfaz

Abre tu navegador y ve a:

```
http://localhost:5678
```

O bien, el dominio que configuraste con HTTPS (si estás en producción).

### 🔐 Seguridad recomendada

* Usa HTTPS con Let's Encrypt o Cloudflare.
* Activa autenticación básica (`N8N_BASIC_AUTH_*`).
* Configura backups frecuentes del volumen de datos (`./n8n_data`).
* Usa variables de entorno para configurar límites y seguridad.

### 📦 Actualización de n8n

Simplemente:

```bash
docker-compose pull
docker-compose down
docker-compose up -d
```

### 🎯 ¿Quieres ayuda con un caso específico?

Puedo guiarte paso a paso para:

* Instalar n8n en tu servidor (con o sin Docker)
* Configurar un proxy con NGINX o Traefik
* Montar flujos para automatizar tareas comunes (emails, bases de datos, APIs, etc.)

### Resumen

¿Quieres utilizar n8n en su versión gratuita con una licencia Community? **Aquí aprenderás cómo instalar y configurar n8n en la plataforma Render.com utilizando Docker y Supabase**. Este método práctico te permitirá manejar n8n de forma gratuita en un entorno completamente configurado según tus necesidades.

#### ¿Qué pasos seguir para iniciar la instalación en Render?

Para comenzar la instalación:

- Ingresa a Render.com y selecciona la opción Get Started.
- Regístrate o accede a tu cuenta, preferiblemente usando Google para facilitar el proceso.
- Una vez en el panel principal, haz clic en Add New Web Service y escoge la opción Existing Image.
- Usa la imagen Docker específica para N8n solicitando la ruta docker.n8n.io.
- Define un nombre para el proyecto (por ejemplo, NHNA self-hosteado), y continúa configurando según las opciones predeterminadas en la ubicación (Oregon, EE.UU.) y elige el plan free con 512 de RAM y 1 CPU.

#### ¿Cómo configurar las variables de entorno para n8n en Render?

La configuración de variables es crucial para asegurar la comunicación entre la plataforma y servicios externos como bases de datos:

- Usa un archivo plantilla para cargar rápidamente las variables de entorno. Este archivo estará disponible en la sección Recursos de la clase.
- Modifica las variables según tu entorno específico, usando Supabase para manejar la base de datos Postgres.
- Presta especial atención a los valores que debes extraer desde Supabase:
- Host
- Puerto
- Password (define uno seguro)
- Esquema (usa “public”)
- Usuario (generalmente Postgres)
- Timezone (por ejemplo, Lima)
- Genera una clave de encriptación seleccionando “256 bits” mediante Random Caching para seguridad adicional.

Una vez configuradas todas estas variables, ejecuta el despliegue, que podría tardar entre 2 a 5 minutos aproximadamente.

#### ¿Qué hacer luego del despliegue de n8n en Render?

Después de completarse el despliegue, verifica que esté correctamente habilitada la variable `N8N Runners Enabled` para trabajos `Pipeline`:

- Dirígete al menú Environment de Render.
- Añade la clave mencionada y establece su valor en True.
- Guarda los cambios e inicia nuevamente el despliegue.

Finalmente, usa el enlace proporcionado por Render para acceder a tu instancia personalizada de n8n. Este enlace será el medio principal para gestionar tu herramienta y flujos de trabajo. Tendrás que completar algunos datos básicos iniciales que solicitará n8n, aunque estos no afectarán tu uso directamente.

Recuerda que existen ligeras diferencias al emplear la versión autogestionada (*self-hosted*) frente a la versión en la nube. Para aclarar cualquier duda siempre podrás revisar la documentación oficial o consultar en los comentarios.

**Archivos de la clase**

[n8nself.env](https://static.platzi.com/media/public/uploads/n8nself_fc0631e8-c7f3-47e6-9a1c-b38fabf46aa3.env)

**Lecturas recomendadas**

[n8nself.env - Google Drive](https://drive.google.com/file/d/1ko-sVsQruotUCoZ1BnvSET_3a6DwZUMK/view?usp=drive_link)

[Cloud Application Platform | Render](https://www.render.com/)

[Supabase | The Open Source Firebase Alternative](https://supabase.com/)

[RandomKeygen - The Secure Password & Keygen Generator](https://randomkeygen.com/)

## Creación de formularios y conexión con Google Sheets en N8N

Crear formularios y conectar sus respuestas con **Google Sheets en n8n** es una de las automatizaciones más populares y útiles. A continuación te explico **paso a paso** cómo hacerlo, usando una de estas dos estrategias:

### ✅ OPCIÓN 1: Usar un formulario externo (como Typeform, Google Forms o Tally) y conectarlo a Google Sheets con n8n

### 1. Crea el formulario (en una plataforma externa)

Puedes usar:

* [Google Forms](https://forms.google.com)
* [Tally.so](https://tally.so)
* [Typeform](https://typeform.com)
* [Jotform](https://jotform.com)

> Asegúrate de que tenga un webhook o integración para notificar a n8n cuando se envía una respuesta.

### 2. Crea un **Webhook** en n8n

* Añade un nodo `Webhook`.
* Método: `POST`
* Path: `/formulario`
* Guarda y copia la URL del webhook.
* Abre en navegador una vez para activar el endpoint.

### 3. Conecta tu formulario al webhook

En tu plataforma de formularios:

* Busca la opción de **Webhooks** o **Notificaciones HTTP**
* Usa la URL del webhook de n8n.
* Asegúrate de enviar los datos como JSON (la mayoría lo hace por defecto).

### 4. Añade un nodo `Google Sheets`

* Autentícate con tu cuenta de Google.
* Selecciona tu hoja de cálculo.
* Acción: `Append Sheet Row`
* Mapea los campos del formulario al archivo de Google Sheets.

### 5. Ejecuta el flujo y prueba

* Envía el formulario.
* Verifica que la hoja de cálculo se actualice correctamente.

### ✅ OPCIÓN 2: Crear un formulario HTML simple alojado por ti y enviar datos a un webhook de n8n

Puedes usar HTML y JavaScript para enviar los datos directamente:

### Ejemplo de formulario:

```html
<form id="myForm">
  <input type="text" name="nombre" placeholder="Tu nombre" required />
  <input type="email" name="email" placeholder="Tu correo" required />
  <button type="submit">Enviar</button>
</form>

<script>
document.getElementById('myForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const data = new FormData(e.target);
  const obj = Object.fromEntries(data.entries());

  await fetch("https://TU_INSTANCIA_N8N/webhook/formulario", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(obj)
  });

  alert("Formulario enviado");
});
</script>
```

Este formulario envía los datos directamente a n8n sin intermediarios.

### 📋 Requisitos para Google Sheets en n8n

1. **Credenciales OAuth2 de Google**: Puedes usar la integración directa o configurar tu propia app en Google Cloud.
2. Asegúrate de tener una hoja creada y accesible.
3. n8n debe tener permisos para leer/escribir en tu Google Drive.

### ¿Quieres que te genere una plantilla completa de flujo en n8n (JSON)?

Puedo hacerlo según la herramienta de formularios que elijas o ayudarte a desplegar el formulario HTML. Solo dime qué prefieres usar:

* Google Forms
* Tally.so
* HTML personalizado
* Otra opción

### Resumen

Optimiza tus procesos digitales creando flujos automatizados con N8N. Este contenido te guiará paso a paso para generar formularios y vincular fácilmente las respuestas con Google Sheets. Aprende a configurar correctamente cada elemento y simplifica tareas repetitivas.

#### ¿Cómo crear un flujo con formularios en N8N?

Lo primero es acceder a la interfaz y seleccionar Create Workflow en la esquina superior derecha. Al crear un nuevo flujo:

- Busca **N8N** y escoge el elemento **On New N8N Form Event**.
- Asigna un título al formulario, por ejemplo, **Form 1**.
- Agrega campos al formulario:
- **Nombre** (tipo texto).
- **E-mail** (tipo email).

Al dar clic en **test step**, el nodo estará listo mostrando claramente los inputs definidos.

#### ¿De qué forma se vinculan los datos recolectados a Google Sheets?

Luego de configurar el formulario, vincula la información recolectada al servicio de hojas de cálculo Google Sheets de la siguiente manera:

- Presiona en el botón **Más** e ingresa **Google Sheets**.
- Selecciona la opción **Append Row in a Sheet**, diseñada para ingresar registros nuevos de forma consecutiva.
- Autoriza la conexión mediante **Sign in with Google**, vinculando correctamente tu cuenta.
- Selecciona la hoja correspondiente, en este caso, **Forms**.
- Configura manualmente los campos usando **Map Each Column Manually**, relacionando **nombre** y **e-mail** con las filas pertinentes.

Aplica un testeo para confirmar la carga exitosa de datos en Google Sheets.

#### ¿Qué funcionalidades tiene la interfaz de N8N?

La interfaz, llamada Canvas, permite interactuar con los flujos desde diversas opciones:

- **Zoom in/out**: ajusta la visualización.
- **Overview**: resumen general de los flujos.
- **Credenciales**: gestión de cualquier herramienta integrada.
- **Templates**: plantillas disponibles para acelerar diseños.
- **Variables e Insights**: disponibles exclusivamente bajo licencia paga, permiten configuraciones avanzadas como variables de entorno y revisión de estadísticas de rendimiento.

Comparte tu experiencia o capturas del flujo desarrollado en la sección de comentarios.

## Configuración de bot de Telegram para registro automático de gastos

Crear un **bot de Telegram para registrar automáticamente gastos** con **n8n** es totalmente posible y muy útil para tu organización financiera. A continuación, te explico cómo configurarlo paso a paso:

### ✅ OBJETIVO

Usar un **bot de Telegram** para registrar mensajes con formato tipo:
`💸 Comida 25`
y que n8n los agregue automáticamente a un **Google Sheet** u otra base de datos (Airtable, Notion, PostgreSQL, etc.).

### 🔧 PASO 1: Crear un bot de Telegram

1. Abre Telegram y busca **@BotFather**.
2. Usa el comando `/newbot`.
3. Asigna un nombre y un username a tu bot.
4. BotFather te dará un **Token**, guárdalo.

### 🔧 PASO 2: Crear el flujo en n8n

### 1. Nodo `Telegram Trigger`

* Tipo: `Telegram Trigger`
* **Authentication**: agrega tu token de bot.
* Tipo de evento: `Message`

### 2. Nodo `Set` (opcional)

* Organiza o mapea datos como `chat_id`, `text`, `fecha`, etc.

### 3. Nodo `Function` (para extraer datos)

```javascript
const text = $json.message.text;
const parts = text.trim().split(' ');

// Ej: "💸 Comida 25"
return [{
  categoria: parts[1] || 'Sin categoría',
  valor: parseFloat(parts[2]) || 0,
  fecha: new Date().toISOString().split('T')[0], // solo la fecha
}];
```

### 4. Nodo `Google Sheets` (u otra base)

* Acción: `Append Sheet Row`
* Mapea los campos:

  * Fecha → `{{$json.fecha}}`
  * Categoría → `{{$json.categoria}}`
  * Valor → `{{$json.valor}}`

### 🧠 Resultado

Cada vez que envíes un mensaje como `💸 Transporte 15`, el gasto será registrado en tu hoja de cálculo o base de datos.

### ✅ OPCIONAL

* **Validaciones:** puedes agregar nodos `IF` para asegurar que el mensaje tenga formato válido.
* **Respuesta automática:** usa el nodo `Telegram` → `Send Message` para confirmar el registro.

### Resumen

Automatizar el registro de gastos cotidianos puede simplificar ampliamente nuestras finanzas personales. Usando Telegram junto con herramientas como n8n y Google Sheets, puedes crear fácilmente un sistema que guarde automáticamente tus comprobantes para evitar pérdida o extravío.

#### ¿Qué necesito para empezar con la automatización de gastos?

Antes que nada, asegúrate de tener una cuenta en Telegram y en n8n. Estos servicios permiten automatizar procesos sin necesidad de programar demasiado.

Tu primer paso es crear un bot en Telegram. Lo puedes hacer desde Telegram Web o desde tu móvil, usando Botfather (el usuario con el chulito azul) y siguiendo estas instrucciones:

- Abre Telegram y coloca en búsqueda "Botfather".
- Haz clic en Start y escribe `/newbot` para comenzar.
- Asigna un nombre fácil, como "Registra gastos Platzi".
- Finaliza siempre el nombre del bot con la palabra "bot".

Obtendrás un token. Recuerda guardarlo en un lugar seguro, pues lo necesitarás para configurar el siguiente paso en N8n.

#### ¿Cómo configurar el workflow en n8n usando Telegram?

Una vez listo el bot, ingresa a N8n y realiza la siguiente configuración:

1. Crea un nuevo flujo seleccionando "Create Workflow".
2. Añade un nodo de Telegram en Triggers, específicamente "On Message".
3. Vincula este nodo con una nueva credencial donde colocarás el access token del bot.

Luego, añade otro nodo adicional de Telegram llamado "Get a file" para descargar los archivos que se envíen por Telegram, configurándolo correctamente con los datos obtenidos previamente.

Para convertir el archivo descargado en un formato legible por otras herramientas, como el campo mimetype de Telegram a binario (ceros y unos), necesitarás insertar un nodo de código.

#### ¿Cómo transformar archivos de Telegram a formato binario?

Para que tu archivo pueda ser procesado por otras plataformas, por ejemplo, OpenAI, sigue estos pasos:

- Inserta un nodo de código en el flujo en N8N.
- Deja predeterminado el modo "run once for all items" y lenguaje JavaScript (podrías usar Python si quieres).
- Emplea el script proporcionado (disponible en los recursos adicionales de la clase), cuya función principal es convertir la extensión mime del archivo descargado en un archivo binario adecuado para servir como insumo en plataformas de automatización posteriores.

Toma en cuenta que este paso es vital si la siguiente herramienta que utilizarás, como OpenAI, solo interpreta formatos binarios.

¿Te has animado ya a probar esta práctica automatización?‌ Tu experiencia e ideas pueden aportar mucho; ¡compártenos cómo te fue!

**Archivos de la clase**

[codigo-js-mime-a-binary.txt](https://static.platzi.com/media/public/uploads/codigo-js-mime-a-binary_13b6407b-9726-4e2b-926f-0868a715a76c.txt)