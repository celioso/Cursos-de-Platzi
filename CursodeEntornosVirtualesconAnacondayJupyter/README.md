# Curso de Entornos Virtuales con Anaconda y Jupyter

## Creación de Entornos Virtuales en Python con PIP y Venv

#### 📌 ¿Qué es un entorno virtual?

Un **entorno virtual** en Python es una carpeta aislada que contiene su propio intérprete de Python y sus propios paquetes, separados del sistema principal. Esto evita conflictos entre proyectos.

### 🧰 Requisitos previos

* Tener **Python 3.3+** instalado.
* `pip` (normalmente se instala automáticamente con Python).

Puedes verificarlo con:

```bash
python --version
pip --version
```

### 🚀 Pasos para crear un entorno virtual con `venv`

#### 1. **Crear el entorno virtual**

```bash
python -m venv nombre_entorno
```

* Esto creará una carpeta llamada `nombre_entorno` con una instalación independiente de Python.

#### 2. **Activar el entorno virtual**

* **Windows**:

```bash
nombre_entorno\Scripts\activate
```

* **macOS/Linux**:

```bash
source nombre_entorno/bin/activate
```

> Al activar, verás el nombre del entorno al inicio de la línea de comandos, por ejemplo: `(nombre_entorno)`

### 📦 Instalar paquetes con `pip`

Una vez dentro del entorno virtual, puedes instalar paquetes con `pip`:

```bash
pip install numpy pandas
```

Puedes ver los paquetes instalados con:

```bash
pip list
```

Y guardar las dependencias:

```bash
pip freeze > requirements.txt
```

Para recrear el entorno en otra máquina:

```bash
pip install -r requirements.txt
```

### 🔚 Desactivar el entorno

Cuando termines:

```bash
deactivate
```

### Resumen

#### ¿Qué son los entornos virtuales en Python y por qué son esenciales?

Imagina que estás en medio de un intenso proyecto de análisis de datos financieros para tu empresa. Utilizas bibliotecas como Pandas y Matplotlib, y todo marcha perfectamente. Pero, al iniciar un nuevo proyecto de Machine Learning, de repente, tus bibliotecas entran en conflicto. Todo deja de funcionar como esperabas. Esto ocurre cuando diferentes proyectos comparten versiones de bibliotecas que no son compatibles o, incluso, diferentes versiones de Python. Aquí es donde los entornos virtuales entran en juego y se vuelven indispensables.

Un **entorno virtual** en Python es una instancia aislada que te permite instalar bibliotecas independientemente para cada proyecto. Esto evita que las dependencias de un proyecto impacten negativamente en otro, además de ofrecer otras ventajas importantes:

- **Aislamiento**: Cada proyecto cuenta con su entorno propio, evitando conflictos entre bibliotecas.
- **Reproducibilidad**: Facilita que otros desarrolladores puedan reproducir tu entorno en sus máquinas, garantizando que el proyecto funcione igual en todos los sistemas.
- **Organización**: Mantiene tus proyectos organizados al controlar rigurosamente las bibliotecas utilizadas.

#### ¿Cómo crear un entorno virtual en Python?

Vamos a dar el primer paso hacia el uso eficiente de Python con entornos virtuales. Crear un entorno virtual es un procedimiento sencillo que se lleva a cabo en pocos pasos desde la terminal de tu computador.

1. **Abrir la terminal**: Inicia abriendo la terminal de tu computador.

2. **Navegar hasta la carpeta deseada**: Desplázate hasta la carpeta donde deseas guardar tus entornos. Por ejemplo, Virtual Environments.

3. **Crear el entorno virtual**: Ejecuta el siguiente comando:

`python3 -m venv MyEnv`

Reemplaza "MyEnv" con el nombre que prefieras para tu entorno virtual.

4. Activar el entorno virtual:

- En Mac y Linux: `bash source MyEnv/bin/activate`
- En Windows: `bash MyEnv\Scripts\activate`

Una vez activado, verás el nombre del entorno al inicio de la línea de comandos, indicando que estás dentro del entorno virtual. Ahora estás listo para instalar los paquetes necesarios en un lugar aislado y sin preocupaciones.

#### ¿Cómo instalar paquetes dentro de un entorno virtual?

Ahora que has creado y activado tu entorno virtual, probablemente desees instalar paquetes esenciales para tu proyecto sin afectar el sistema global. Para esto, usarás pip, el manejador de paquetes de Python:

- Instalar un paquete (por ejemplo, Pandas y Matplotlib):

`pip install pandas matplotlib`

Este comando no solo instala los paquetes, sino que también se ocupa de todas sus dependencias necesarias.

Puedes verificar que las instalaciones han sido exitosas abriendo el intérprete de Python y ejecutando:

```python
import pandas
import matplotlib
```

Si no encuentras errores, significa que los paquetes están perfectamente instalados dentro de tu entorno virtual.

#### ¿Qué sucede fuera del entorno virtual?

Al desactivar el entorno virtual con el comando `deactivate`, regresas al entorno global de tu sistema. Aquí, si intentas importar los mismos paquetes que instalaste en el entorno virtual usando `import pandas`, por ejemplo, podrías encontrar errores de módulo no encontrado. Esto confirma que las bibliotecas instaladas en el entorno virtual no afectan ni interfieren con tu configuración global.

#### ¿Cuándo es útil usar Anaconda?

Aunque ya dominas la creación de entornos con `venv` y `pip`, existe una herramienta que puede facilitarte aún más la vida en proyectos de ciencia de datos y Machine Learning complejos: **Anaconda**. Esta plataforma robusta es ideal en las siguientes situaciones:

- **Proyectos complejos**: Si trabajas con voluminosos datos, análisis complejos o modelos avanzados.
- **Paquetes difíciles de instalar**: Anaconda simplifica la instalación de paquetes que requieren compilación o dependen de bibliotecas externas.
- **Entornos reproducibles**: Facilita compartir y replicar proyectos sin preocuparse por diferencias de versiones.

Anaconda viene con más de 250 bibliotecas listas para usar, incluidas las más populares herramientas de ciencia de datos como NumPy, Scikit Learn, y, obviamente, Pandas y Matplotlib.

La creación de entornos con `venv` y `pip` es un excelente punto de partida, pero imagina un flujo de trabajo donde una herramienta se ajusta a tus proyectos en ciencia de datos con precisión y eficiencia. Descubre el poder de Anaconda en tus siguientes aventuras en Machine Learning y transforma tu manera de abordar proyectos especializados.

## Instalación de Anaconda en Windows, Linux y Mac

Aquí tienes una guía clara para instalar **Anaconda** en **Windows, Linux y macOS**. Anaconda es una distribución de Python con más de 1.500 paquetes para ciencia de datos, machine learning, y análisis.

### 💻 1. Windows

### ✅ Pasos:

1. **Descarga el instalador**
   Ve a [https://www.anaconda.com/download](https://www.anaconda.com/download) y descarga la versión para Windows (64-bit, Graphical Installer).

2. **Ejecuta el instalador**

   * Haz doble clic en el `.exe`.
   * Acepta términos y condiciones.
   * Instálalo "Just for me".
   * Marca “Add Anaconda to PATH” (opcional pero útil).
   * Finaliza la instalación.

3. **Verifica instalación**
   Abre *Anaconda Prompt* y ejecuta:

   ```bash
   conda --version
   ```

### 🐧 2. Linux

### ✅ Pasos:

1. **Descarga el instalador**
   Desde terminal:

   ```bash
   wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
   ```

2. **Ejecuta el instalador**

   ```bash
   bash Anaconda3-2024.02-1-Linux-x86_64.sh
   ```

   * Presiona Enter para avanzar.
   * Acepta la licencia.
   * Elige la ruta de instalación (por defecto está bien).
   * Al final, te preguntará si deseas añadir Anaconda al `PATH`: responde `yes`.

3. **Reinicia la terminal** y verifica:

   ```bash
   conda --version
   ```

### 🍎 3. macOS

### ✅ Pasos:

1. **Descarga el instalador**
   Desde [https://www.anaconda.com/download](https://www.anaconda.com/download) selecciona la versión para macOS (Intel o Apple Silicon).

2. **Instalación gráfica o por terminal**

   * Ejecuta el `.pkg` si es gráfico.
   * O usa terminal:

     ```bash
     bash Anaconda3-2024.02-1-MacOSX-x86_64.sh
     ```

3. **Sigue los pasos del instalador** y confirma agregar Anaconda al `PATH`.

4. **Verifica instalación:**

   ```bash
   conda --version
   ```

### 🧪 Verifica entorno base y acceso a Jupyter

```bash
conda info --envs
jupyter notebook
```

### Resumen

#### ¿Por qué es popular Anaconda entre los desarrolladores de ciencia de datos?

Anaconda es una plataforma indispensable para más de 20 millones de desarrolladores en el mundo de la ciencia de datos y Machine Learning. Grandes empresas como Facebook, NASA y Tesla la eligen para desarrollar modelos de inteligencia artificial y gestionar proyectos complejos. Con Anaconda, los científicos de datos pueden manejar cientos de paquetes y librerías de Python y R dentro de entornos virtuales controlados, garantizando estabilidad y reproducibilidad en sus proyectos.

#### ¿Cuáles son las principales ventajas de usar Anaconda?

1. **Gestión de entornos virtuales**: Con Conda, puedes crear y gestionar entornos virtuales específicos para cada proyecto, evitando conflictos de dependencias.

2. **Instalación simplificada de paquetes**: La instalación de paquetes como Numpy, Pandas o Scikit Learn se realiza con un solo comando.

3. **Incorporación de Jupyter Notebooks**: Las notebooks originales vienen preinstaladas, facilitando el desarrollo y la presentación de proyectos.

#### ¿Cómo instalar Anaconda en Windows, Linux y Mac?

La instalación de Anaconda puede variar ligeramente dependiendo del sistema operativo. Aquí te explicamos cómo hacerlo para Windows, Linux (utilizando WSL) y Mac:

#### ¿Cómo instalar Anaconda en Linux usando WSL?

1. Descarga el instalador de Linux desde la página de Anaconda.

2. Copia el archivo .sh a la carpeta home de tu distribución Linux (por ejemplo, Ubuntu) dentro de WSL.

3. Abre una terminal en la misma carpeta y ejecuta el comando:

`bash nombre-del-archivo.sh`

4. Acepta los términos y condiciones y sigue los pasos para completar la instalación.

5. Verifica la instalación con:

`conda env list`

####¿Cómo instalar Anaconda en Windows?

1. Descarga el instalador para Windows desde la página de Anaconda.

2. Ejecuta el instalador y sigue las instrucciones en pantalla, aceptando los términos y condiciones.

3. Decide si instalar para todos los usuarios o solo para tu cuenta, y elige la opción de añadir Anaconda al PATH si lo deseas.

4. Verifica la instalación usando Anaconda Prompt o PowerShell:

`conda info`

#### ¿Cómo instalar Anaconda en Mac?

1. En la página de descargas de Anaconda, elige la opción adecuada para tu tipo de procesador (Apple Silicon o Intel Chip).

3. Descarga y ejecuta el instalador gráfico o emplea la línea de comandos según prefieras.

5. Acepta los términos y condiciones y completa el proceso de instalación.

7. Verifica que Anaconda está instalado correctamente mediante la terminal:

`conda info`

#### ¿Cuáles son las diferencias entre Anaconda, Miniconda y PIP?

Anaconda, Miniconda y PIP son herramientas para la administración de paquetes en Python, pero cada una tiene sus particularidades:

- **Anacond**a:

- Incluye más de 250 paquetes preinstalados.

- Es ideal para proyectos de ciencia de datos, Machine Learning e inteligencia artificial.

- **Miniconda**:

- Proporciona una instalación más pequeña con menos de 70 paquetes.

- Recomendado para quienes desean más control y saben qué paquetes necesitarán.

- **PIP**:

- No incluye paquetes preinstalados y es más general.

- Útil en cualquier ámbito de desarrollo en Python.

#### Consejos para el uso de Anaconda en proyectos de ciencia de datos

- Familiarízate con los comandos principales de Conda, especialmente si trabajas en un entorno profesional.
- Explora el cheat sheet de Anaconda disponible en PDF para comprender mejor sus capacidades.
- No dudes en usar los comentarios de foros o plataformas educativas para consultar dudas sobre la instalación o uso en diferentes sistemas operativos.

Recuerda que la práctica constante y la familiarización con las herramientas son clave para dominar la ciencia de datos con plataformas como Anaconda. ¡Explora, practica y evoluciona en tu carrera en este fascinante campo!

**Lecturas recomendadas**

[Linux en Windows y Windows en Linux (Tutorial WSL) - YouTube](https://www.youtube.com/watch?v=Qy44XLpiChc&t=582s "Linux en Windows y Windows en Linux (Tutorial WSL) - YouTube")

[Download Anaconda Distribution | Anaconda](https://www.anaconda.com/download "Download Anaconda Distribution | Anaconda")

[Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf "Cheat Sheet")

## Gestión de Entornos Virtuales con Conda y Anaconda

La **gestión de entornos virtuales con Conda y Anaconda** es fundamental para mantener tus proyectos de Python organizados, con versiones específicas de paquetes sin interferencias entre sí.

### 🔧 ¿Qué es un entorno virtual?

Un entorno virtual es un espacio aislado donde puedes instalar una versión específica de Python y paquetes, sin afectar otros entornos o la instalación global del sistema.

### 📦 Conda vs. Anaconda

* **Anaconda** es una distribución de Python que incluye Conda, Jupyter, y más de 1.500 paquetes científicos.
* **Conda** es el gestor de entornos y paquetes usado por Anaconda (también se puede usar solo, vía Miniconda).

### 🛠️ Comandos esenciales de Conda para gestionar entornos

#### 1. **Crear un nuevo entorno**

```bash
conda create --name mi_entorno python=3.10
```

#### 2. **Listar entornos existentes**

```bash
conda env list
```

#### 3. **Activar un entorno**

```bash
conda activate mi_entorno
```

#### 4. **Desactivar el entorno actual**

```bash
conda deactivate
```

#### 5. **Eliminar un entorno**

```bash
conda remove --name mi_entorno --all
```

#### 6. **Instalar un paquete en un entorno activo**

```bash
conda install pandas
```

#### 7. **Exportar los paquetes de un entorno**

```bash
conda env export > environment.yml
```

#### 8. **Crear un entorno desde un archivo .yml**

```bash
conda env create -f environment.yml
```

### ✅ Buenas prácticas

* Usa un entorno por proyecto.
* Usa `environment.yml` para facilitar la reproducción del entorno en otros equipos.
* Usa `conda activate` antes de ejecutar scripts o notebooks para asegurarte de que los paquetes se cargan del entorno correcto.

### Recursos

#### ¿Por qué es importante gestionar entornos virtuales en Python?

Gestionar entornos virtuales en Python es esencial para desarrollar múltiples proyectos de manera eficiente y organizada. Permite mantener las dependencias y versiones de paquetes aisladas entre sí, lo que evita conflictos y errores en el código. Imagina trabajar simultáneamente en proyectos que requieren diferentes versiones de Python o librerías: sin un entorno virtual, podrías enfrentar complicaciones innecesarias. Es aquí donde Conda, el gestor de entornos de Anaconda, juega un papel crucial al ofrecerte la posibilidad de crear ambientes separados para cada proyecto.

#### ¿Cómo crear y gestionar entornos virtuales con Conda?

Para crear y gestionar entornos virtuales con Conda, es importante seguir una serie de pasos bien definidos, lo que garantiza un manejo óptimo de los recursos y una organización coherente.

#### ¿Cómo crear un entorno virtual?

Para empezar, asegúrate de tener Anaconda instalado y estar en el base neutral. El siguiente comando crea un nuevo entorno virtual:

`conda create --name example`

Si deseas configurar una versión específica de Python, simplemente añade el número de la versión al final del comando:

`conda create --name new_env python=3.9`

#### ¿Cómo activar y desactivar un entorno virtual?

Una vez creado, el entorno debe activarse para poder trabajar de manera efectiva:

`conda activate example`

Para volver a la línea de base o desactivar cualquier entorno activo, usa:

`conda deactivate`

#### ¿Cómo verificar la versión de Python en uso?

Después de activar un entorno, es recomendable verificar la versión de Python para asegurarse de su correcta configuración:

`python --version`

Esto ayuda a confirmar que el entorno está configurado con la versión esperada, minimizando errores debido a discrepancias en versiones.

#### ¿Cómo instalar paquetes en un entorno virtual?

Instalar paquetes en el entorno correcto es crucial para evitar conflictos. Asegúrate de estar en el entorno adecuado antes de instalar cualquier librería. Por ejemplo, para instalar NumPy y Pandas en tu entorno virtual, usa:

`conda install numpy pandas`

Esto garantiza que las librerías se instalen solo en el entorno activo, aisladas de otros proyectos.

#### ¿Cómo verificar paquetes instalados?

Para verificar qué paquetes están instalados en un entorno específico, utiliza el comando:

`conda list`

Si deseas verificar los paquetes de un entorno específico sin activarlo, indica el nombre del entorno:

`conda list --name example`

#### ¿Cómo revisar todos los entornos existentes?

Con frecuencia, es útil tener a mano una lista de todos los entornos disponibles. Para obtener esta lista, utiliza el siguiente comando:

`conda env list`

Esto arroja una lista de todos los entornos creados, permitiéndote gestionar fácilmente tu espacio de trabajo.

Aplica estos conocimientos y experimenta creando tu entorno virtual con una versión de Python específica y las siguientes librerías: Pandas, Scikit Learn y Matplotlib. Comparte tus aprendizajes y resultados en los comentarios. A medida que practiques, ganarás confianza y eficiencia en la gestión de tus proyectos de datos. ¡Sigue explorando y aprendiendo!

#### El error que ves es porque estás intentando ejecutar:

```powershell
conda init example
```

Y `example` **no es una opción válida** para `conda init`.

### ✅ Cómo usar correctamente `conda init`

Este comando se usa para inicializar Conda en el shell que estás utilizando (como PowerShell o CMD), para que puedas usar `conda activate` sin errores.

#### Si estás usando **PowerShell** (como en tu caso):

```powershell
conda init powershell
```

#### Si estás usando **CMD**:

```cmd
conda init cmd.exe
```

#### Si estás en una terminal de **Git Bash o WSL**:

```bash
conda init bash
```

### 🔄 Después de ejecutar `conda init`:

1. **Cierra** la terminal.
2. **Ábrela nuevamente** para que los cambios surtan efecto.
3. Luego prueba:

   ```powershell
   conda activate base
   ```

#### Eliminar

Para **eliminar un entorno (ambiente) de Conda**, utiliza el siguiente comando desde tu terminal o PowerShell:

```bash
conda env remove --name nombre_del_entorno
```

### ✅ Ejemplo:

Si tu entorno se llama `fundamentos-etl`, ejecuta:

```bash
conda env remove --name fundamentos-etl
```

### 🔎 Paso para verificar que se eliminó:

Después de ejecutar el comando, puedes verificar que ya no está con:

```bash
conda env list
```

Esto mostrará los entornos restantes.



**Lecturas recomendadas**

[Environments — Anaconda documentation](https://docs.anaconda.com/working-with-conda/environments/)

[Installing conda packages — Anaconda documentation](https://docs.anaconda.com/working-with-conda/packages/install-packages/)

## Gestión y Limpieza de Entornos Virtuales con Conda

La **gestión y limpieza de entornos virtuales con Conda** es esencial para mantener tu sistema organizado, ahorrar espacio en disco y evitar conflictos entre dependencias. A continuación te explico cómo hacerlo paso a paso:

### 🧰 1. **Ver entornos existentes**

Lista todos los entornos creados:

```bash
conda env list
```

o

```bash
conda info --envs
```

### 🔄 2. **Activar un entorno**

Antes de trabajar en él:

```bash
conda activate nombre_del_entorno
```

### 🚪 3. **Desactivar un entorno**

Cuando termines de usarlo:

```bash
conda deactivate
```

### ❌ 4. **Eliminar un entorno**

Para limpiar los entornos que ya no usas:

```bash
conda env remove --name nombre_del_entorno
```

Ejemplo:

```bash
conda env remove --name fundamentos-etl
```

### 🧼 5. **Limpiar caché de paquetes**

Conda guarda paquetes descargados; puedes liberar espacio con:

```bash
conda clean --all
```

Opciones comunes:

* `--tarballs`: elimina archivos `.tar.bz2` descargados
* `--index-cache`: borra caché del índice de paquetes
* `--packages`: elimina paquetes que no están en ningún entorno

Ejemplo completo:

```bash
conda clean --all --yes
```

### 🔍 6. **Revisar el tamaño de cada entorno (opcional, con Anaconda Navigator)**

Si usas la interfaz gráfica de Anaconda, puedes ver el tamaño y contenido de cada entorno fácilmente desde "Environments".

### Resumen

#### ¿Cómo gestionar y limpiar entornos virtuales en Conda?

La gestión eficiente de entornos virtuales en Conda es esencial para garantizar un desarrollo sin complicaciones. Al manejar múltiples entornos y librerías, es fácil que el sistema se sature con elementos no necesarios, lo que puede llevar a confusiones y problemas de almacenamiento. ¿Cómo podemos optimizar el espacio de nuestros sistemas y mantener el control sobre los paquetes que realmente necesitamos? Aquí te lo explicamos paso a paso.

#### ¿Cómo listar y eliminar entornos virtuales?

Para comenzar a limpiar, primero necesitamos conocer qué entornos virtuales hemos creado.

1. **Listar entornos virtuales**: Usamos el siguiente comando para obtener una lista completa.

`conda env list`

Esto nos mostrará, por ejemplo: `base`, `example` y `newenv`.

2. **Eliminar un entorno virtual**: Si decides que un entorno (como newenv) ya no es necesario, puedes eliminarlo completamente. Asegúrate de especificar que también quieres eliminar todos sus paquetes y dependencias.

`conda remove -n newenv --all`

Después de ejecutar el comando, verifica que el entorno haya sido eliminado listando de nuevo los entornos.

#### ¿Cómo manejar paquetes dentro de un entorno?

A veces, podrías orientar tus esfuerzos de limpieza a nivel de paquetes especificos dentro de un entorno.

1. **Activar un entorno virtual**: Antes de eliminar paquetes, activa el entorno donde están instalados.

`conda activate example`

2. **Listar los paquetes instalados**: Obtén una lista de los paquetes actuales en el entorno.

`conda list`

3. **Remover paquetes innecesarios**: Elimina los paquetes que ya no necesitas, como pandas, mediante el siguiente comando.

`conda remove pandas`

Vuelve a listar los paquetes para asegurar que `pandas` ya no está presente, mientras que otros, como `NumPy`, permanecen intactos.

#### ¿Cómo limpiar la caché de paquetes en Conda?

Los paquetes descargados pero no utilizados pueden consumir un espacio considerable en tu sistema. Limpiar la caché es, por tanto, una tarea crucial.

1. **Limpiar parcialmente la caché**: Puedes empezar eliminando los paquetes no necesarios.

`conda clean --packages`

Es recomendable hacer esto regularmente para liberar espacio adicional.

2. **Limpiar completamente toda la caché**: Si deseas eliminar todos los paquetes sin uso y otros archivos manejados globalmente, utiliza:

`conda clean --all`

#### Tarea práctica

Para consolidar estos conceptos, crea un nuevo entorno virtual llamado `entorno_tarea` e instala Python 3.9. Después, adiciona las librerías `Pandas`, `Matplotlib` y `Scikit-learn`. Luego, elimina `Matplotlib` y verifica la lista de paquetes para asegurarte de que se ha removido correctamente. Por último, limpia la caché de este entorno para dejar tu sistema en óptimas condiciones.

No dudes en experimentar y familiarizarte con estos comandos para dominar la gestión de entornos virtuales en tu flujo de trabajo. ¡La organización y el control son clave para un desarrollo eficaz!

## Gestión de Entornos Virtuales y Paquetes con Conda

La **gestión de entornos virtuales y paquetes con Conda** es fundamental para mantener tus proyectos de Python (y otros lenguajes) organizados, evitando conflictos entre librerías y versiones. A continuación, te explico cómo manejarlo:

### 🧪 ¿Qué es Conda?

**Conda** es un gestor de entornos y paquetes que permite:

* Crear entornos virtuales independientes.
* Instalar paquetes específicos en cada entorno.
* Aislar proyectos para evitar conflictos de dependencias.

### 🧰 1. Gestión de Entornos Virtuales

### 🔹 Crear un entorno nuevo:

```bash
conda create --name mi_entorno python=3.10
```

### 🔹 Listar entornos disponibles:

```bash
conda env list
```

### 🔹 Activar un entorno:

```bash
conda activate mi_entorno
```

### 🔹 Desactivar un entorno:

```bash
conda deactivate
```

### 🔹 Eliminar un entorno:

```bash
conda remove --name mi_entorno --all
```

### 📦 2. Gestión de Paquetes

### 🔹 Instalar un paquete:

```bash
conda install numpy
```

O desde un canal específico (ej: conda-forge):

```bash
conda install -c conda-forge pandas
```

### 🔹 Listar paquetes instalados:

```bash
conda list
```

### 🔹 Actualizar un paquete:

```bash
conda update nombre_paquete
```

### 🔹 Eliminar un paquete:

```bash
conda remove nombre_paquete
```

### 📁 3. Exportar y Reproducir Entornos

### 🔹 Exportar entorno a un archivo:

```bash
conda env export > environment.yml
```

### 🔹 Crear un entorno desde un archivo:

```bash
conda env create -f environment.yml
```

### 🛠️ Consejo

* Usa **entornos virtuales separados por proyecto** para evitar conflictos.
* Prefiere `conda` sobre `pip` dentro de entornos Conda para mantener la compatibilidad.
* Puedes combinar Conda y Pip con cuidado.

### Resumen

#### ¿Cómo actualizar paquetes en entornos virtuales con Conda?

Mantener tus paquetes actualizados es vital para garantizar el buen funcionamiento de tus proyectos, ya que las actualizaciones traen nuevas funcionalidades y parches de seguridad. La herramienta Conda facilita este proceso de manera efectiva y sencilla. Para comenzar, activa el entorno virtual donde deseas realizar la actualización. Utiliza el comando:

`conda activate example`

Una vez dentro del entorno, actualiza un paquete específico, como numpy, con:

`conda update numpy`

Conda te indicará si el paquete ya está actualizado o procederá con la actualización. También puedes actualizar todos los paquetes de tu entorno con:

`conda update --all`

Recuerda siempre probar estos cambios en entornos de desarrollo antes de llevarlos a producción, para evitar conflictos o problemas de compatibilidad.

#### ¿Cómo clonar un entorno virtual con Conda?

A veces, necesitas hacer cambios significativos en un entorno sin comprometer el proyecto original. La solución es clonar el entorno. Comienza desactivando el entorno actual con:

`conda deactivate`

Luego, para clonar el entorno, usa:

`conda create --name new_example --clone example`

Una vez completado, puedes activar el nuevo entorno con:

`conda activate new_example`

Puedes verificar que la clonación se hizo correctamente listando todos los entornos disponibles:

`conda env list`

#### ¿Cómo exportar y compartir entornos virtuales?

Exportar un entorno te permite compartirlo o replicarlo en otras máquinas, algo esencial al trabajar en equipo. El archivo `.yml` que Conda genera contiene toda la información necesaria. Para exportar tu entorno, activa el entorno deseado y usa:

```bash
conda activate sample
conda env export > environment.yml
```

Con este archivo creado, puedes explorar su contenido con:

`cat environment.yml`

Este archivo indica el nombre del entorno, los canales y las librerías instaladas incluyendo sus versiones. Para recrear un entorno a partir de este archivo en otra máquina, elimina primero cualquier entorno previo con:

```bash
conda deactivate
conda remove --name example --all
```

Y luego recrea el entorno usando el archivo `.yml`:

`conda env create -f environment.yml`

#### ¿Es posible instalar librerías a partir de un archivo `.yml`?

Claro que sí, puedes instalar librerías definidas en un archivo `.yml`. Supongamos que se te ha compartido un archivo con ciertas librerías, como pandas o matplotlib. Explora el archivo con:

`cat env.yml`

Para agregar estas librerías a un nuevo entorno, crea primero el entorno:

`conda create --name my_env`

Actívalo y luego actualiza con las librerías del archivo:

```bash
conda activate my_env
conda env update -f env.yml
```

Verifica los paquetes instalados con:

`conda list`

Al finalizar, recuerda desactivar tu entorno para mantener todo ordenado:

`conda deactivate`

Implementar estas prácticas no solo asegura la continuidad de tu proyecto, sino también facilita la colaboración en equipo y la gestión de dependencias. ¡Continúa aprendiendo y experimenta con Conda para optimizar tus flujos de trabajo!

## Gestión de Entornos Virtuales con Anaconda Navigator

La **gestión de entornos virtuales con Anaconda Navigator** te permite crear, clonar, eliminar y administrar entornos fácilmente a través de una interfaz gráfica, sin necesidad de usar la terminal. Es ideal para usuarios que prefieren no trabajar con la línea de comandos.

### 🧭 ¿Qué es Anaconda Navigator?

Anaconda Navigator es una aplicación gráfica incluida con Anaconda que facilita:

* Manejo de entornos virtuales.
* Instalación de paquetes.
* Lanzamiento de herramientas como Jupyter Notebook, Spyder, VSCode, etc.

### 🛠️ Pasos para Gestionar Entornos Virtuales

### 1. **Abrir Anaconda Navigator**

* En Windows: busca *Anaconda Navigator* en el menú de inicio.
* En macOS/Linux: ejecuta `anaconda-navigator` en la terminal.

### 2. **Ir a la pestaña “Environments” (Entornos)**

Aquí verás una lista de los entornos existentes, incluido el entorno base.

### 3. **Crear un nuevo entorno**

* Haz clic en el botón **“Create”**.
* Asigna un **nombre**.
* Selecciona la **versión de Python** (ej. 3.10).
* Haz clic en **Create**.

> 💡 También puedes incluir R si trabajas con análisis estadístico.

### 4. **Activar un entorno**

* Haz clic en el entorno en la lista.
* Luego selecciona el botón **“Open With”** y elige, por ejemplo, **Jupyter Notebook** o **Terminal**.

### 5. **Instalar paquetes en el entorno**

* Con el entorno seleccionado, ve al menú desplegable y selecciona:

  * **Installed**, **Not Installed** o **All**.
* Escribe el nombre del paquete en el buscador (ej. `pandas`).
* Marca la casilla y haz clic en **Apply**.

### 6. **Clonar un entorno**

* Selecciona el entorno → botón **“Clone”**.
* Asigna un nuevo nombre y clónalo.

### 7. **Eliminar un entorno**

* Selecciona el entorno → botón **“Remove”**.

---

### ✅ Ventajas de usar Anaconda Navigator

| Ventaja                    | Descripción                                                              |
| -------------------------- | ------------------------------------------------------------------------ |
| Interfaz gráfica           | No necesitas saber comandos.                                             |
| Integración directa        | Puedes lanzar Jupyter, Spyder, VSCode, etc., directamente.               |
| Entornos aislados          | Cada entorno tiene sus propias librerías, sin afectar al sistema global. |
| Gestión visual de paquetes | Instala, actualiza o elimina con solo unos clics.                        |

### Resumen

#### ¿Qué es Anaconda Navigator?

Anaconda Navigator es una interfaz gráfica que simplifica la gestión de entornos virtuales y paquetes sin la necesidad de utilizar comandos de terminal. Ideal para usuarios que prefieren una experiencia menos técnica, Navigator ofrece facilidad y accesibilidad, lo que lo convierte en una herramienta invaluable para los profesionales que manejan múltiples proyectos. Desde su interfaz, puedes crear y eliminar entornos virtuales, instalar y gestionar paquetes, e iniciar herramientas como Jupyter Notebook y Spyder directamente.

#### ¿Cómo iniciar Anaconda Navigator?

Para empezar, abre la terminal y escribe **anaconda-navigator**. Una vez que ejecutes este comando, la interfaz de Navigator se abrirá después de unos segundos. Al cargar, podrás ver diversas herramientas disponibles, como Jupyter Notebooks, PyCharm y Visual Studio Code, a las que puedes acceder desde Navigator sin conectarte a una cuenta.

#### ¿Cómo gestionar entornos virtuales?

Dentro de Anaconda Navigator, tienes la posibilidad de gestionar de manera sencilla tus entornos virtuales:

- **Crear un nuevo entorno**: Dirígete a la opción para crear entornos, introduce un nombre (por ejemplo, "ejemplo dos"), selecciona el lenguaje (como Python o R) y la versión deseada. Al crear el entorno, se generará automáticamente y podrá ser accedido para ver los paquetes instalados por defecto.

- **Instalar paquetes**: Desde el entorno, navega a los paquetes no instalados, busca el paquete deseado (por ejemplo, Pandas), selecciona la opción "aplicar" y espera a que se complete la instalación.

- **Exportar e importar entornos**: Puedes exportar el entorno a un archivo `.yml` utilizando la opción "Backup". Para importar, selecciona el archivo guardado y Navigator instalará automáticamente los paquetes en el nuevo entorno.

#### ¿Cómo actualizar y eliminar entornos?

- **Actualizar versiones de paquetes**: Si necesitas actualizar un paquete, dirígete a la sección correspondiente, chequea la versión actual (por ejemplo, Python 3.10.15) y elige una anterior o más reciente. Aplica los cambios y espera a que la nueva versión se instale.

- **Eliminar entornos**: Simplemente presiona el botón "remove" en el entorno que deseas eliminar. Confirma la acción y el entorno será eliminado del sistema.

#### Consejos para usar Anaconda Navigator

Aunque Anaconda Navigator es una herramienta poderosa para manejar entornos y paquetes visualmente, es recomendable que te familiarices también con el uso de la terminal. La combinación de ambas herramientas te permitirá desarrollar habilidades valiosas en el ámbito profesional, donde la rapidez y organización son esenciales. Además, Navigator facilita la integración con herramientas relevantes como Jupyter Notebooks y Spyder, lo que lo hace ideal para proyectos colaborativos.

Empieza a explorar las funcionalidades de Anaconda Navigator para optimizar tu flujo de trabajo y enriquecer tus conocimientos de gestión de entornos en Python. ¡Sigue aprendiendo y descubriendo nuevas formas de eficientizar tu trabajo con esta versátil herramienta!

**Lecturas recomendadas**

[Overview — Anaconda documentation](https://docs.anaconda.com/navigator/overview/)

## Uso de Jupyter Notebooks para Ciencia de Datos con Anaconda

El **uso de Jupyter Notebooks con Anaconda** es una de las formas más populares y eficientes de trabajar en proyectos de **Ciencia de Datos**, gracias a su entorno interactivo y su integración con Python y bibliotecas clave como Pandas, NumPy, Matplotlib y Scikit-learn.

### ✅ ¿Qué es Jupyter Notebook?

Jupyter Notebook es una aplicación web que permite:

* Escribir y ejecutar código Python en celdas.
* Documentar con texto, fórmulas (Markdown + LaTeX), imágenes, etc.
* Visualizar gráficos y resultados en tiempo real.

### 🧰 ¿Cómo usar Jupyter Notebook con Anaconda?

1. **Instala Anaconda**
   Si no lo tienes: [https://www.anaconda.com/download](https://www.anaconda.com/download)

2. **Abre Anaconda Navigator**

   * Desde el menú inicio (Windows).
   * O ejecuta en terminal: `anaconda-navigator`.

3. **Inicia Jupyter Notebook**

   * En Navigator, haz clic en “**Launch**” en la opción de Jupyter Notebook.
   * Esto abrirá una ventana en tu navegador.

4. **Crea un nuevo notebook**

   * Haz clic en **New → Python 3** (o la versión que uses).

5. **Empieza a trabajar**

   * Puedes escribir código en celdas, ejecutarlo con `Shift + Enter` y ver los resultados debajo.
   * También puedes crear celdas de texto para notas o explicaciones.

### 📦 Ejemplo rápido

```python
import pandas as pd

# Cargar datos
df = pd.read_csv("archivo.csv")

# Ver los primeros registros
df.head()
```

Y luego puedes agregar celdas de texto como esta para documentar tu proceso.

### 💡 Ventajas de Jupyter para Ciencia de Datos

* Ideal para exploración y análisis de datos.
* Visualización integrada.
* Fácil de compartir (exportar como HTML o PDF).
* Compatible con otros lenguajes (R, Julia, etc. mediante kernels).

### Resumen

#### ¿Qué es Jupyter Notebooks y por qué es relevante para la ciencia de datos?

Jupyter Notebooks ha transformado la manera en que interactuamos con el código al permitir un entorno interactivo donde puedes combinar programación, visualización de datos y texto descriptivo en un solo documento. Originado de la combinación de los lenguajes Julia, Python y R, Jupyter ha ganado relevancia en la ciencia de datos. Sus beneficios incluyen:

- **Documentación y ejecución combinadas**: Crea reportes claros y reproducibles en el mismo archivo.
- **Visualizaciones en tiempo real**: Ejecuta el código y visualiza los resultados inmediatamente.
- **Entornos interactivos**: Experimenta con pequeños bloques de código en un ciclo interactivo conocido como REPL (Read, Eval, Print, Loop).

#### ¿Cómo iniciar y utilizar Jupyter Notebooks desde Anaconda?

Iniciar Jupyter Notebooks desde Anaconda es un proceso sencillo que se realiza desde la terminal. Sigue estos pasos para comenzar:

1. **Inicia el servidor de Jupyter**: En la terminal, ejecuta el comando` jupyter notebook`. Con esto, se abrirá la página de Jupyter en tu navegador web.
2. **Crea un nuevo notebook**: Haz clic en "New" y selecciona "Python 3" para iniciar un nuevo documento.
3. **Interacción con celdas**: En Jupyter, puedes crear celdas de código o de texto en Markdown. Puedes ejecutar fácilmente el código en las celdas y reorganizarlas según sea necesario.

```python
# Ejemplo de código en una celda de Jupyter Notebook
print("Hola, Mundo")
```

#### ¿Cómo exportar y compartir notebooks en diferentes formatos?

Una de las funcionalidades clave de Jupyter es la capacidad para guardar y compartir notebooks en diversos formatos:

- **Renombrar el notebook**: Antes de exportar, asegúrate de cambiar el nombre del archivo si es necesario. Esto se hace en la parte superior del documento.
- **Exportar el notebook**: Ve a la pestaña "File" y selecciona el formato deseado para descargar, que puede ser el formato predeterminado .ipynb, PDF, o HTML, entre otros.

#### ¿Cómo manejar archivos externos y datos en Jupyter Notebooks?

Trabajar con datos es esencial en Jupyter Notebooks. Puedes cargar y manipular archivos de datos como CSVs usando bibliotecas populares como Pandas:

```python
import pandas as pd

# Cargando un archivo CSV
data = pd.read_csv('datos.csv')
print(data.head())
```

Para cargar archivos, puedes usar la opción "Upload" en Jupyter y asegurarte de que el archivo esté en el directorio raíz o donde está el notebook.

#### ¿Cómo crear y trabajar en entornos virtuales con Anaconda?

La gestión de entornos virtuales en Anaconda permite mantener proyectos organizados y evitar conflictos de dependencias. Aquí te explicamos cómo hacerlo:

1. **Crea un nuevo entorno**: Usa el comando conda create -n nombre_del_entorno para crear un entorno nuevo.
2. **Activa el entorno**: Para cambiar al nuevo entorno, ejecuta conda activate nombre_del_entorno.
3. **Instala Jupyter en el entorno**: Si necesitas usar Jupyter en un nuevo entorno, ejecútalo usando conda install jupyter.
4. **Instala librerías necesarias**: Para cualquier módulo adicional, como NumPy o Matplotlib, instala las librerías necesarias usando conda install nombre_del_paquete.

#### ¿Qué hacer si una librería no está instalada en el entorno?

Si encuentras un error al ejecutar un código que requiere de una librería no instalada, como NumPy o Matplotlib, dentro de un entorno virtual, será necesario instalarla:

`conda install numpy matplotlib`

Y así, puedes volver a intentar ejecutar tu código. Lograrás una configuración óptima para trabajar con distintos proyectos de ciencia de datos en Jupyter Notebooks.

**Lecturas recomendadas**

[GitHub - platzi/anaconda-jupyter at JupyterNotebooks](https://github.com/platzi/venvs-anaconda-jupyter/tree/JupyterNotebooks)

## Comandos mágicos en Jupyter Notebook: Atajos y funcionalidades clave

Los **comandos mágicos** en **Jupyter Notebook** son instrucciones especiales que comienzan con `%` (para comandos de línea) o `%%` (para comandos de celda) y ofrecen funcionalidades avanzadas para hacer tu trabajo más eficiente.

### 🪄 Tipos de Comandos Mágicos

#### 1. **Comandos de línea (`%`)**

Afectan solo a la línea en la que se escriben.

| Comando              | Descripción                                   |
| -------------------- | --------------------------------------------- |
| `%time`              | Mide el tiempo que tarda una línea de código. |
| `%who`               | Muestra las variables definidas.              |
| `%lsmagic`           | Lista todos los comandos mágicos disponibles. |
| `%pwd`               | Muestra el directorio actual.                 |
| `%cd`                | Cambia de directorio.                         |
| `%run archivo.py`    | Ejecuta un archivo `.py` en el notebook.      |
| `%matplotlib inline` | Muestra gráficos directamente en el notebook. |

#### Ejemplo:

```python
%time sum(range(1000000))
```

#### 2. **Comandos de celda (`%%`)**

Afectan toda la celda.

| Comando                 | Descripción                                                    |
| ----------------------- | -------------------------------------------------------------- |
| `%%time`                | Mide el tiempo de ejecución de toda la celda.                  |
| `%%writefile nombre.py` | Guarda el contenido de la celda en un archivo.                 |
| `%%capture`             | Captura la salida de la celda (útil para evitar mostrar logs). |
| `%%bash`                | Ejecuta comandos de Bash directamente desde la celda.          |
| `%%html`                | Permite escribir HTML dentro de la celda.                      |

#### Ejemplo:

```python
%%bash
echo "Hola desde bash"
```

### ⌨️ Atajos de teclado útiles

* `Shift + Enter`: Ejecuta la celda y pasa a la siguiente.
* `Ctrl + Enter`: Ejecuta la celda y permanece en ella.
* `A`: Insertar celda **arriba** (en modo comando).
* `B`: Insertar celda **abajo**.
* `D D`: Elimina la celda actual (presiona dos veces `D`).
* `M`: Convertir celda a **Markdown**.
* `Y`: Convertir celda a **código**.
* `Esc`: Salir del modo edición a comando.
* `H`: Mostrar todos los atajos disponibles.

### 💡 Consejo Pro

Para ver toda la documentación sobre los comandos mágicos, ejecuta:

```python
%magic
```

### Resumen

#### ¿Qué son los comandos mágicos en Jupyter Notebook?

Jupyter Notebook es una herramienta imprescindible para científicos de datos y programadores que buscan realizar análisis interactivos de manera eficiente. Los comandos mágicos son una funcionalidad poderosa que optimiza y acelera las tareas cotidianas dentro del entorno. Estos atajos no solo permiten manipular el entorno de trabajo, sino también ejecutar comandos del sistema operativo, medir tiempos de ejecución y más sin necesidad de abandonar el notebook.

Existen dos tipos principales de comandos mágicos:

- **Comandos de línea mágica**: Se identifican con un solo signo de porcentaje (%) y afectan solo la línea de código donde se utilizan.
- **Comandos de celda mágica**: Utilizan un doble signo de porcentaje (%%) y aplican su efecto a toda la celda de código.

#### ¿Cómo organizar el entorno de Jupyter Notebook?

Antes de sumergirse en el uso de comandos mágicos, es crucial mantener un entorno de trabajo ordenado. Comienza creando una nueva carpeta para concentrar todos tus notebooks. Nombrar de manera significativa y estructurada tus archivos y carpetas facilita la navegación y gestión de tu proyecto.

`%mkdir notebooks`

Una vez creada la carpeta, mueve tus archivos actuales a este directorio para tener todo a mano y ordenado.

#### ¿Cómo listar archivos y directorio actual?

Para verificar el contenido de tu directorio actual, utiliza el comando mágico `%ls`. Es similar al comando 'ls' en la terminal de Unix y te mostrará todos los archivos en tu directorio.

`%ls`

Para conocer en qué directorio te encuentras trabajando, el comando `%pwd` te proporcionará el directorio de trabajo actual.

`%pwd`

#### ¿Cómo medir tiempos de ejecución?

Medir el tiempo que tarda en ejecutarse una línea de código puede ser crucial para optimizar procesos. El comando `%time` es perfecto para esto. Por ejemplo, calcula el tiempo que tarda en ejecutarse una suma en una lista.

`%time sum([x for x in range(10000)])`

Si deseas medir el tiempo de ejecución de toda una celda, puedes usar %%time.

```python
%%time
result = []
for i in range(10000):
    result.append(i ** 2)
```

#### ¿Cómo trabajar con variables y archivos?

Puedes obtener un panorama general de las variables en tu entorno utilizando `%whos`. Te mostrará detalles como nombre de variable y tipo de dato.

`%whos`

Para almacenar código de una celda en un archivo, utiliza el siguiente método:

```python
%%writefile file.py
print("Este código fue guardado en un archivo")
```

Luego, para correr el archivo creado, utiliza `%run`.

`%run file.py`

#### ¿Cómo visualizar gráficos directamente en Jupyter?

Si trabajas con la librería Matplotlib y deseas que los gráficos se generen en línea, el comando `%matplotlib inline` es esencial. Esto evita la creación de ventanas separadas para los gráficos.

`%matplotlib inline`

#### ¿Qué es TimeIt y cómo maximiza el análisis de tiempo?

TimeIt es la versión avanzada de `time` y sirve para correr el mismo bloque de código múltiples veces, proporcionando el promedio del tiempo de ejecución.

```python
%%timeit
result = []
for i in range(10000):
    result.append(i ** 2)
```

#### ¿Cómo reiniciar variables en el entorno?

Si necesitas liberar memoria o reiniciar variables, el comando `%reset` te preguntará qué variables deseas eliminar. Esto es especialmente útil para trabajos intensivos en memoria.

`%reset`

#### ¿Cómo integrar librerías como Pandas eficientemente?

Al utilizar bibliotecas como Pandas, puedes combinar comandos mágicos con operaciones de lectura para hacer tu trabajo más ágil.

```python
import pandas as pd
%time data = pd.read_csv('datos.csv')
```

Con cada uso de Jupyter Notebooks, los comandos mágicos se convertirán en herramientas clave que facilitarán tus sesiones de análisis y programación. ¡Sigue explorando y practicando para dominar estas magias!

**Lecturas recomendadas**

[GitHub - platzi/anaconda-jupyter at ComandosMagicos](https://github.com/platzi/venvs-anaconda-jupyter/tree/ComandosMagicos)