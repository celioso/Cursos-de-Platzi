# Curso de Expresiones Regulares

## Todo lo que aprenderás sobre expresiones regulares

Este curso te va a enseñar qué son las expresiones regulares y cómo utilizarlas.
Por ejemplo aplicaciones de búsqueda y filtrado, las expresiones regulares son extremadamente potentes, aprende a utilizarlas en este curso.

## Expresiones Regulares en Programación: Una Guía Básica

**¿Qué son las expresiones regulares?**

Las expresiones regulares, a menudo abreviadas como **regex** o **regexp**, son patrones que se utilizan para buscar y manipular texto. Son una herramienta poderosa en la programación que permite encontrar coincidencias específicas dentro de cadenas de caracteres. Imagina que tienes un gran documento y quieres encontrar todas las direcciones de correo electrónico. Una expresión regular te permitiría hacer esto de forma rápida y eficiente.

**¿Para qué se utilizan?**

* **Validación de datos:** Verificar si un campo de un formulario cumple con un formato específico (por ejemplo, direcciones de correo electrónico, números de teléfono, contraseñas).
* **Extracción de información:** Obtener datos específicos de un texto, como números, fechas o nombres.
* **Reemplazo de texto:** Buscar y reemplazar patrones de texto dentro de una cadena.
* **Análisis de texto:** Realizar tareas de procesamiento de lenguaje natural, como tokenización, lematización y stemming.

**¿Cómo funcionan?**

Una expresión regular es una secuencia de caracteres que define un patrón de búsqueda. Estos caracteres pueden ser literales (los caracteres exactos que quieres buscar) o metacaracteres (caracteres especiales que tienen un significado especial dentro de una expresión regular).

**Ejemplos de metacaracteres comunes:**

* **`.` (punto):** Coincide con cualquier carácter excepto una nueva línea.
* **`^`:** Coincide con el inicio de una cadena.
* **`$`:** Coincide con el final de una cadena.
* **`*`:** Coincide con cero o más repeticiones del carácter anterior.
* **`+`:** Coincide con una o más repeticiones del carácter anterior.
* **`?`:** Coincide con cero o una ocurrencia del carácter anterior.
* **`[]`:** Define un conjunto de caracteres.
* **`()`:** Agrupa sub-expresiones.

**Ejemplo de expresión regular:**

La expresión regular `\d{3}-\d{3}-\d{4}` buscaría un patrón de números de teléfono en el formato XXX-XXX-XXXX.

**¿En qué lenguajes se utilizan?**

Casi todos los lenguajes de programación modernos soportan expresiones regulares, incluyendo:

* **JavaScript:** Se utilizan con el objeto `RegExp`.
* **Python:** El módulo `re` proporciona funciones para trabajar con expresiones regulares.
* **Java:** La clase `java.util.regex.Pattern` se utiliza para compilar expresiones regulares.
* **Perl:** Es conocido por su potente soporte para expresiones regulares.
* **PHP:** Se utilizan con las funciones `preg_match`, `preg_replace`, etc.

**¿Dónde aprender más?**

Existen numerosos recursos en línea para aprender más sobre expresiones regulares. Algunos de los más populares incluyen:

* **MDN Web Docs (JavaScript):** [https://developer.mozilla.org/es/docs/Web/JavaScript/Guide/Regular_expressions](https://developer.mozilla.org/es/docs/Web/JavaScript/Guide/Regular_expressions)
* **Regex101:** Una herramienta interactiva para probar expresiones regulares: [https://regex101.com/](https://regex101.com/)
* **Tutoriales en línea:** Busca tutoriales en plataformas como YouTube o cursos en línea especializados en programación.

**En resumen,** las expresiones regulares son una herramienta fundamental para cualquier programador que trabaje con texto. Conocer los conceptos básicos y practicar con ejemplos te permitirá dominar esta poderosa técnica y resolver una amplia gama de problemas de programación.

## ¿Qué son las expresiones regulares?

[Tutorial de algoritmos de programación | Cursos Platzi - YouTube](https://www.youtube.com/watch?v=SDv2vOIFIj8)

### Expresiones Regulares: Un Patrón para Buscar Texto

**Imagina que tienes un enorme libro y quieres encontrar todas las palabras que empiezan con "super".** Podrías pasar horas buscando manualmente, pero existe una forma más rápida y eficiente: ¡las expresiones regulares!

**¿Qué son las expresiones regulares?**

Son secuencias de caracteres que forman un patrón de búsqueda. Se utilizan principalmente para:

* **Buscar patrones:** Encontrar texto que coincida con un patrón específico dentro de una cadena más grande.
* **Reemplazar texto:** Sustituir partes de una cadena que coincidan con un patrón.
* **Validar datos:** Verificar si una cadena cumple con un formato determinado (por ejemplo, direcciones de correo electrónico, números de teléfono).

**¿Cómo funcionan?**

Las expresiones regulares utilizan caracteres especiales llamados **metacaracteres** para definir los patrones de búsqueda. Por ejemplo:

* **`.` (punto):** Coincide con cualquier carácter (excepto un salto de línea).
* **`^`:** Coincide con el inicio de una cadena.
* **`$`:** Coincide con el final de una cadena.
* **`*`:** Coincide con cero o más repeticiones del carácter anterior.
* **`+`:** Coincide con una o más repeticiones del carácter anterior.
* **`?`:** Coincide con cero o una ocurrencia del carácter anterior.
* **`[]`:** Define un conjunto de caracteres.

**Ejemplo:**

Para encontrar todas las palabras que empiezan con "super", podrías usar la expresión regular: `^super`

**¿Dónde se utilizan?**

Las expresiones regulares son muy útiles en muchos lenguajes de programación, como:

* **JavaScript:** Se utilizan con el objeto `RegExp`.
* **Python:** El módulo `re` proporciona funciones para trabajar con expresiones regulares.
* **Java:** La clase `java.util.regex.Pattern` se utiliza para compilar expresiones regulares.
* **Perl:** Es conocido por su potente soporte para expresiones regulares.
* **PHP:** Se utilizan con las funciones `preg_match`, `preg_replace`, etc.

**¿Por qué son útiles?**

* **Automatización:** Permiten automatizar tareas repetitivas de búsqueda y reemplazo de texto.
* **Flexibilidad:** Se pueden crear patrones muy complejos para encontrar casi cualquier tipo de texto.
* **Eficiencia:** Son mucho más eficientes que las búsquedas manuales.

**¿Quieres aprender más?**

Existen muchos recursos en línea para profundizar en el tema. Te recomiendo:

* **MDN Web Docs (JavaScript):** [https://developer.mozilla.org/es/docs/Web/JavaScript/Guide/Regular_expressions](https://developer.mozilla.org/es/docs/Web/JavaScript/Guide/Regular_expressions)
* **Regex101:** Una herramienta interactiva para probar expresiones regulares: [https://regex101.com/](https://regex101.com/)

**¿Tienes alguna pregunta específica sobre las expresiones regulares?** Por ejemplo, puedo explicarte cómo utilizarlas en un lenguaje de programación concreto, o resolver un problema específico que tengas.

**Ejemplo práctico:**

Supongamos que quieres validar una dirección de correo electrónico. Una expresión regular simple para esto podría ser: `^[^\s@]+@[^\s@]+\.[^\s@]+$`

Esta expresión verifica que haya al menos un carácter antes del símbolo `@`, luego el símbolo `@`, seguido de al menos un carácter y un punto, y finalmente al menos un carácter más.

## Aplicaciones de las expresiones regulares

Buscar e investigar sobre Expresiones Regulares puede ser muy intimidante.
/^(.){5}\w?[a-Z|A-Z|0-9]$/ig
En serio pueden parecer muy, muy raras; pero la verdad es que no lo son.

En esta clase aprenderás, para qué te puede servir el saber usar bien las Expresiones Regulares; y es, en pocas palabras, para buscar.

Las expresiones regulares (regex) tienen numerosas aplicaciones en diferentes áreas de la informática, principalmente en la manipulación y análisis de texto. A continuación se presentan algunas de las aplicaciones más comunes:

### 1. **Búsqueda y reemplazo de texto**
   - **Aplicación**: Se usan para buscar patrones específicos en documentos o archivos y reemplazarlos.
   - **Ejemplo**: En un archivo de texto, se puede reemplazar todas las direcciones de correo electrónico con un texto específico.
   - **Herramientas**: `sed`, `vim`, `grep`, editores de texto como Notepad++ o VS Code.

   ```bash
   sed 's/[0-9]\{10\}/[CENSURADO]/g' archivo.txt
   ```

### 2. **Validación de entradas**
   - **Aplicación**: Verificar si la entrada del usuario sigue un formato específico, como correos electrónicos, números de teléfono, códigos postales, etc.
   - **Ejemplo**: Validar si una dirección de correo electrónico ingresada por el usuario tiene el formato correcto.
   - **Herramientas**: Usado en formularios web con JavaScript, PHP, Python, y Bash.

   ```bash
   if [[ "$email" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
     echo "Email válido"
   else
     echo "Email no válido"
   fi
   ```

### 3. **Extracción de datos**
   - **Aplicación**: Extraer datos relevantes de grandes volúmenes de texto basándose en patrones. Por ejemplo, extraer fechas, números de teléfono o URLs de un archivo.
   - **Ejemplo**: Extraer todas las URLs de un archivo HTML.
   - **Herramientas**: `grep`, `awk`, `Python`.

   ```bash
   grep -oP 'https?://\S+' archivo.html
   ```

### 4. **Análisis y limpieza de datos**
   - **Aplicación**: Limpiar datos en bruto (por ejemplo, eliminar caracteres no deseados, espacios en blanco) o transformar datos a formatos más manejables.
   - **Ejemplo**: Eliminar caracteres especiales de una lista de nombres en un archivo CSV.
   - **Herramientas**: Python (pandas), `sed`, `awk`.

   ```python
   import re
   data_cleaned = re.sub('[^A-Za-z0-9]+', '', data)
   ```

### 5. **Procesamiento de logs**
   - **Aplicación**: Analizar archivos de registro (logs) y extraer información relevante, como fechas, IPs o mensajes de error.
   - **Ejemplo**: Extraer todas las direcciones IP de un archivo de registro del servidor.
   - **Herramientas**: `grep`, `awk`, herramientas de análisis de logs.

   ```bash
   grep -oP '\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b' log.txt
   ```

### 6. **Filtrado de datos**
   - **Aplicación**: Seleccionar o descartar líneas que coinciden con ciertos patrones de texto.
   - **Ejemplo**: Filtrar un archivo CSV para encontrar solo las líneas que contienen una palabra clave.
   - **Herramientas**: `grep`, `awk`, `find`.

   ```bash
   grep 'ERROR' system.log
   ```

### 7. **Automatización de tareas repetitivas**
   - **Aplicación**: Automatizar tareas como renombrar archivos en masa o reorganizar grandes volúmenes de texto.
   - **Ejemplo**: Renombrar todos los archivos con una extensión específica.
   - **Herramientas**: Bash scripts, `rename`.

   ```bash
   rename 's/\.jpeg$/\.jpg/' *.jpeg
   ```

### 8. **Desarrollo web y SEO**
   - **Aplicación**: Identificar y corregir problemas en el código HTML, como etiquetas no cerradas o enlaces rotos.
   - **Ejemplo**: Encontrar todas las imágenes sin etiquetas `alt` en un archivo HTML.
   - **Herramientas**: Python, editores de texto, `grep`.

   ```bash
   grep -oP '<img [^>]*?(?<!alt)>' archivo.html
   ```

### 9. **Manipulación de archivos de configuración**
   - **Aplicación**: Modificar archivos de configuración (como archivos de configuración de servidores) en sistemas operativos basados en Unix.
   - **Ejemplo**: Cambiar el valor de una clave específica en un archivo `.conf`.
   - **Herramientas**: `sed`, `awk`.

   ```bash
   sed -i 's/^Listen 80$/Listen 8080/' /etc/httpd/conf/httpd.conf
   ```

### 10. **Testing y depuración**
   - **Aplicación**: Utilizar expresiones regulares en scripts de prueba automatizados para validar salidas específicas.
   - **Ejemplo**: Validar que una salida de texto cumple con un formato determinado en una prueba automatizada.
   - **Herramientas**: Python (unittest), Bash, `grep`.

### Resumen:
Las expresiones regulares son una herramienta esencial para tareas de manipulación de texto, análisis de datos y automatización, brindando flexibilidad y potencia para manejar patrones complejos.

## Introducción al lenguaje de expresiones regulares

Con las expresiones regulares vamos a solucionar problemas reales, problemas del día a día.

¿Qué pasa si queremos buscar en un texto (txt, csv, log, cualquiera), todos los números de teléfonos que hay?
Tendríamos que considerar por ejemplo, que un teléfono de México serían 10 dígitos; hay quienes los separan con guión, hay quienes los separan con puntos, hay quienes no los separan sino que tienen los 10 dígitos exactos, y este patrón puede cambiar para otros países.

Esto mismo sucede con números de tarjetas de crédito, códigos postales, dirección de correos, formatos de fechas o montos, etc.

[Online regex tester and debugger: PHP, PCRE, Python, Golang and JavaScript](https://regex101.com/)

## El caracter (.)

[Para trabajar caracteres](https://regexr.com/)

¿Qué es un archivo de texto, por ejemplo un CSV?
¿Qué es una cadena de caracteres?

Cada espacio en una cadena de texto se llena con un caracter, esto lo necesitamos tener perfectamente claro para comenzar a trabajar con expresiones regulares

Abriremos nuestro editor, qué en este curso recomendamos ATOM, vamos a presionar CTRL + F y podremos buscar por match idénticos.

[El código ASCII Completo, tabla con los codigos ASCII completos, caracteres simbolos letras ascii, ascii codigo, tabla ascii, codigos ascii, caracteres ascii, codigos, tabla, caracteres, simbolos, control, imprimibles, extendido, letras, vocales, signos,](http://www.elcodigoascii.com.ar/)

El carácter **`.`** (punto) en expresiones regulares es un **comodín** que representa **cualquier carácter**, excepto una nueva línea (`\n`).

### Ejemplos de uso:
1. **Coincide con cualquier carácter simple:**
   - Expresión regular: `a.b`
   - Coincide con:
     - `aab`
     - `acb`
     - `a5b`
   - No coincide con:
     - `ab` (falta un carácter entre `a` y `b`)
     - `a\nb` (porque no coincide con una nueva línea)

2. **Combinar el `.` con otros patrones:**
   - Expresión regular: `..`
   - Coincide con cualquier secuencia de **dos caracteres**.
   - Por ejemplo, coincide con `aa`, `b2`, `@#`, etc.

### Uso avanzado:
Para hacer que el punto **literal** (es decir, que represente un punto en lugar de "cualquier carácter"), debe escaparse con una barra invertida (`\.`). 

- **Ejemplo**: `a\.b` coincide con `a.b` pero **no** con `acb` o `a5b`.

El `.` es muy útil en expresiones regulares cuando se quiere flexibilizar una búsqueda, permitiendo que cualquier carácter esté en una posición específica.

## Las clases predefinidas y construidas

Las búsquedas en las expresiones regulares funcionan en múltiplos de la cantidad de caracteres que explícitamente indicamos.

En las expresiones regulares (regex), las **clases de caracteres** permiten definir conjuntos de caracteres específicos que deben coincidir con una posición en el texto. Estas clases pueden ser **predefinidas** o **personalizadas** (construidas manualmente). Vamos a explorar ambas:

### 1. **Clases predefinidas**
Son clases ya construidas que permiten hacer coincidir ciertos tipos de caracteres comunes. Algunas de las más utilizadas son:

| Clase         | Descripción                                                         | Ejemplo          |
|---------------|---------------------------------------------------------------------|------------------|
| `.`           | Cualquier carácter (excepto nueva línea `\n`)                       | `a.b` coincide con `acb`, `a1b`, pero no con `ab`. |
| `\d`          | Un dígito (equivalente a `[0-9]`)                                   | `\d\d` coincide con `12`, `45`. |
| `\D`          | Cualquier carácter que **no** sea un dígito (equivalente a `[^0-9]`) | `\D\D` coincide con `ab`, `AZ`, pero no con `12`. |
| `\w`          | Un carácter de palabra (letra, dígito o guion bajo: `[a-zA-Z0-9_]`) | `\w\w` coincide con `ab`, `a1`, pero no con `@!`. |
| `\W`          | Cualquier carácter que **no** sea una palabra (equivalente a `[^a-zA-Z0-9_]`) | `\W\W` coincide con `!@`, `##`, pero no con `ab`. |
| `\s`          | Un espacio en blanco (incluye espacio, tabulador, nueva línea)      | `\s` coincide con un espacio o tab. |
| `\S`          | Cualquier carácter que **no** sea un espacio en blanco              | `\S\S` coincide con `ab`, `a1`, pero no con ` ` (espacio). |

### 2. **Clases construidas (personalizadas)**
Estas se definen con corchetes `[]` y contienen un conjunto de caracteres que deben coincidir. Puedes especificar rangos de caracteres o combinaciones de caracteres específicos.

#### Ejemplos:
- **`[aeiou]`**: Coincide con cualquier vocal (minúsculas).
  - Ejemplo: En `hola`, coincide con `o` y `a`.
  
- **`[0-9]`**: Coincide con cualquier dígito.
  - Ejemplo: En `a5b`, coincide con `5`.
  
- **`[A-Z]`**: Coincide con cualquier letra mayúscula.
  - Ejemplo: En `Casa`, coincide con `C`.

- **`[^a-z]`**: Coincide con cualquier carácter que **no** sea una letra minúscula (el acento circunflejo `^` indica negación).
  - Ejemplo: En `Hola!`, coincide con `H`, `!`.

#### Combinación de clases:
Puedes combinar clases predefinidas y construidas para mayor flexibilidad:
- **`[a-zA-Z0-9]`**: Coincide con cualquier letra o número (minúsculas y mayúsculas).
- **`[\d\s]`**: Coincide con cualquier dígito o espacio en blanco.

### Resumen:
- **Clases predefinidas**: Simplifican el uso de patrones comunes como dígitos, letras, espacios, etc. (`\d`, `\w`, `\s`).
- **Clases construidas**: Te permiten definir conjuntos de caracteres específicos utilizando rangos o caracteres personalizados (`[a-z]`, `[A-Z0-9_]`).

Estas clases son fundamentales para escribir expresiones regulares potentes y flexibles, facilitando la búsqueda, validación y manipulación de texto.

Para buscar números **hexadecimales** utilizando **expresiones regulares**, se debe crear un patrón que reconozca los caracteres válidos en el sistema hexadecimal. Los números hexadecimales están compuestos por los dígitos del `0` al `9` y las letras de la `A` a la `F` (o sus equivalentes en minúsculas `a-f`).

### Patrón de expresión regular para números hexadecimales:
Un número hexadecimal típico comienza con `0x` o `0X` seguido de una secuencia de dígitos y letras (`0-9`, `A-F`, `a-f`).

#### Expresión regular para detectar hexadecimales:
```regex
0[xX][0-9a-fA-F]+
```

### Desglose del patrón:
- **`0[xX]`**: Coincide con `0x` o `0X`, que es el prefijo común para los números hexadecimales.
- **`[0-9a-fA-F]`**: Coincide con cualquier dígito hexadecimal (0-9) o letra (A-F o a-f).
- **`+`**: Indica que el patrón debe coincidir con **uno o más** caracteres hexadecimales.

### Ejemplos de coincidencias:
- `0x1A3F`
- `0X4b2c`
- `0xabcdef`

### Ejemplo de uso en un contexto de código bash:
Si estás buscando números hexadecimales en una cadena, puedes usar esta expresión regular en herramientas de la línea de comandos como `grep`:

```bash
echo "Valores: 0x1A3F, 0x2B4C, no es hexadecimal: 12345" | grep -oE '0[xX][0-9a-fA-F]+'
```

Esto imprimirá los números hexadecimales encontrados:
```
0x1A3F
0x2B4C
``` 

Este enfoque permite filtrar y extraer números hexadecimales de texto o archivos fácilmente.

## Los delimitadores: +, *, ?

En **expresiones regulares**, los delimitadores `+`, `*`, y `?` son **metacaracteres** que se utilizan para definir la cantidad de veces que un patrón o un carácter debe aparecer. Estos son conocidos como **cuantificadores**.

### 1. **El delimitador `*` (cero o más)**
El asterisco `*` significa que el elemento anterior puede aparecer **cero o más veces**. En otras palabras, es opcional y puede repetirse indefinidamente.

#### Ejemplo:
- Expresión regular: `a*`
  - Coincide con:
    - Cadena vacía `""`
    - `a`, `aa`, `aaa`, etc.
    - Cualquier cadena con cero o más `a`s.
  
- Expresión regular: `ba*`
  - Coincide con:
    - `b` (porque `a*` puede ser cero)
    - `ba`, `baa`, `baaa`, etc.

### 2. **El delimitador `+` (uno o más)**
El signo más `+` significa que el elemento anterior debe aparecer **al menos una vez** (es decir, **uno o más**).

#### Ejemplo:
- Expresión regular: `a+`
  - Coincide con:
    - `a`
    - `aa`, `aaa`, etc.
    - No coincide con la cadena vacía `""` porque se requiere al menos una `a`.

- Expresión regular: `ba+`
  - Coincide con:
    - `ba`, `baa`, `baaa`, etc.
    - No coincide con `b` (porque se requiere al menos una `a`).

### 3. **El delimitador `?` (cero o uno)**
El signo de interrogación `?` significa que el elemento anterior es **opcional** y puede aparecer **cero o una vez**.

#### Ejemplo:
- Expresión regular: `a?`
  - Coincide con:
    - Cadena vacía `""`
    - `a`
    - No coincide con `aa` (porque se permite solo una `a` como máximo).

- Expresión regular: `ba?`
  - Coincide con:
    - `b`
    - `ba`
    - No coincide con `baa` (porque solo se permite una `a` como máximo).

### Resumen:

| Metacaracter | Significado                               | Ejemplo               | Descripción               |
|--------------|-------------------------------------------|-----------------------|---------------------------|
| `*`          | Cero o más repeticiones del elemento      | `a*`                  | Coincide con `""`, `a`, `aa`, `aaa` |
| `+`          | Una o más repeticiones del elemento       | `a+`                  | Coincide con `a`, `aa`, `aaa`, pero no con `""` |
| `?`          | Cero o una repetición del elemento        | `a?`                  | Coincide con `""` o `a`, pero no con `aa` |

### Ejemplos combinados:
- **`[a-z]*`**: Coincide con cualquier cantidad de letras minúsculas (incluyendo cero).
- **`\d+`**: Coincide con uno o más dígitos (ej. `123`, `5`).
- **`colou?r`**: Coincide con `color` o `colour` (la `u` es opcional).

Estos delimitadores son muy útiles para crear patrones flexibles cuando se trabaja con expresiones regulares.

![regulares ](images/regExcheatsheet-1.png)

![regulares ](images/regExcheatsheet-2.png)

## Los contadores {1,4}

En **expresiones regulares**, los **contadores** o **cuantificadores** como `{n,m}` permiten especificar exactamente cuántas veces un carácter o patrón debe aparecer en la coincidencia. Este tipo de cuantificador es más preciso que los delimitadores básicos como `*`, `+` y `?`.

### **Formato: `{n,m}`**
- **`n`**: El número mínimo de repeticiones.
- **`m`**: El número máximo de repeticiones.

### Variantes de uso de los contadores `{n,m}`:
1. **`{n,m}`**: El patrón debe aparecer al menos `n` veces y como máximo `m` veces.
2. **`{n,}`**: El patrón debe aparecer al menos `n` veces, sin límite superior.
3. **`{n}`**: El patrón debe aparecer exactamente `n` veces.

### Ejemplo 1: **`{1,4}`**
Este cuantificador significa que el patrón debe aparecer **al menos una vez y como máximo cuatro veces**.

#### Ejemplo:
- Expresión regular: `a{1,4}`
  - Coincide con:
    - `a`, `aa`, `aaa`, `aaaa`
  - No coincide con:
    - `""` (cadena vacía), `aaaaa` (más de 4 repeticiones).

### Ejemplo 2: **`{2,}`**
Este patrón significa que el elemento debe aparecer **al menos 2 veces**, sin un límite superior.

#### Ejemplo:
- Expresión regular: `b{2,}`
  - Coincide con:
    - `bb`, `bbb`, `bbbb`, `bbbbb`, etc.
  - No coincide con:
    - `b` (menos de 2 repeticiones).

### Ejemplo 3: **`{3}`**
Este patrón requiere que el elemento aparezca **exactamente 3 veces**.

#### Ejemplo:
- Expresión regular: `c{3}`
  - Coincide con:
    - `ccc`
  - No coincide con:
    - `cc`, `cccc` (ni menos ni más de 3 repeticiones).

### Aplicaciones prácticas:
- **Validar números de teléfono**:
  - Expresión regular: `\d{3,4}-\d{7}` podría coincidir con un número de teléfono que tiene un prefijo de 3 o 4 dígitos, seguido de un guion y luego 7 dígitos.
  
- **Buscar palabras con una longitud específica**:
  - Expresión regular: `[a-zA-Z]{4,6}` coincidiría con palabras que tienen entre 4 y 6 letras.

### Ejemplos combinados:
1. **`[a-z]{2,5}`**: Coincide con entre 2 y 5 letras minúsculas consecutivas.
   - Coincide con: `ab`, `abcd`, `abcde`
   - No coincide con: `a` (menos de 2 letras), `abcdef` (más de 5 letras).

2. **`\d{1,3}`**: Coincide con entre 1 y 3 dígitos.
   - Coincide con: `5`, `12`, `987`
   - No coincide con: `1234` (más de 3 dígitos).

### Resumen:
| Cuantificador | Significado                                         | Ejemplo            | Descripción               |
|---------------|-----------------------------------------------------|--------------------|---------------------------|
| `{n,m}`       | Al menos `n` veces y como máximo `m` veces           | `a{1,4}`           | Coincide con `a`, `aa`, `aaa`, `aaaa` |
| `{n,}`        | Al menos `n` veces, sin límite superior              | `b{2,}`            | Coincide con `bb`, `bbb`, `bbbb`... |
| `{n}`         | Exactamente `n` veces                                | `c{3}`             | Coincide con `ccc`, pero no con `cc` o `cccc` |

Los contadores son esenciales para definir patrones con una precisión específica en las expresiones regulares, adaptándose a las necesidades de validación o búsqueda de datos más exactos.

![Contadores {1,4}](images/clase-8.png)

## El caso de (?) como delimitador

El **`?`** en las expresiones regulares es un **cuantificador** que indica que el elemento anterior es **opcional**, es decir, puede aparecer **cero o una vez**.

### Explicación de **`?` como delimitador**
Cuando se coloca el signo de interrogación **`?`** después de un carácter, grupo o patrón, le dice a la expresión regular que ese elemento puede aparecer **una vez o ninguna**.

### Ejemplos del uso de **`?`**:

1. **Expresión regular: `colou?r`**
   - Coincide con:
     - `color`
     - `colour`
   - En este caso, la letra `u` es opcional. Por lo tanto, la expresión coincide con ambas versiones de la palabra: con o sin la `u`.

2. **Expresión regular: `a?b`**
   - Coincide con:
     - `b`
     - `ab`
   - Aquí, la letra `a` es opcional. La expresión coincidirá tanto si `a` está presente como si no lo está.

3. **Expresión regular: `https?://`**
   - Coincide con:
     - `http://`
     - `https://`
   - El `?` indica que la `s` es opcional, por lo que coincide con ambas formas de un enlace web, con o sin el prefijo `https`.

### **¿Cómo funciona el delimitador `?`?**
- **Con caracteres individuales**: 
   - **`a?`**: Coincide con **`a`** o la cadena vacía.
   
- **Con grupos de caracteres (entre paréntesis)**:
   - **`(abc)?`**: Coincide con **`abc`** o la cadena vacía (es decir, todo el grupo es opcional).
   
- **En combinación con otros metacaracteres**: 
   - En combinación con otros cuantificadores como `*` o `+`, el `?` puede alterar el comportamiento de estos, como en el caso del "modo no codicioso" (lazy mode).

### **Uso avanzado: Modo no codicioso (`?`)**
Además de su uso como cuantificador, el **`?`** puede usarse para modificar el comportamiento de otros cuantificadores (`*`, `+`, `{n,m}`) para que sean **no codiciosos**. Esto significa que intentarán coincidir con la menor cantidad de caracteres posible.

#### Ejemplo:
- **Expresión regular codiciosa: `a.*b`**
  - Coincide con la mayor cantidad de caracteres posible entre `a` y `b`.
  
- **Expresión regular no codiciosa: `a.*?b`**
  - Coincide con la menor cantidad de caracteres entre `a` y `b`.

#### Ejemplo detallado:
Dado el texto: `"a123b456b"`
- **Expresión codiciosa (`a.*b`)** coincidirá con: `a123b456b`
- **Expresión no codiciosa (`a.*?b`)** coincidirá con: `a123b` (la primera coincidencia mínima).

### Resumen:
- **`?`** como delimitador indica **cero o una vez**.
- Puede aplicarse a caracteres individuales o grupos de caracteres.
- También se usa para hacer patrones **no codiciosos** cuando se combina con otros cuantificadores como `*` o `+`.

### Ejemplos comunes:
1. **`colou?r`**: La `u` es opcional, coincide con `color` o `colour`.
2. **`https?://`**: La `s` es opcional, coincide con `http://` o `https://`.
3. **`a?b`**: La `a` es opcional, coincide con `b` o `ab`.

## Not (^), su uso y sus peligros

El símbolo **`^`** en las **expresiones regulares** tiene varios usos, dependiendo del contexto en el que se emplee. Uno de sus principales usos es como **operador de negación** o **"not"** cuando se utiliza dentro de **clases de caracteres** (conjuntos de caracteres definidos entre corchetes `[]`). Este uso puede ser muy útil, pero también puede ser peligroso si no se maneja con cuidado, ya que puede provocar coincidencias inesperadas o no deseadas.

### **Uso de `^` como "not" en clases de caracteres**

Cuando el **`^`** se coloca **al principio** de una **clase de caracteres** (después del corchete de apertura `[`), indica que se debe hacer una negación de los caracteres que aparecen después de él, es decir, busca **cualquier carácter que no esté en la clase de caracteres**.

#### Ejemplo:
- **Expresión regular**: `[^a-z]`
  - Coincide con cualquier carácter **que no sea una letra minúscula**.
  - Coincide con: `1`, `@`, `A`, `#`, etc.
  - No coincide con: `a`, `b`, `z`, etc.

#### Ejemplo 2:
- **Expresión regular**: `[^0-9]`
  - Coincide con cualquier carácter **que no sea un dígito**.
  - Coincide con: `a`, `B`, `!`, `#`, etc.
  - No coincide con: `0`, `1`, `9`, etc.

### **Uso de `^` al inicio de la expresión regular**
Cuando el **`^`** se usa **fuera de una clase de caracteres** (es decir, fuera de los corchetes `[]`), **al inicio de la expresión regular**, tiene un significado completamente diferente. Se usa para indicar que la coincidencia debe ocurrir al **inicio de una línea o cadena**.

#### Ejemplo:
- **Expresión regular**: `^abc`
  - Coincide con la cadena `abc` **solo si está al principio** de la línea o cadena.
  - Coincide con: `abc123`, `abc...`
  - No coincide con: `123abc`, `aabc`.

### **Peligros del uso de `^` como "not"**
El uso del **`^`** para negación puede generar problemas si no se entiende bien su contexto o si se utiliza de manera imprecisa. Aquí algunos **peligros comunes**:

1. **Malinterpretación del patrón**:
   Si el **`^`** no está en la posición correcta dentro de una clase de caracteres, su significado puede cambiar y no realizar la negación esperada. 

   - **Ejemplo incorrecto**: `[a-z^]`  
     Aquí, el `^` es solo un carácter normal en la clase de caracteres, y la expresión coincidirá con cualquier letra minúscula o con el símbolo `^`, pero **no hará ninguna negación**.

2. **Negación involuntaria de caracteres esperados**:
   Si se define mal una clase de caracteres o se omiten algunos elementos clave, es posible que la negación coincida con más caracteres de los que se pretendía.

   - **Ejemplo**: `[^a-zA-Z0-9]`
     Esta expresión coincide con cualquier carácter que **no sea una letra o un número**. Esto incluye espacios, signos de puntuación y caracteres especiales, lo que puede no ser deseado si solo se quiere evitar letras y números.

3. **Coincidencias no deseadas**:
   - Si se usa **en combinaciones más complejas**, como en contraseñas o validación de datos, el uso incorrecto de `[^...]` puede provocar que se acepten caracteres que deberían haber sido rechazados o viceversa.

4. **Confusión en la lectura**:
   Cuando se mezclan varios patrones, la colocación del **`^`** puede volverse confusa para otros desarrolladores o incluso para uno mismo al revisar el código. La **legibilidad** es clave, y no siempre es obvio que `[^...]` es una negación, especialmente si la clase de caracteres es larga.

### **Consejos para evitar problemas**:
1. **Colocación correcta del `^`**: Asegúrate de que el `^` esté justo después del corchete de apertura `[` si lo estás usando para negar una clase de caracteres. Cualquier otro lugar lo interpretará como un carácter literal.

2. **Revisar el conjunto de caracteres negados**: Asegúrate de que la clase de caracteres negados esté claramente definida para evitar negar accidentalmente caracteres que deberían ser válidos.

3. **Combinación con otros patrones**: Usa paréntesis para agrupar expresiones complejas cuando sea necesario. Esto puede ayudarte a asegurarte de que el **`^`** solo aplique a la parte del patrón que realmente deseas negar.

### **Ejemplo peligroso mal usado**:
- **Expresión regular incorrecta**: `[a-z^0-9]`
  - Aquí, la `^` no está al principio, por lo que en lugar de negar letras y números, coincidirá con letras minúsculas, números, o el carácter `^`, lo cual probablemente no sea lo que se espera.

### Resumen:
- **`^` en una clase de caracteres (`[^...]`)**: Niega el conjunto de caracteres dentro de los corchetes. Coincide con cualquier carácter que **no** esté en el conjunto.
- **`^` al inicio de la expresión regular**: Coincide con el inicio de una línea o cadena.
- **Peligros**: Colocación incorrecta o ambigua puede provocar coincidencias inesperadas o patrones mal definidos, afectando la precisión de la búsqueda o validación.

Es importante usar el **`^`** con precaución y claridad, tanto para evitar malentendidos como para asegurar que el patrón funcione según lo esperado.

## Principio (^) y final de linea ($)

En las **expresiones regulares**, los símbolos **`^`** y **`$`** son usados como **anclas** para marcar el **principio** y el **final de una línea** o cadena de texto, respectivamente. Estos delimitadores son clave para controlar dónde debe ocurrir la coincidencia dentro del texto.

### **Uso del `^` (principio de línea o cadena)**
El **`^`** indica que la coincidencia debe comenzar en el **inicio de una línea o cadena**. Solo se produce una coincidencia si el patrón aparece desde el primer carácter.

#### Ejemplo:
- **Expresión regular**: `^abc`
  - Coincide con la secuencia `abc` solo si está al **principio de la línea**.
  - Coincidirá con:
    - `"abc123"`
    - `"abc..."`
  - No coincidirá con:
    - `"123abc"`
    - `"xyzabc"`

#### Ejemplo en varias líneas:
Si se tiene un texto con saltos de línea:
```
abc
123abc
```
- La expresión **`^abc`** solo coincide con la primera línea, donde `abc` está al principio.

### **Uso del `$` (final de línea o cadena)**
El **`$`** indica que la coincidencia debe estar en el **final de una línea o cadena**. Solo se produce una coincidencia si el patrón aparece justo antes del final de la línea.

#### Ejemplo:
- **Expresión regular**: `xyz$`
  - Coincide con la secuencia `xyz` solo si está al **final de la línea**.
  - Coincidirá con:
    - `"123xyz"`
    - `"abcxyz"`
  - No coincidirá con:
    - `"xyz123"`
    - `"xyzabc"`

#### Ejemplo en varias líneas:
Dado el texto:
```
xyz
abcxyz
xyz123
```
- La expresión **`xyz$`** solo coincidirá con la primera y segunda línea, donde `xyz` está al final.

### **Combinación de `^` y `$` (patrón que coincide con toda la línea o cadena)**
Cuando se usan juntos, **`^`** y **`$`** pueden definir un patrón que **debe coincidir exactamente con toda la línea o cadena**, sin importar lo que haya antes o después.

#### Ejemplo:
- **Expresión regular**: `^abc$`
  - Coincide solo si la línea contiene exactamente el texto `abc`.
  - Coincidirá con:
    - `"abc"`
  - No coincidirá con:
    - `"123abc"`
    - `"abc123"`
    - `"abc xyz"`

#### Uso práctico:
- Para asegurarse de que una cadena cumple exactamente con un formato, se puede usar **`^`** y **`$`** para que todo el texto coincida con el patrón y no solo una parte.

### **Resumen de los usos**:
1. **`^`**: Coincide con el **principio de la línea o cadena**.
   - Ejemplo: **`^abc`** solo coincide con `"abc123"` pero no con `"123abc"`.
2. **`$`**: Coincide con el **final de la línea o cadena**.
   - Ejemplo: **`xyz$`** solo coincide con `"abcxyz"` pero no con `"xyzabc"`.
3. **`^patrón$`**: Coincide con una línea o cadena que **exactamente** coincida con el patrón completo.
   - Ejemplo: **`^abc$`** solo coincide con `"abc"`, sin ningún otro carácter antes o después.

### **Casos de uso**:
- **Validación de formatos**:
   - Para asegurarse de que una entrada cumpla con un formato específico, como una dirección de correo electrónico o un número de teléfono.
   
- **Coincidencias estrictas**:
   - Para evitar coincidencias parciales en grandes cadenas de texto donde solo se desea capturar información precisa.

Ambos símbolos son extremadamente útiles para realizar coincidencias precisas y controladas en archivos de texto, entradas de usuario y más.

## Logs

En el contexto de **expresiones regulares (regex)**, los **logs** pueden ser analizados y filtrados eficientemente para extraer patrones específicos, identificar eventos o detectar errores recurrentes en grandes cantidades de datos. Las expresiones regulares permiten buscar, filtrar y extraer partes relevantes de los registros de log, lo que facilita la depuración y el análisis de datos.

### **Aplicaciones de Expresiones Regulares en Logs**
1. **Buscar patrones específicos**:
   - Filtrar líneas que contienen errores, advertencias, o eventos importantes.
   
2. **Extraer información relevante**:
   - Obtener fechas, direcciones IP, códigos de estado, usuarios, entre otros.

3. **Agrupación de eventos**:
   - Clasificar eventos por su severidad o tipo.

### **Ejemplos comunes de uso de Expresiones Regulares en Logs**

#### 1. **Buscar registros de errores (`ERROR`)**
Puedes utilizar una expresión regular para buscar cualquier línea que contenga la palabra **ERROR** en un archivo de log.

- **Regex**: `.*ERROR.*`
- Esto buscará cualquier línea que tenga la palabra **ERROR** en cualquier parte de la línea.
  
  Ejemplo:
  ```
  Sep 22 12:34:56 hostname systemd[1]: ERROR Failed to start Apache2 Web Server.
  ```

#### 2. **Filtrar por direcciones IP**
Si quieres extraer direcciones IP de los logs, puedes usar la siguiente expresión regular:

- **Regex**: `\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`
- Esto buscará direcciones IP en formato IPv4.
  
  Ejemplo:
  ```
  192.168.1.1 - - [22/Sep/2024:12:34:56] "GET / HTTP/1.1" 200 2326
  ```

  La regex coincidirá con **`192.168.1.1`**.

#### 3. **Extraer fechas en formato estándar**
En muchos logs, las fechas están en el formato `dd/mm/yyyy` o similar. Para extraer fechas en ese formato:

- **Regex**: `\b\d{2}/\d{2}/\d{4}\b`
- Esto encontrará fechas como `22/09/2024`.
  
  Ejemplo:
  ```
  192.168.1.1 - - [22/09/2024:12:34:56] "GET / HTTP/1.1" 200 2326
  ```

  La regex coincidirá con **`22/09/2024`**.

#### 4. **Extraer códigos de estado HTTP**
En logs de servidores web como **Apache** o **Nginx**, puedes extraer los códigos de estado HTTP (como 200, 404, 500, etc.) con una expresión regular.

- **Regex**: `"\s(\d{3})\s"`
- Esto buscará números de tres dígitos, que generalmente corresponden a los códigos de estado HTTP.
  
  Ejemplo:
  ```
  192.168.1.1 - - [22/Sep/2024:12:34:56] "GET / HTTP/1.1" 200 2326
  ```

  La regex coincidirá con **`200`**, el código de estado.

#### 5. **Filtrar logs por una hora específica**
Si tienes logs con marcas de tiempo y quieres filtrar eventos ocurridos en una hora específica (por ejemplo, entre 14:00 y 14:59), puedes usar:

- **Regex**: `\b14:\d{2}:\d{2}\b`
- Esto buscará cualquier marca de tiempo que comience con **14:xx:xx**.
  
  Ejemplo:
  ```
  Sep 22 14:15:45 hostname sshd[12345]: Accepted password for user
  ```

  La regex coincidirá con **`14:15:45`**.

### **Combinaciones complejas de Expresiones Regulares en Logs**

#### 1. **Buscar registros con errores y extraer detalles de IP y timestamp**
Supongamos que queremos buscar líneas que contienen la palabra **ERROR**, extraer la dirección IP, y también la marca de tiempo asociada.

- **Regex**: `(\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b).*ERROR.*(\b\d{2}/\d{2}/\d{4}\b)`
  
  Esta expresión regular busca:
  1. Una dirección IP.
  2. La palabra **ERROR**.
  3. Una fecha en formato **dd/mm/yyyy**.
  
  Ejemplo:
  ```
  192.168.1.1 - - [22/09/2024:12:34:56] "ERROR Failed to connect to database"
  ```

  Coincidirá con **`192.168.1.1`** y **`22/09/2024`**.

#### 2. **Validar direcciones IPv6**
Si necesitas trabajar con IPv6, la expresión regular es más compleja:

- **Regex**: `\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b`
  
  Esto buscará direcciones IP en formato IPv6.
  
  Ejemplo:
  ```
  fe80::1ff:fe23:4567:890a - - [22/Sep/2024:12:34:56] "GET / HTTP/1.1" 200 2326
  ```

  Coincidirá con **`fe80::1ff:fe23:4567:890a`**.

### **Delimitadores y operadores útiles en logs con regex**

1. **`.*`**: Coincide con cualquier cantidad de caracteres (incluso ninguno).
   - Ejemplo: `.*ERROR.*` busca líneas que contengan la palabra ERROR en cualquier parte.
   
2. **`|`**: Operador OR.
   - Ejemplo: `ERROR|WARNING` busca líneas que contengan **ERROR** o **WARNING**.
   
3. **`^` y `$`**: Coinciden con el inicio y el final de una línea.
   - Ejemplo: `^ERROR` coincide con líneas que comienzan con **ERROR**.

4. **`[0-9]`**: Clase de caracteres que coincide con cualquier dígito del 0 al 9.
   - Ejemplo: `\d{3}` coincide con cualquier secuencia de tres dígitos, útil para códigos de estado HTTP.

### **Beneficios de usar expresiones regulares en logs**
- **Automatización**: Permiten extraer información relevante automáticamente en sistemas grandes.
- **Precisión**: Filtran eventos específicos sin tener que revisar manualmente cada línea.
- **Flexibilidad**: Pueden ajustarse a distintos formatos de logs, ya que son agnósticas al formato del texto.

### **Herramientas que usan regex para análisis de logs**
1. **grep**:
   - Comando de Unix/Linux para buscar patrones en archivos.
   - Ejemplo:
     ```
     grep -E "ERROR.*192\.168\.\d{1,3}\.\d{1,3}" /var/log/syslog
     ```

2. **Logwatch**:
   - Herramienta para generar informes a partir de registros de log, donde se pueden aplicar filtros con regex.

3. **Splunk y ELK**:
   - Herramientas de análisis que permiten usar regex para extraer datos de registros en tiempo real.

En resumen, las **expresiones regulares** son una herramienta poderosa para analizar logs y extraer información crítica de manera automatizada y precisa.

## Teléfono

Las **expresiones regulares** son extremadamente útiles para validar, formatear o extraer números de teléfono en diversos formatos. Dependiendo del país o región, los números de teléfono pueden tener distintas longitudes y formatos. A continuación, te explico cómo las expresiones regulares pueden aplicarse a la gestión de números de teléfono.

### **Validación de números de teléfono con expresiones regulares**

#### 1. **Formato internacional (E.164)**
El formato E.164 es el estándar internacional para números de teléfono y sigue el patrón de un código de país seguido por el número de abonado, sin espacios ni caracteres especiales.

- **Ejemplo de número**: `+12345678901`
- **Regex**: `^\+?[1-9]\d{1,14}$`
  - `^`: Comienza la expresión.
  - `\+?`: Coincide con el símbolo "+" opcional.
  - `[1-9]`: El primer dígito debe ser un número del 1 al 9.
  - `\d{1,14}`: Acepta entre 1 y 14 dígitos después del código del país.

#### 2. **Formato estándar con separadores**
Muchos números de teléfono se escriben con separadores, como espacios, guiones o puntos.

- **Ejemplo de número**: `123-456-7890` o `123.456.7890`
- **Regex**: `^\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}$`
  - `\(?\d{3}\)?`: Acepta un código de área opcional entre paréntesis.
  - `[-. ]?`: Acepta un guion, punto o espacio opcional como separador.
  - `\d{3}`: Acepta los siguientes tres dígitos.
  - `\d{4}`: Acepta los últimos cuatro dígitos.

#### 3. **Formato con código de país**
Si se espera que el número incluya un código de país, se puede agregar el símbolo "+" y permitir espacios o guiones entre los números.

- **Ejemplo de número**: `+1-234-567-8901`
- **Regex**: `^\+?[0-9]{1,4}?[-. ]?\(?[0-9]{1,4}?\)?[-. ]?[0-9]{1,4}[-. ]?[0-9]{1,9}$`
  - `\+?[0-9]{1,4}?`: Código de país opcional con hasta 4 dígitos.
  - `[-. ]?`: Separador opcional (guion, punto o espacio).
  - `\(?[0-9]{1,4}?\)?`: Código de área opcional, entre paréntesis o sin ellos.
  - `\d{1,9}`: El número de teléfono en sí, que puede variar en longitud.

#### 4. **Números de teléfono de diferentes países**
Dependiendo del país, el formato de los números de teléfono puede variar. Aquí algunos ejemplos:

- **Estados Unidos** (10 dígitos, con o sin guiones):
  - **Regex**: `^\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}$`

- **Reino Unido** (empiezan con 0, 10 o 11 dígitos):
  - **Regex**: `^(\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}$`

- **México** (código de país y 10 dígitos):
  - **Regex**: `^\+?52\s?(\d{2,3})?[-. ]?\d{3}[-. ]?\d{4}$`

### **Aplicaciones en sistemas de validación de números de teléfono**
Las expresiones regulares se usan ampliamente en:

1. **Formularios web**: Para validar que el usuario introduzca un número de teléfono en el formato adecuado.
2. **Aplicaciones móviles**: Asegurar que los números de contacto se ingresen correctamente antes de almacenar o realizar llamadas.
3. **Bases de datos**: Verificar o formatear números telefónicos antes de guardarlos.
4. **Sistemas CRM**: Estandarizar el formato de los números de teléfono.

### **Uso avanzado: Normalización de números de teléfono**
Cuando los números de teléfono se almacenan en una base de datos, es común normalizarlos en un formato específico para simplificar el procesamiento posterior. Las expresiones regulares pueden ser útiles para:

- **Eliminar caracteres no deseados**: Como espacios, guiones o paréntesis.
- **Agregar el código de país** si está ausente.
  
**Ejemplo de normalización**:
Transformar números de teléfono con diferentes formatos (por ejemplo, `123-456-7890`, `(123) 456 7890`, `+1 123 456 7890`) en un formato unificado: `+11234567890`.

Para eliminar todos los separadores:
- **Regex**: `[^0-9+]` 
- Esto elimina cualquier carácter que no sea un número o el signo `+`.

En resumen, las **expresiones regulares** son esenciales para validar, extraer y normalizar números de teléfono en diversas aplicaciones y sistemas, ayudando a garantizar consistencia y precisión.

## URLs

Las **expresiones regulares** son extremadamente útiles para trabajar con **URLs** (Uniform Resource Locators) en tareas como la validación, extracción o manipulación de enlaces web. A continuación, te explico cómo se aplican las expresiones regulares para manejar URLs.

### **Validación de URLs con expresiones regulares**

Una URL válida generalmente tiene una estructura bien definida que incluye el protocolo (http, https, ftp), el dominio, el puerto opcional y los posibles subdirectorios, parámetros o fragmentos. Aunque las URL pueden ser complejas, una expresión regular puede ayudar a validar las más comunes.

#### 1. **Estructura básica de una URL**
La estructura típica de una URL es:

```
protocolo://dominio.extensión/ruta?parámetros#fragmento
```

- **Protocolo**: `http`, `https`, `ftp`
- **Dominio**: `www.example.com`
- **Ruta**: `/folder/page`
- **Parámetros**: `?key1=value1&key2=value2`
- **Fragmento**: `#section`

#### 2. **Expresión regular básica para validar URLs**
Una expresión regular que valide una URL puede ser:

```regex
^https?:\/\/([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,}(\/[a-zA-Z0-9#]+\/?)*$
```

- `^https?`: El protocolo debe comenzar con "http" o "https".
- `:\/\/`: Doble barra después del protocolo.
- `([a-zA-Z0-9\-]+\.)+`: Un dominio con caracteres alfanuméricos y guiones, seguido por un punto.
- `[a-zA-Z]{2,}`: La extensión del dominio, con al menos 2 caracteres (ej. ".com", ".org").
- `(\/[a-zA-Z0-9#]+\/?)*`: Subdirectorios opcionales, seguidos por "/".
- `$`: El final de la cadena.

#### 3. **Expresión regular avanzada para URLs**
Una expresión regular más robusta que incluye subdominios, parámetros y fragmentos:

```regex
^(https?|ftp):\/\/([a-zA-Z0-9\.-]+)\.([a-zA-Z]{2,6})(\/[a-zA-Z0-9\&%_\./-~-]*)?(\?[a-zA-Z0-9\&%_\./-~-]*)?(#[a-zA-Z0-9\&%_\./-~-]*)?$
```

- **Protocolo**: `^(https?|ftp)` permite los protocolos `http`, `https` y `ftp`.
- **Dominio**: `([a-zA-Z0-9\.-]+)` permite dominios con letras, números, puntos y guiones.
- **Extensión**: `\.([a-zA-Z]{2,6})` permite extensiones de dominio de 2 a 6 caracteres (ej., `.com`, `.museum`).
- **Ruta opcional**: `(\/[a-zA-Z0-9\&%_\./-~-]*)?` acepta una ruta que puede incluir directorios y archivos.
- **Parámetros opcionales**: `(\?[a-zA-Z0-9\&%_\./-~-]*)?` acepta parámetros de la forma `?key=value`.
- **Fragmentos opcionales**: `(\#[a-zA-Z0-9\&%_\./-~-]*)?` permite fragmentos de la forma `#section`.

#### 4. **Expresiones regulares simplificadas por secciones**

- **Protocolo**: `^(https?|ftp)://`
  - Solo permite `http`, `https` o `ftp`.

- **Dominio**: `([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,6}`
  - Permite dominios como `www.example.com`, `sub.example.org`.

- **Ruta**: `(\/[a-zA-Z0-9\-._~!$&'()*+,;=:@%]*)*`
  - Permite rutas que incluyan una combinación de caracteres y símbolos.

- **Parámetros**: `(\?[a-zA-Z0-9\-._~!$&'()*+,;=:@%]*)?`
  - Incluye parámetros como `?id=123`.

- **Fragmentos**: `(\#[a-zA-Z0-9\-._~!$&'()*+,;=:@%]*)?`
  - Para fragmentos como `#section`.

### **Aplicaciones comunes en sistemas**

1. **Validación de formularios**: Para asegurarse de que un campo de URL sea válido antes de enviar un formulario web.
2. **Extracción de enlaces**: En documentos o páginas HTML para extraer todas las URLs mediante scrapers.
3. **Normalización de URLs**: Asegurarse de que las URLs sigan un formato estándar antes de ser almacenadas en bases de datos o utilizadas en sistemas de análisis.
4. **Filtrado de contenido**: En sistemas de seguridad para bloquear o permitir ciertas URLs basadas en patrones específicos.

### **Uso avanzado: Captura de componentes específicos**

Las expresiones regulares pueden usarse para **extraer partes específicas de una URL**, como el protocolo, el dominio, la ruta o los parámetros:

- **Capturar protocolo**: `^(https?|ftp)`
- **Capturar dominio**: `([a-zA-Z0-9-]+\.[a-zA-Z]{2,6})`
- **Capturar ruta**: `\/[a-zA-Z0-9\-._~!$&'()*+,;=:@%]*`
- **Capturar parámetros**: `\?[a-zA-Z0-9\-._~!$&'()*+,;=:@%]*`

### **Ejemplos de uso**
- **Validar URLs en un formulario web** para evitar que los usuarios ingresen enlaces incorrectos.
- **Extraer todos los enlaces de una página web** con una expresión regular en un script de scraping.
- **Filtrar URLs maliciosas** o prohibidas en un sistema de seguridad.

En resumen, las expresiones regulares son una herramienta poderosa para manejar URLs en múltiples aplicaciones, facilitando la validación, extracción y procesamiento de enlaces en una amplia gama de contextos.

**Ejemplo**: `https?:\/\/[\w\-\.]*\.\w{2,5}\/?\S*`

## Mails

Quedamos en que ya podemos definir URLs, y dentro de las URLs están los dominios. No es infalible, pero es muy útil para detectar la gran mayoría de errores que cometen los usuarios al escribir sus emails.

Las **expresiones regulares** (regex) son extremadamente útiles para validar, extraer o manipular direcciones de correo electrónico (**mails**). En el caso de los correos electrónicos, el formato es bastante estándar, pero pueden existir variaciones debido a diferentes reglas permitidas por los proveedores de correos. A continuación, te explico cómo usar expresiones regulares para manejar correos electrónicos.

### **Estructura básica de un correo electrónico**
Una dirección de correo electrónico tiene la siguiente estructura:

```
usuario@dominio.extension
```

- **Usuario**: Puede contener letras, números, puntos (`.`), guiones (`-`), guiones bajos (`_`), entre otros caracteres.
- **Dominio**: Generalmente contiene letras, números y puntos, separados por subdominios opcionales.
- **Extensión**: Por lo general, es una cadena de 2 a 6 caracteres (por ejemplo, `.com`, `.org`, `.info`).

### **Expresión regular básica para validar un correo electrónico**
Un ejemplo simple de regex que valida una dirección de correo electrónico:

```regex
^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}$
```

- `^`: Indica el inicio de la cadena.
- `[a-zA-Z0-9._%+-]+`: Captura una secuencia de letras, números, puntos (`.`), guiones bajos (`_`), porcentajes (`%`), más (`+`) o guiones (`-`) en la parte del usuario.
- `@`: El símbolo `@` que separa el nombre de usuario del dominio.
- `[a-zA-Z0-9.-]+`: Captura el dominio que puede contener letras, números, puntos (`.`) o guiones (`-`).
- `\.`: Un punto que separa el dominio de la extensión.
- `[a-zA-Z]{2,6}`: La extensión del dominio, que suele ser de 2 a 6 caracteres (como `.com`, `.edu`, `.gov`).
- `$`: Indica el final de la cadena.

### **Expresión regular avanzada para correos electrónicos**
Una expresión regular más completa que maneja casos más complicados podría ser:

```regex
^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}$
```

Esta versión maneja:
- Nombres de usuario con una amplia gama de caracteres permitidos.
- Dominios que pueden tener varios niveles, como `sub.dominio.com`.
- Extensiones de dominio que pueden tener 2 a 6 caracteres.

### **Explicación detallada del regex**
1. **Parte del usuario (`usuario`)**:
   ```regex
   [a-zA-Z0-9._%+-]+
   ```
   - Acepta letras (mayúsculas y minúsculas), números, puntos (`.`), guiones bajos (`_`), porcentajes (`%`), más (`+`) y guiones (`-`).
   - El signo `+` asegura que haya al menos un carácter.

2. **El símbolo `@`**:
   - Se captura de forma literal con `@`.

3. **Parte del dominio (`dominio.extension`)**:
   ```regex
   [a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}
   ```
   - El dominio puede contener letras, números, puntos y guiones. La expresión `\.` captura el punto antes de la extensión.
   - La extensión debe tener entre 2 y 6 caracteres.

### **Ejemplos de correos electrónicos válidos**
- `usuario@dominio.com`
- `user.name@sub.dominio.org`
- `correo123@ejemplo.co`
- `nombre-apellido@empresa.edu`

### **Ejemplos de correos electrónicos inválidos**
- `usuario@dominio` (sin extensión).
- `usuario@@dominio.com` (dos signos `@`).
- `usuario@dominio..com` (puntos dobles en el dominio).

### **Aplicaciones comunes de expresiones regulares para correos**
1. **Validación de formularios**: Se utiliza para asegurarse de que el campo de correo electrónico sea válido antes de enviar un formulario web.
2. **Filtrado de correos electrónicos**: En sistemas de correo o bases de datos, para extraer o listar direcciones de correo válidas.
3. **Extracción de correos electrónicos de texto**: Al analizar un archivo de texto o una página web, las expresiones regulares pueden usarse para extraer todas las direcciones de correo electrónico.
4. **Normalización**: Para limpiar y estandarizar direcciones de correo electrónico antes de almacenarlas o procesarlas.

### **Ejemplo de uso en scripts**
Para extraer correos electrónicos de un texto en un archivo, podrías usar una expresión regular en un script de Python o Bash. Ejemplo en Python:

```python
import re

texto = """
Aquí hay algunos correos: usuario1@example.com, persona.name@domain.org y user@empresa.edu.
"""
# Regex para extraer correos electrónicos
patron = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}'
correos = re.findall(patron, texto)
print(correos)
```

Este script devolvería una lista con todas las direcciones de correo encontradas en el texto.

### **Conclusión**
Las expresiones regulares son una herramienta poderosa para trabajar con correos electrónicos, permitiendo desde la validación en formularios hasta la extracción en sistemas complejos. La flexibilidad del regex permite manejar la variedad de formatos que pueden tener los correos electrónicos mientras se asegura la exactitud en las aplicaciones donde se implementa.

## localizacione

Las **expresiones regulares** (o regex) son una herramienta poderosa para buscar y manipular texto de manera eficiente, y también pueden ser útiles para identificar **localizaciones** dentro de datos. Las localizaciones pueden incluir **países**, **ciudades**, **direcciones**, y **coordenadas geográficas**, entre otras. Dependiendo del tipo de localización que busques, las expresiones regulares pueden adaptarse a distintos patrones de búsqueda.

### Aplicaciones de las Expresiones Regulares en Localizaciones

1. **Direcciones IP**
   - Se pueden usar expresiones regulares para identificar y validar direcciones IP en formato IPv4 o IPv6.
   
   **Expresión Regular para IPv4:**
   ```regex
   \b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b
   ```
   Esta regex valida direcciones IP como `192.168.0.1`.

2. **Direcciones de Correo Electrónico (Emails)**
   - Las localizaciones pueden incluir la validación de correos electrónicos, lo que es útil en formularios web o sistemas de registro.

   **Expresión Regular para Correo Electrónico:**
   ```regex
   [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
   ```

3. **Coordenadas Geográficas**
   - Las coordenadas geográficas (latitud y longitud) pueden seguir patrones específicos que también se pueden validar o extraer con expresiones regulares.

   **Expresión Regular para Coordenadas Geográficas (en formato decimal):**
   ```regex
   [-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)
   ```
   Esto valida coordenadas como `40.7128, -74.0060` (Nueva York).

4. **Nombres de Países o Ciudades**
   - Puedes usar expresiones regulares para localizar y capturar nombres de países o ciudades a partir de listas predefinidas o patrones textuales.
   
   **Expresión Regular para Países Comunes (ejemplo simplificado):**
   ```regex
   \b(Argentina|Brasil|Chile|Colombia|México|España|Francia)\b
   ```

5. **Códigos Postales**
   - Los códigos postales también pueden variar de un país a otro, y las expresiones regulares se pueden utilizar para validar estos formatos.

   **Expresión Regular para Códigos Postales en Estados Unidos:**
   ```regex
   \b\d{5}(?:-\d{4})?\b
   ```

   **Expresión Regular para Códigos Postales en España:**
   ```regex
   \b\d{5}\b
   ```

6. **Direcciones Web (URLs)**
   - Localizar URLs en texto también es una tarea común que se puede hacer con expresiones regulares.
   
   **Expresión Regular para URL:**
   ```regex
   https?:\/\/(www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(/[a-zA-Z0-9#?&%=.-]*)?
   ```

7. **Números de Teléfono**
   - Dependiendo de la localización, los números de teléfono tienen diferentes formatos. Las expresiones regulares permiten validar números de teléfonos nacionales e internacionales.

   **Expresión Regular para Números de Teléfono Internacionales:**
   ```regex
   \+?[1-9]{1,4}[-\s]?\(?\d{1,3}\)?[-\s]?\d{1,4}[-\s]?\d{1,4}[-\s]?\d{1,9}
   ```

   **Ejemplos válidos:**
   - `+1-800-555-5555` (EE.UU.)
   - `+44 20 7946 0958` (Reino Unido)

### Uso de Expresiones Regulares en Localizaciones

Las expresiones regulares son esenciales en aplicaciones que manejan grandes volúmenes de datos, como:

1. **Extracción de Datos Geográficos**
   - Se utilizan para extraer localizaciones en bases de datos de direcciones o coordenadas geográficas de texto libre.

2. **Validación de Formularios**
   - Al recolectar datos de localización, como códigos postales, direcciones IP o coordenadas, las expresiones regulares validan automáticamente los datos de entrada.

3. **Sistemas de Geolocalización**
   - En servicios que detectan o procesan direcciones IP o geográficas, las regex ayudan a estructurar datos desorganizados.

### Conclusión

Las expresiones regulares permiten la **validación**, **búsqueda** y **manipulación** de localizaciones en texto. Pueden reconocer patrones como **direcciones**, **números de teléfono**, **coordenadas geográficas**, **direcciones IP**, y otros elementos geolocalizados con alta precisión, lo que es útil para aplicaciones que manejan información geográfica o direcciones globales.

[what3words | Addressing the world](https://what3words.com/)

## Búsqueda y reemplazo

La **búsqueda y reemplazo** utilizando **expresiones regulares** es una técnica poderosa en los scripts y editores de texto. Permite localizar patrones específicos dentro de un texto y reemplazarlos de manera eficiente. Esta función es ampliamente utilizada en tareas de procesamiento de texto, edición masiva de archivos y manipulación de datos.

### Sintaxis de Búsqueda y Reemplazo en Varios Contextos

#### 1. **Uso en `sed` (Stream Editor)**

`sed` es una herramienta muy utilizada en sistemas Unix/Linux para realizar búsquedas y reemplazos en flujos de texto o archivos.

- **Comando Básico:**
  ```bash
  sed 's/patrón_a_buscar/reemplazo/' archivo.txt
  ```

  **Ejemplo:**
  Si quieres reemplazar todas las ocurrencias de "foo" por "bar" en un archivo:
  ```bash
  sed 's/foo/bar/g' archivo.txt
  ```
  - `s`: significa "sustituir" (substitute).
  - `/g`: indica que se reemplacen todas las ocurrencias en una línea.

- **Usar Expresiones Regulares:**
  ```bash
  sed 's/[0-9]\{2\}-[0-9]\{3\}/XXXX-XXX/g' archivo.txt
  ```
  Esto reemplaza cualquier coincidencia de un número de 2 dígitos seguido de un guion y un número de 3 dígitos por `XXXX-XXX`.

#### 2. **Uso en `vim` o `vi` (Editor de Texto)**

En los editores como `vim` o `vi`, puedes realizar reemplazos utilizando la siguiente sintaxis:

- **Comando Básico:**
  ```bash
  :%s/patrón_a_buscar/reemplazo/g
  ```

  **Ejemplo:**
  Reemplazar todas las ocurrencias de "error" por "solución":
  ```bash
  :%s/error/solución/g
  ```

  - `%`: significa que el reemplazo se hace en todo el archivo.
  - `g`: indica que se aplique en todas las ocurrencias de la línea.

- **Con Expresiones Regulares:**
  ```bash
  :%s/\d\{2\}-\d\{3\}/XXXX-XXX/g
  ```

#### 3. **Uso en Lenguajes de Programación (Ejemplo en Python)**

En lenguajes como Python, la búsqueda y reemplazo con expresiones regulares se realiza usando el módulo `re`.

- **Ejemplo en Python:**
  ```python
  import re
  
  texto = "El número de teléfono es 123-456-7890"
  nuevo_texto = re.sub(r'\d{3}-\d{3}-\d{4}', 'XXX-XXX-XXXX', texto)
  print(nuevo_texto)
  ```

  - `re.sub()`: se utiliza para buscar un patrón y reemplazarlo por un nuevo texto.
  - `\d{3}-\d{3}-\d{4}`: patrón que busca números de teléfono en el formato `123-456-7890`.
  - `XXX-XXX-XXXX`: el reemplazo.

#### 4. **Uso en `awk`**

Aunque `awk` es más utilizado para el procesamiento de texto basado en columnas, también soporta búsqueda y reemplazo.

- **Comando Básico:**
  ```bash
  awk '{gsub(/patrón_a_buscar/, "reemplazo")}1' archivo.txt
  ```

  **Ejemplo:**
  ```bash
  awk '{gsub(/foo/, "bar")}1' archivo.txt
  ```

#### 5. **Uso en Herramientas de Edición de Texto (Ejemplo en Sublime Text, VS Code)**

En editores de texto modernos como Sublime Text o VS Code, puedes realizar búsquedas y reemplazos avanzados usando expresiones regulares.

- **Comando en Sublime Text:**
  1. Pulsa `Ctrl + H` para abrir la herramienta de búsqueda y reemplazo.
  2. Marca la opción de "regex" (representada por `.*`).
  3. Introduce el patrón a buscar y el reemplazo.

  **Ejemplo:**
  Para reemplazar números de teléfono:
  - Búsqueda: `(\d{3})-(\d{3})-(\d{4})`
  - Reemplazo: `($1) $2-$3`

### Caracteres Especiales y Delimitadores en Expresiones Regulares

- `.`: Cualquier carácter excepto salto de línea.
- `*`: Cero o más ocurrencias del carácter anterior.
- `+`: Una o más ocurrencias del carácter anterior.
- `?`: Cero o una ocurrencia del carácter anterior.
- `\d`: Cualquier dígito (equivalente a `[0-9]`).
- `\w`: Cualquier carácter de palabra (letras, dígitos o guiones bajos).
- `\s`: Cualquier espacio en blanco (espacio, tabulación, etc.).
- `{n,m}`: Aparece al menos `n` veces, pero no más de `m`.

### Aplicaciones Comunes de Búsqueda y Reemplazo con Expresiones Regulares

1. **Reemplazar números de teléfono** en un texto para ocultar la información personal.
2. **Normalizar fechas** que están en diferentes formatos (por ejemplo, convertir `dd/mm/yyyy` a `yyyy-mm-dd`).
3. **Eliminar espacios en blanco adicionales** o caracteres innecesarios en el texto.
4. **Sanitizar direcciones de correo electrónico** para anonimización de datos.
5. **Reformatear URLs** o texto HTML en archivos para limpieza o actualización masiva de documentos.

### Conclusión

La búsqueda y reemplazo con expresiones regulares es una herramienta clave en la automatización de tareas de procesamiento de texto y datos, lo que permite trabajar con grandes volúmenes de texto de manera eficiente. Ya sea en scripts, editores de texto o lenguajes de programación, las expresiones regulares brindan flexibilidad para buscar patrones y modificarlos según las necesidades del usuario.

**Nota**: $1,$2 borra todo lo que no esta en las clases

se usa `^\d+::([\w\s:,\(\)'\.\-&!\/]+)\s\((\d\d\d\d)\)::.*$` pasar a sql `insert into movies (year, title) values($2, '$1');`, json: `{title:"$1", year:$2}`

## Uso de REGEX para descomponer querys GET

Al hacer consultas a sitios web mediante el método GET se envían todas las variables al servidor a través de la misma URL.

La parte de esta url que viene luego del signo de interrogación ? se le llama query del request que es: `variable1=valor1&variable2=valor2&...` y así tantas veces como se necesite. En esta clase veremos como extraer estas variables usando expresiones regulares.

El uso de **expresiones regulares (REGEX)** para descomponer las **queries GET** de URLs es muy útil para extraer parámetros y valores clave, especialmente en la manipulación de datos web. Una query GET en una URL tiene la forma:

```
https://example.com/page?param1=value1&param2=value2&param3=value3
```

Aquí, todo lo que viene después del signo de interrogación (`?`) es la **cadena de consulta** o **query string**, con los parámetros clave y sus valores.

### Pasos para descomponer una query GET

1. **Identificar el patrón básico de la query**:
   Las queries GET suelen estar formadas por pares de **clave=valor** separados por el carácter `&`.

2. **Uso de REGEX para extraer parámetros y valores**:
   Un patrón REGEX típico para descomponer una query string es el siguiente:

   ```regex
   ([\w%]+)=([\w%]+)
   ```

   Donde:
   - `[\w%]+` busca una secuencia de caracteres alfanuméricos o el símbolo `%` (para valores codificados en URL).
   - El símbolo `=` separa el parámetro de su valor.
   - Este patrón se repite para cada par clave-valor.

### Descomposición en Python usando REGEX

Veamos cómo se puede usar esta expresión regular en Python para descomponer una query string:

#### Ejemplo de Código Python

```python
import re

# Query string de ejemplo
query = "param1=value1&param2=value2&param3=value3"

# Expresión regular para extraer pares clave-valor
pattern = r'([\w%]+)=([\w%]+)'

# Usar findall para encontrar todos los pares clave-valor
matches = re.findall(pattern, query)

# Mostrar resultados
for match in matches:
    print(f"Parámetro: {match[0]}, Valor: {match[1]}")
```

#### Salida:
```
Parámetro: param1, Valor: value1
Parámetro: param2, Valor: value2
Parámetro: param3, Valor: value3
```

### Explicación:

- `re.findall()` busca todas las coincidencias del patrón en la query string y devuelve una lista de tuplas donde cada tupla contiene un par clave-valor.
- Este enfoque es simple y puede manejar queries con múltiples parámetros.

### Ampliaciones para casos más complejos

1. **Soporte para valores codificados en URL**: Si los valores de la query string están codificados, pueden contener caracteres especiales como `%20` para espacios. En ese caso, deberías incluir `%` y caracteres hexadecimales en tu expresión regular.

   ```regex
   ([\w%]+)=([\w%]+)
   ```

   Esto captura valores que están codificados en URL.

2. **Opcionalidad de parámetros vacíos**: A veces, los valores de los parámetros pueden estar vacíos, por ejemplo:
   
   ```
   https://example.com/page?param1=value1&param2=&param3=value3
   ```

   Para manejar este caso, puedes ajustar la regex para que permita valores vacíos:
   
   ```regex
   ([\w%]+)=([\w%]*)
   ```

   El asterisco (`*`) en la segunda parte del patrón permite que el valor esté vacío.

### Usar REGEX en otras herramientas

#### 1. **Bash con `grep` o `sed`**
   En bash, puedes usar `grep` o `sed` para hacer algo similar. Por ejemplo:

   ```bash
   echo "param1=value1&param2=value2" | grep -oP '(\w+)=(\w+)'
   ```

#### 2. **Uso en JavaScript**

   En JavaScript, puedes usar la función `match()` o `exec()` para aplicar la expresión regular:

   ```javascript
   const query = "param1=value1&param2=value2&param3=value3";
   const regex = /([\w%]+)=([\w%]+)/g;
   let match;
   while ((match = regex.exec(query)) !== null) {
       console.log(`Parámetro: ${match[1]}, Valor: ${match[2]}`);
   }
   ```

#### 3. **Herramientas para analizar URLs completas**

   Si quieres trabajar con URLs completas y no solo la query string, puedes usar el siguiente patrón:

   ```regex
   https?:\/\/[\w.-]+\/[\w.-]*\?([\w%=&]+)
   ```

   Esto separa la URL principal de la query string.

[\?&](\w+)=([^&\n]+) `- $1 => $2`

### Conclusión

El uso de **expresiones regulares** es una herramienta poderosa para descomponer queries GET y extraer parámetros de una URL. Esta técnica es especialmente útil cuando trabajas con URLs complejas, análisis de logs o procesamiento web.

## Explicación del Proyecto

Vamos a utilizar un archivo de resultados de partidos de fútbol histórico con varios datos. El archivo es un csv de más de 39000 líneas diferentes.

Con cada lenguaje intentaremos hacer una solución un poquito diferente para aprovecharlo y saber cómo utilizar expresiones regulares en cada uno de los lenguajes.

Usaremos las expresiones regulares en:

- Perl
- PHP
- Python
- Javascript

### comando bash

`ls -l results.csv` muestra en kilobytes
`ls -lh results.csv` muestar en MegaBytes
`wc -l results.csv ` se utiliza para contar líneas
`more results.csv` se utiliza para visualizar el contenido de archivos de texto de manera paginada

## Perl

**Perl** es un lenguaje de programación de propósito general conocido por su potencia en el procesamiento de texto, su flexibilidad y la capacidad para realizar tareas de administración del sistema, desarrollo web, automatización, manipulación de archivos y más. Perl se popularizó principalmente por su capacidad de manejar expresiones regulares y archivos de texto de manera eficiente, siendo una opción común para scripts y tareas del sistema en los años 90 y 2000.

### Características de Perl:
1. **Procesamiento de texto**: Perl es extremadamente potente para manipular, analizar y procesar grandes cantidades de texto utilizando expresiones regulares.
2. **Lenguaje interpretado**: Perl no requiere compilación; los scripts se ejecutan directamente por el intérprete de Perl.
3. **Flexibilidad**: Puedes escribir código de manera muy concisa o detallada, lo que lo hace adecuado tanto para pequeños scripts como para aplicaciones más grandes.
4. **Multiplataforma**: Funciona en una variedad de sistemas operativos, como Unix, Linux, Windows, y macOS.
5. **CPAN (Comprehensive Perl Archive Network)**: Una extensa biblioteca de módulos y paquetes reutilizables para casi cualquier tarea imaginable, facilitando el desarrollo en Perl.

### Sintaxis básica de Perl:

1. **Hola Mundo en Perl**:
   ```perl
   #!/usr/bin/perl
   print "Hola, Mundo!\n";
   ```

2. **Variables**:
   - **Escalares**: Almacenan un solo valor (números o cadenas), se identifican con `$`.
     ```perl
     $nombre = "Juan";
     $edad = 30;
     ```
   - **Arrays**: Listas de valores, se identifican con `@`.
     ```perl
     @frutas = ("manzana", "banana", "naranja");
     ```
   - **Hashes**: Pares clave-valor, se identifican con `%`.
     ```perl
     %capitales = ("Argentina" => "Buenos Aires", "Colombia" => "Bogotá");
     ```

3. **Estructuras de control**:
   - **If-else**:
     ```perl
     if ($edad > 18) {
         print "Es mayor de edad\n";
     } else {
         print "Es menor de edad\n";
     }
     ```

   - **Bucles**:
     ```perl
     # Bucle for
     for (my $i = 0; $i < 10; $i++) {
         print "$i\n";
     }

     # Bucle foreach para arrays
     foreach my $fruta (@frutas) {
         print "$fruta\n";
     }
     ```

4. **Expresiones regulares**:
   Perl es famoso por su soporte nativo a expresiones regulares. Por ejemplo, buscar una palabra en una cadena:
   ```perl
   if ($cadena =~ /Perl/) {
       print "La cadena contiene 'Perl'\n";
   }
   ```

5. **Lectura y escritura de archivos**:
   - Para leer un archivo:
     ```perl
     open(my $archivo, "<", "datos.txt") or die "No se pudo abrir el archivo: $!";
     while (my $linea = <$archivo>) {
         print $linea;
     }
     close($archivo);
     ```

   - Para escribir en un archivo:
     ```perl
     open(my $archivo, ">", "salida.txt") or die "No se pudo abrir el archivo: $!";
     print $archivo "Esta es una línea de texto.\n";
     close($archivo);
     ```

### Aplicaciones comunes de Perl:

1. **Administración del sistema**: Tareas de automatización, como el manejo de archivos, logs y scripts para la gestión de servidores.
2. **Procesamiento de texto**: Transformación y manipulación de grandes cantidades de texto, como extracción de datos y generación de informes.
3. **Desarrollo web**: Perl se utiliza en el desarrollo web (especialmente en la era del CGI) para generar contenido dinámico.
4. **Bioinformática**: Debido a sus capacidades de manipulación de texto, Perl ha sido utilizado para procesar secuencias genómicas.
5. **Testing y Automatización**: Módulos como `Test::Simple` y `Test::More` hacen de Perl una excelente opción para pruebas automatizadas.

Perl es un lenguaje robusto y poderoso, especialmente en tareas de automatización y procesamiento de datos. Aunque ha sido superado en popularidad por lenguajes más recientes como Python y Ruby, sigue siendo una herramienta valiosa en varias industrias y aplicaciones.

Las **expresiones regulares** en Perl son una de las características más poderosas y distintivas del lenguaje. Perl es famoso por su soporte nativo y flexible para la manipulación de texto utilizando **regex** (expresiones regulares), que permiten buscar, extraer, reemplazar y validar patrones dentro de cadenas de texto.

### Sintaxis básica de expresiones regulares en Perl

1. **Operador de coincidencia `=~`**:
   El operador `=~` se utiliza para aplicar una expresión regular a una cadena.
   ```perl
   $cadena = "Hola, Mundo!";
   if ($cadena =~ /Mundo/) {
       print "¡Coincidencia encontrada!\n";
   }
   ```
   En este ejemplo, `/Mundo/` es la expresión regular, y busca la palabra "Mundo" dentro de la variable `$cadena`.

2. **Negación de coincidencia `!~`**:
   El operador `!~` se utiliza cuando se quiere comprobar que una cadena **no** coincide con la expresión regular.
   ```perl
   if ($cadena !~ /Adiós/) {
       print "¡No se encontró la palabra 'Adiós'!\n";
   }
   ```

### Caracteres especiales en regex de Perl

- **`.` (punto)**: Coincide con cualquier carácter excepto el salto de línea (`\n`).
  ```perl
  if ("abc" =~ /a.c/) { print "Coincide\n"; }
  # Coincidiría con cualquier cadena que tenga una "a", seguida de cualquier carácter y luego una "c", como "abc" o "a1c".
  ```

- **`^`**: Indica el **inicio** de una línea.
  ```perl
  if ("Hola Mundo" =~ /^Hola/) { print "Coincide\n"; }
  # Coincide con cadenas que empiezan con "Hola".
  ```

- **`$`**: Indica el **final** de una línea.
  ```perl
  if ("Hola Mundo" =~ /Mundo$/) { print "Coincide\n"; }
  # Coincide con cadenas que terminan en "Mundo".
  ```

- **`\d`**: Coincide con cualquier dígito (equivalente a `[0-9]`).
  ```perl
  if ("123abc" =~ /\d\d\d/) { print "Coincide\n"; }
  ```

- **`\w`**: Coincide con cualquier carácter alfanumérico o guion bajo (equivalente a `[A-Za-z0-9_]`).
  ```perl
  if ("nombre_usuario" =~ /\w+/) { print "Coincide\n"; }
  ```

- **`\s`**: Coincide con cualquier carácter de espacio en blanco (espacio, tabulador, salto de línea).
  ```perl
  if ("Hola Mundo" =~ /\s/) { print "Coincide\n"; }
  ```

### Modificadores comunes en Perl

- **`i`**: Hace la búsqueda **insensible a mayúsculas y minúsculas**.
  ```perl
  if ("Hola Mundo" =~ /hola/i) { print "Coincide\n"; }
  ```

- **`g`**: Aplica la expresión regular de forma **global**, es decir, busca todas las coincidencias en lugar de detenerse en la primera.
  ```perl
  $texto = "uno dos tres";
  $texto =~ s/\s/_/g;  # Reemplaza todos los espacios con "_"
  print $texto;  # uno_dos_tres
  ```

### Clases de caracteres

- **`[ ]`**: Define un **conjunto de caracteres**. Coincide con cualquier carácter que esté dentro de los corchetes.
  ```perl
  if ("hola" =~ /[aeiou]/) { print "Contiene una vocal\n"; }
  ```

- **`[^ ]`**: El **acento circunflejo** dentro de un conjunto niega el conjunto, es decir, coincide con cualquier carácter que **no** esté en los corchetes.
  ```perl
  if ("123" =~ /[^0-9]/) { print "No contiene solo números\n"; }
  ```

### Repeticiones

- **`*`**: Coincide con **cero o más** repeticiones del patrón anterior.
  ```perl
  if ("abc" =~ /a.*/ ) { print "Coincide\n"; }  # Coincide con "a" seguido de cualquier número de caracteres.
  ```

- **`+`**: Coincide con **uno o más** repeticiones del patrón anterior.
  ```perl
  if ("abc" =~ /a.+/) { print "Coincide\n"; }
  ```

- **`?`**: Coincide con **cero o una** repetición del patrón anterior.
  ```perl
  if ("color" =~ /colou?r/) { print "Coincide\n"; }
  # Coincide tanto con "color" como con "colour".
  ```

- **`{n,m}`**: Coincide con **entre n y m** repeticiones del patrón anterior.
  ```perl
  if ("aaa" =~ /a{2,3}/) { print "Coincide\n"; }
  # Coincide con "aa" o "aaa".
  ```

### Grupos y captura

- **Paréntesis `()`**: Se utilizan para agrupar partes de la expresión regular, permitiendo aplicar operadores a todo el grupo o capturar coincidencias para reutilizarlas.
  ```perl
  if ("2023-09-21" =~ /(\d{4})-(\d{2})-(\d{2})/) {
      print "Año: $1, Mes: $2, Día: $3\n";
  }
  # Imprime: Año: 2023, Mes: 09, Día: 21
  ```

### Reemplazo con regex

Para reemplazar un patrón utilizando expresiones regulares, se usa la función `s///`.

```perl
$cadena = "Hola Mundo";
$cadena =~ s/Mundo/Perl/;
print $cadena;  # Hola Perl
```

Si quieres reemplazar todas las coincidencias, usa el modificador `g` (global):
```perl
$cadena = "Mundo Mundo";
$cadena =~ s/Mundo/Perl/g;
print $cadena;  # Perl Perl
```

### Ejemplos de uso de expresiones regulares en Perl

1. **Validar un número de teléfono**:
   ```perl
   $telefono = "555-123-4567";
   if ($telefono =~ /^\d{3}-\d{3}-\d{4}$/) {
       print "Número de teléfono válido\n";
   } else {
       print "Número de teléfono inválido\n";
   }
   ```

2. **Validar una dirección de correo electrónico**:
   ```perl
   $email = "usuario@ejemplo.com";
   if ($email =~ /^[\w\.-]+@[a-zA-Z\d-]+\.[a-zA-Z]{2,6}$/) {
       print "Correo electrónico válido\n";
   }
   ```

3. **Extraer URLs de un texto**:
   ```perl
   $texto = "Visita http://www.ejemplo.com y https://www.perl.org";
   while ($texto =~ /(https?:\/\/[^\s]+)/g) {
       print "URL encontrada: $1\n";
   }
   ```

### Conclusión

Las expresiones regulares en Perl son extremadamente poderosas para manipular y analizar texto. Su sintaxis es flexible, y ofrece herramientas avanzadas para búsquedas, reemplazos y validación de patrones. Debido a su soporte nativo, Perl es una excelente opción cuando se requiere trabajar intensivamente con texto y patrones complejos.

[The Perl Programming Language - www.perl.org](https://www.perl.org/)

## PHP

Las **expresiones regulares en PHP** son una poderosa herramienta para buscar, validar y manipular texto mediante patrones. PHP admite dos tipos de sintaxis para las expresiones regulares:

1. **PCRE (Perl Compatible Regular Expressions)**: Usan funciones como `preg_match`, `preg_replace`, etc.
2. **Posix**: Aunque es más antigua y menos común, generalmente se recomienda usar PCRE por ser más potente.

### Funciones clave para trabajar con expresiones regulares en PHP:

#### 1. **`preg_match`**
Busca un patrón dentro de una cadena.
```php
<?php
$cadena = "Mi número es 12345";
$patron = "/\d+/";  // Busca un número en la cadena

if (preg_match($patron, $cadena, $coincidencias)) {
    echo "Se encontró un número: " . $coincidencias[0];
} else {
    echo "No se encontró un número.";
}
?>
```

#### 2. **`preg_match_all`**
Busca todas las coincidencias de un patrón.
```php
<?php
$cadena = "Mis números son 123 y 456";
$patron = "/\d+/";  // Busca todos los números en la cadena

preg_match_all($patron, $cadena, $coincidencias);

print_r($coincidencias);  // Mostrará todos los números encontrados
?>
```

#### 3. **`preg_replace`**
Reemplaza partes de una cadena que coinciden con un patrón.
```php
<?php
$cadena = "Tengo 2 perros y 3 gatos";
$patron = "/\d+/";  // Busca todos los números
$reemplazo = "#";

$nuevaCadena = preg_replace($patron, $reemplazo, $cadena);
echo $nuevaCadena;  // Resultado: Tengo # perros y # gatos
?>
```

#### 4. **`preg_split`**
Divide una cadena en un arreglo utilizando una expresión regular como delimitador.
```php
<?php
$cadena = "uno, dos, tres, cuatro";
$patron = "/,\s+/";  // Divide por comas y espacios

$resultado = preg_split($patron, $cadena);
print_r($resultado);  // Resultado: Array ( [0] => uno [1] => dos [2] => tres [3] => cuatro )
?>
```

### Ejemplos de patrones comunes en PHP:

#### 1. **Validar un correo electrónico:**
```php
<?php
$email = "ejemplo@dominio.com";
$patron = "/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/";

if (preg_match($patron, $email)) {
    echo "Correo válido";
} else {
    echo "Correo inválido";
}
?>
```

#### 2. **Validar una URL:**
```php
<?php
$url = "https://www.ejemplo.com";
$patron = "/\bhttps?:\/\/[a-z0-9.-]+\.[a-z]{2,6}\b/";

if (preg_match($patron, $url)) {
    echo "URL válida";
} else {
    echo "URL inválida";
}
?>
```

#### 3. **Validar un número de teléfono:**
```php
<?php
$telefono = "123-456-7890";
$patron = "/^\d{3}-\d{3}-\d{4}$/";

if (preg_match($patron, $telefono)) {
    echo "Teléfono válido";
} else {
    echo "Teléfono inválido";
}
?>
```

### Delimitadores y modificadores:
- **Delimitadores**: Las expresiones regulares en PHP se delimitan con caracteres como `/`. Por ejemplo, `/\d+/` busca dígitos.
- **Modificadores**:
  - `i`: Ignora mayúsculas/minúsculas.
  - `m`: Trata la cadena como multilínea.
  - `s`: Permite que el `.` coincida con saltos de línea.

**ver los visitantes ganadores**

```php
<?php

$file = fopen("results.csv", "r");

$match = 0;
$nomatch = 0;

while(!feof($file)) {
    $line = fgets($file);
    if (preg_match(
        '/^2018\-01\-(\d\d),.*$/',
        $line,
        $m
        )
    ) {
        print_r($m);
        $match++;

    }
    else {
        $nomatch++;
    }
}

fclose($file);

printf("\n\nmatch: %d\nno match: %d\n", $match, $nomatch);
```

### Conclusión
Las expresiones regulares en PHP son extremadamente útiles para validaciones y manipulaciones de texto. Utilizando funciones como `preg_match`, `preg_replace`, y `preg_split`, puedes implementar filtros de datos complejos y realizar búsquedas avanzadas en tus aplicaciones.

**Lecturas recomendadas**

[XAMPP Installers and Downloads for Apache Friends](https://www.apachefriends.org/es/index.html)

## Python

Las **expresiones regulares** en Python son una herramienta muy potente para trabajar con patrones en cadenas de texto. Se utilizan para buscar, validar, extraer o manipular partes de texto de una forma eficiente y flexible.

En Python, el módulo **`re`** proporciona todas las funciones necesarias para trabajar con expresiones regulares.

### Funciones principales de `re`

#### 1. **`re.match()`**
Busca el patrón solo al comienzo de la cadena.
```python
import re

patron = r"\d{3}"  # Busca un número de 3 dígitos
cadena = "123 hola mundo"

if re.match(patron, cadena):
    print("Coincidencia al inicio de la cadena")
else:
    print("No hay coincidencia al inicio")
```

#### 2. **`re.search()`**
Busca el patrón en cualquier parte de la cadena.
```python
import re

patron = r"\d{3}"  # Busca un número de 3 dígitos
cadena = "Hola, tengo 123 perros"

resultado = re.search(patron, cadena)
if resultado:
    print(f"Se encontró: {resultado.group()}")
else:
    print("No se encontró coincidencia")
```

#### 3. **`re.findall()`**
Encuentra todas las coincidencias de un patrón en la cadena y las devuelve en una lista.
```python
import re

patron = r"\d+"  # Busca todos los números
cadena = "Tengo 2 perros, 3 gatos y 4 ratones"

resultados = re.findall(patron, cadena)
print(resultados)  # ['2', '3', '4']
```

#### 4. **`re.sub()`**
Reemplaza las coincidencias de un patrón por otro valor.
```python
import re

patron = r"\d+"  # Busca todos los números
cadena = "Tengo 2 perros y 3 gatos"

nueva_cadena = re.sub(patron, "#", cadena)
print(nueva_cadena)  # Tengo # perros y # gatos
```

#### 5. **`re.split()`**
Divide una cadena utilizando un patrón como delimitador.
```python
import re

patron = r"\s+"  # Usa los espacios como delimitador
cadena = "Hola mundo, ¿cómo estás?"

resultado = re.split(patron, cadena)
print(resultado)  # ['Hola', 'mundo,', '¿cómo', 'estás?']
```

### Ejemplos de expresiones regulares comunes

#### 1. **Validar un correo electrónico**
```python
import re

patron = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
correo = "ejemplo@correo.com"

if re.match(patron, correo):
    print("Correo válido")
else:
    print("Correo inválido")
```

#### 2. **Validar un número de teléfono**
```python
import re

patron = r"^\d{3}-\d{3}-\d{4}$"  # Formato: 123-456-7890
telefono = "123-456-7890"

if re.match(patron, telefono):
    print("Número válido")
else:
    print("Número inválido")
```

#### 3. **Validar una URL**
```python
import re

patron = r"https?://(www\.)?[a-zA-Z0-9-]+(\.[a-zA-Z]{2,6})+"
url = "https://www.ejemplo.com"

if re.match(patron, url):
    print("URL válida")
else:
    print("URL inválida")
```

### Caracteres especiales en expresiones regulares

- `.`: Coincide con cualquier carácter excepto saltos de línea.
- `\d`: Coincide con cualquier dígito.
- `\D`: Coincide con cualquier carácter que **no** sea un dígito.
- `\w`: Coincide con cualquier carácter alfanumérico (letras, números y guiones bajos).
- `\W`: Coincide con cualquier carácter que **no** sea alfanumérico.
- `\s`: Coincide con cualquier espacio en blanco (espacios, tabulaciones, saltos de línea).
- `\S`: Coincide con cualquier carácter que **no** sea un espacio en blanco.

### Delimitadores y modificadores

- `^`: Inicio de la cadena.
- `$`: Fin de la cadena.
- `*`: Cero o más repeticiones del carácter anterior.
- `+`: Una o más repeticiones del carácter anterior.
- `?`: Cero o una repetición del carácter anterior.
- `{m,n}`: De `m` a `n` repeticiones del carácter anterior.

### Ejemplo avanzado: Extraer parámetros de una URL
```python
import re

url = "https://www.ejemplo.com?param1=valor1&param2=valor2&param3=valor3"
patron = r"(\w+)=([\w\d]+)"

resultados = re.findall(patron, url)
print(resultados)  # [('param1', 'valor1'), ('param2', 'valor2'), ('param3', 'valor3')]
```

Con las expresiones regulares, puedes realizar operaciones complejas en cadenas de texto de una manera eficiente y concisa. ¡Son extremadamente útiles en muchos casos!

**Lecturas recomendadas**

[Welcome to Python.org](https://www.python.org/)

## Java

En Java, las expresiones regulares se utilizan principalmente con las clases **`Pattern`** y **`Matcher`**, que pertenecen al paquete `java.util.regex`. Las expresiones regulares permiten realizar búsquedas y manipulaciones complejas de cadenas de texto, como coincidencias de patrones, reemplazo de texto y validación de formatos.

### Clases clave:
- **`Pattern`**: Representa un patrón de expresión regular compilado.
- **`Matcher`**: Se utiliza para realizar las búsquedas y obtener coincidencias del patrón en una cadena de texto.

### Ejemplo básico de uso de expresiones regulares en Java:

```java
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class RegexExample {
    public static void main(String[] args) {
        // Definir el patrón (expresión regular)
        String regex = "\\d{3}-\\d{2}-\\d{4}"; // Busca coincidencias en formato "###-##-####" (como un número de seguro social)
        
        // Compilar el patrón
        Pattern pattern = Pattern.compile(regex);
        
        // Cadena de ejemplo donde buscar el patrón
        String text = "Mi número de seguro social es 123-45-6789.";
        
        // Crear el Matcher para buscar coincidencias
        Matcher matcher = pattern.matcher(text);
        
        // Comprobar si hay coincidencias
        if (matcher.find()) {
            System.out.println("Se encontró una coincidencia: " + matcher.group());
        } else {
            System.out.println("No se encontró ninguna coincidencia.");
        }
    }
}
```

### Explicación del ejemplo:
1. **`String regex = "\\d{3}-\\d{2}-\\d{4}"`**:
   - Esta expresión regular busca una secuencia de 3 dígitos, seguida de un guion, 2 dígitos más, otro guion y finalmente 4 dígitos.
   - El patrón se utiliza para identificar números de formato como un número de seguro social en EE. UU. (###-##-####).

2. **`Pattern.compile(regex)`**: 
   - Compila la expresión regular para que pueda ser utilizada por un objeto `Matcher`.

3. **`Matcher matcher = pattern.matcher(text)`**:
   - Crea un `Matcher` que buscará el patrón dentro del texto.

4. **`matcher.find()`**: 
   - Busca coincidencias del patrón en el texto.
   
5. **`matcher.group()`**: 
   - Devuelve la coincidencia encontrada.

### Métodos comunes en `Pattern` y `Matcher`:
- **`Pattern.compile(String regex)`**: Compila una expresión regular en un patrón.
- **`matcher.matches()`**: Devuelve `true` si la cadena completa coincide con la expresión regular.
- **`matcher.find()`**: Busca la próxima coincidencia de la expresión regular en el texto.
- **`matcher.group()`**: Devuelve la subsecuencia del texto que coincide con la expresión regular.
- **`matcher.replaceAll(String replacement)`**: Reemplaza todas las coincidencias de la expresión regular en la cadena original con una nueva cadena.

### Validar una dirección de correo electrónico:
```java
import java.util.regex.Pattern;

public class EmailValidator {
    public static void main(String[] args) {
        String email = "ejemplo@test.com";
        String regex = "^[\\w-\\.]+@[\\w-]+\\.[a-z]{2,}$";

        boolean isValid = Pattern.matches(regex, email);
        if (isValid) {
            System.out.println(email + " es una dirección de correo válida.");
        } else {
            System.out.println(email + " no es válida.");
        }
    }
}
```

### Aplicaciones de expresiones regulares en Java:
- **Validación**: Números de teléfono, correos electrónicos, direcciones IP.
- **Buscar y reemplazar**: Modificar formatos de texto, como fechas o direcciones.
- **Extracción**: Extraer partes de una cadena que coincidan con un patrón.
- **Transformaciones**: Convertir texto con formatos específicos a otros formatos.

Java proporciona un potente motor de expresiones regulares que es eficiente y flexible para manejar patrones complejos de texto.

**Nota**: para iniciar el archivo java por teminal se utiliza el siguiente codigo `javac <archivo.java>` o compilarod en java 11 `javac --release 11 regex.java` para compilar y iniciarlo el archivo se usa `java <archivo>`

**Lecturas recomendadas**

[Descarga gratuita de software de Java](https://www.java.com/es/download/)

## Java aplicado

Java es un lenguaje de programación de propósito general utilizado en una variedad de aplicaciones debido a su portabilidad, escalabilidad y robustez. Aquí te muestro algunas de las áreas y aplicaciones más comunes donde Java se utiliza ampliamente:

### 1. **Desarrollo Web**
Java es una de las tecnologías más populares para el desarrollo de aplicaciones web empresariales y sitios web dinámicos, principalmente utilizando **Java EE** (Enterprise Edition) o **Spring Framework**.
- **Servlets** y **JSP (JavaServer Pages)** son tecnologías comunes para crear aplicaciones web dinámicas.
- **Spring Boot** se utiliza para construir aplicaciones web modernas y microservicios.
  
**Ejemplo de una simple aplicación web usando Servlets:**

```java
import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;

public class HelloWorldServlet extends HttpServlet {
    public void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        response.setContentType("text/html");
        PrintWriter out = response.getWriter();
        out.println("<h1>Hello, World</h1>");
    }
}
```

### 2. **Aplicaciones Empresariales**
Java es un lenguaje muy usado en el desarrollo de **aplicaciones empresariales a gran escala**, como **sistemas de gestión empresarial (ERP)**, **gestión de la cadena de suministro (SCM)**, o **bancos**. Las soluciones como **Spring**, **EJB (Enterprise Java Beans)** y **JPA (Java Persistence API)** son populares.

### 3. **Desarrollo de Aplicaciones Móviles**
El **desarrollo de aplicaciones móviles Android** está profundamente relacionado con Java. Aunque **Kotlin** ha ganado popularidad, **Java** sigue siendo un lenguaje base para las aplicaciones de Android.

```java
public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }
}
```

### 4. **Aplicaciones de Escritorio**
Java proporciona **Swing** y **JavaFX** para el desarrollo de interfaces gráficas de usuario (GUI), lo que permite crear aplicaciones de escritorio multiplataforma.

**Ejemplo básico de una ventana GUI usando Swing:**

```java
import javax.swing.*;

public class HelloWorldSwing {
    public static void main(String[] args) {
        JFrame frame = new JFrame("Hello World");
        JLabel label = new JLabel("Hello, World!", JLabel.CENTER);
        frame.add(label);
        frame.setSize(300, 100);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}
```

### 5. **Sistemas Distribuidos y Microservicios**
Java es utilizado en aplicaciones que manejan arquitecturas distribuidas, como los **microservicios**. El framework **Spring Boot** junto con **Spring Cloud** es muy popular para desarrollar microservicios. 

### 6. **Aplicaciones Financieras**
Muchos **sistemas financieros** y aplicaciones bancarias están construidos con Java por su fiabilidad y rendimiento. Herramientas como **Apache Kafka** y **Apache Camel** son utilizadas para manejar grandes volúmenes de transacciones.

### 7. **Big Data**
Java se usa en herramientas y plataformas **Big Data**, como **Apache Hadoop** y **Apache Spark**, que permiten procesar grandes cantidades de datos distribuidos en clústeres.

### 8. **Juegos**
Java también se usa para desarrollar **juegos**, aunque no es el lenguaje principal en la industria de los videojuegos. Sin embargo, juegos populares como **Minecraft** fueron creados originalmente en Java.

### 9. **Sistemas Embebidos**
Java también está presente en dispositivos de **Internet of Things (IoT)** y sistemas embebidos, donde **Java ME (Micro Edition)** es utilizado.

### 10. **Cloud Computing**
Con la popularidad de las plataformas **cloud**, Java se utiliza en el desarrollo de aplicaciones que se ejecutan en la nube, aprovechando plataformas como **AWS**, **Google Cloud** o **Microsoft Azure**.

---

### Beneficios del uso de Java:
1. **Multiplataforma**: El lema "write once, run anywhere" se cumple gracias a la **Java Virtual Machine (JVM)**.
2. **Ecosistema y Comunidad**: Tiene un amplio ecosistema de librerías, frameworks y herramientas, junto con una activa comunidad.
3. **Seguridad**: Java ofrece características de seguridad integradas, lo que lo hace ideal para aplicaciones bancarias y empresariales.
4. **Rendimiento**: Aunque no es tan rápido como el código nativo, el rendimiento de Java es generalmente lo suficientemente alto para la mayoría de las aplicaciones empresariales y web.

En resumen, Java es muy versátil y se adapta a casi cualquier tipo de proyecto, desde pequeñas aplicaciones móviles hasta grandes sistemas distribuidos.

**Nota**: para ver cuantos filas tiene el archivo `java regex | wc -l`.

**Lecturas recomendadas**

[Java SE - Downloads | Oracle Technology Network | Oracle](https://www.oracle.com/technetwork/java/javase/downloads/index.html)
[Configurar la variable de entorno PATH para Java | Tutorial de Java | Abrirllave.com](https://www.abrirllave.com/java/configurar-la-variable-de-entorno-path.php)

## JavaScript

En **JavaScript**, las **expresiones regulares** (o **regex**) son patrones utilizados para hacer coincidir combinaciones de caracteres en cadenas. Estas expresiones son muy útiles para validar, buscar, o reemplazar patrones específicos de texto dentro de una cadena. JavaScript proporciona un soporte robusto para trabajar con expresiones regulares a través de su clase nativa `RegExp` y algunos métodos de las cadenas (`String`).

### Creación de una expresión regular

En JavaScript, las expresiones regulares se pueden crear de dos maneras:

1. **Usando una notación literal**: 
   ```javascript
   const regex = /patron/;
   ```

2. **Usando el constructor `RegExp`**:
   ```javascript
   const regex = new RegExp("patron");
   ```

### Métodos de `RegExp` y `String` asociados a expresiones regulares

1. **`test()`**: Verifica si una cadena coincide con el patrón de la expresión regular y devuelve `true` o `false`.
   ```javascript
   const regex = /abc/;
   const cadena = "abcde";
   console.log(regex.test(cadena)); // true
   ```

2. **`match()`**: Devuelve las coincidencias encontradas en una cadena. Si no se encuentran coincidencias, devuelve `null`.
   ```javascript
   const cadena = "2023-09-26";
   const resultado = cadena.match(/\d{4}/); // Busca un grupo de 4 dígitos
   console.log(resultado); // ["2023"]
   ```

3. **`replace()`**: Reemplaza coincidencias en una cadena.
   ```javascript
   const cadena = "Estoy aprendiendo JavaScript";
   const resultado = cadena.replace(/JavaScript/, "regex");
   console.log(resultado); // "Estoy aprendiendo regex"
   ```

4. **`split()`**: Divide una cadena en un arreglo según el patrón de la expresión regular.
   ```javascript
   const cadena = "uno, dos, tres";
   const partes = cadena.split(/,\s*/); // Divide por comas y espacios
   console.log(partes); // ["uno", "dos", "tres"]
   ```

### Caracteres y Metacaracteres Comunes en Expresiones Regulares

- **`.`**: Representa cualquier carácter excepto un salto de línea.
  ```javascript
  const regex = /.a./;
  const cadena = "car";
  console.log(regex.test(cadena)); // true ("car" coincide)
  ```

- **`^`**: Indica el **inicio** de una cadena.
  ```javascript
  const regex = /^Hola/;
  const cadena = "Hola mundo";
  console.log(regex.test(cadena)); // true
  ```

- **`$`**: Indica el **final** de una cadena.
  ```javascript
  const regex = /mundo$/;
  const cadena = "Hola mundo";
  console.log(regex.test(cadena)); // true
  ```

- **`*`**: Coincide con **cero o más** repeticiones del carácter anterior.
  ```javascript
  const regex = /ho*/;
  console.log(regex.test("hoooola")); // true
  ```

- **`+`**: Coincide con **una o más** repeticiones del carácter anterior.
  ```javascript
  const regex = /ho+/;
  console.log(regex.test("hola")); // true
  console.log(regex.test("h")); // false
  ```

- **`?`**: Coincide con **cero o una** ocurrencia del carácter anterior (opcional).
  ```javascript
  const regex = /colou?r/;
  console.log(regex.test("color")); // true
  console.log(regex.test("colour")); // true
  ```

- **`{n}`**: Coincide con **exactamente n** ocurrencias del carácter anterior.
  ```javascript
  const regex = /\d{4}/;
  const cadena = "El año es 2024";
  console.log(regex.test(cadena)); // true (coincide con 2024)
  ```

### Clases de Caracteres Predefinidas

- **`\d`**: Coincide con cualquier **dígito** (equivalente a `[0-9]`).
- **`\D`**: Coincide con cualquier carácter que **no sea un dígito**.
- **`\w`**: Coincide con cualquier carácter alfanumérico (letras y números, incluyendo el guion bajo).
- **`\W`**: Coincide con cualquier carácter que **no sea alfanumérico**.
- **`\s`**: Coincide con cualquier **espacio en blanco** (incluye tabulaciones y saltos de línea).
- **`\S`**: Coincide con cualquier carácter que **no sea un espacio en blanco**.

### Ejemplos de Aplicaciones de Expresiones Regulares en JavaScript

1. **Validar un correo electrónico**:
   ```javascript
   const emailRegex = /^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$/;
   console.log(emailRegex.test("example@domain.com")); // true
   ```

2. **Validar un número de teléfono** (formato internacional):
   ```javascript
   const phoneRegex = /^\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}$/;
   console.log(phoneRegex.test("+1 (555) 555-5555")); // true
   ```

3. **Buscar URLs en un texto**:
   ```javascript
   const urlRegex = /https?:\/\/(www\.)?[\w-]+\.\w{2,}(\/\S*)?/gi;
   const texto = "Visita https://example.com o http://another-site.org para más información.";
   console.log(texto.match(urlRegex)); // ["https://example.com", "http://another-site.org"]
   ```

4. **Validar una fecha en formato YYYY-MM-DD**:
   ```javascript
   const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
   console.log(dateRegex.test("2024-09-26")); // true
   ```

### Conclusión

Las expresiones regulares en JavaScript son una herramienta muy poderosa para buscar, validar y manipular texto. Al usarlas correctamente, puedes simplificar tareas complejas como la validación de formularios, la extracción de datos de texto, o la transformación de cadenas.

## `grep` y `find` desde consola

Las **expresiones regulares** pueden usarse en la línea de comandos de Unix/Linux para buscar patrones de texto mediante herramientas como **`grep`** y **`find`**. Estas herramientas son muy útiles para procesar y buscar información dentro de archivos o directorios. A continuación te explico cómo utilizar expresiones regulares con ambas.

## Uso de `grep` con expresiones regulares

El comando `grep` busca patrones en archivos o en la salida de otros comandos. Soporta expresiones regulares básicas y extendidas.

### Sintaxis básica
```bash
grep [opciones] "expresión_regular" archivo
```

### Ejemplos con `grep`:

1. **Buscar una palabra en un archivo**:
   ```bash
   grep "patrón" archivo.txt
   ```

2. **Buscar con expresiones regulares básicas**:
   Supongamos que queremos buscar todas las líneas que comiencen con un número de cuatro dígitos en el archivo `log.txt`:
   ```bash
   grep "^[0-9]\{4\}" log.txt
   ```

3. **Buscar de forma insensible a mayúsculas y minúsculas**:
   ```bash
   grep -i "patrón" archivo.txt
   ```

4. **Buscar con expresiones regulares extendidas**:
   Usamos `-E` para habilitar expresiones regulares extendidas (equivalente a `egrep`):
   ```bash
   grep -E "(error|warning)" archivo.log
   ```

5. **Buscar recursivamente en todos los archivos de un directorio**:
   ```bash
   grep -r "patrón" /ruta/del/directorio
   ```

6. **Mostrar solo el número de línea donde se encuentra el patrón**:
   ```bash
   grep -n "patrón" archivo.txt
   ```

7. **Buscar líneas que no coinciden con un patrón** (inverso):
   ```bash
   grep -v "patrón" archivo.txt
   ```

### Combinaciones útiles con `grep`:

- **Buscar una palabra exacta**: El uso de la opción `-w` asegura que `grep` busque solo la palabra completa.
  ```bash
  grep -w "error" archivo.txt
  ```

- **Contar las ocurrencias** de un patrón en un archivo con `-c`:
  ```bash
  grep -c "patrón" archivo.txt
  ```

## Uso de `find` con expresiones regulares

El comando `find` se utiliza para buscar archivos y directorios en un sistema de archivos, y también puede usar expresiones regulares para encontrar archivos con nombres que sigan ciertos patrones.

### Sintaxis básica de `find`
```bash
find [ruta] [opciones] [expresión]
```

### Ejemplos con `find`:

1. **Buscar archivos por nombre** (exactamente):
   ```bash
   find /ruta -name "archivo.txt"
   ```

2. **Buscar archivos con expresiones regulares**:
   Usamos la opción `-regex` para indicar que queremos utilizar una expresión regular.
   ```bash
   find /ruta -regex ".*\.txt$"
   ```

   Este ejemplo busca todos los archivos que terminen en `.txt`.

3. **Buscar archivos que coincidan con un patrón en su nombre**:
   Para buscar archivos que comiencen con "log" y terminen con números, podemos usar:
   ```bash
   find /ruta -regex ".*/log[0-9]+"
   ```

4. **Buscar archivos por extensión** (p. ej., archivos `.log` o `.txt`):
   ```bash
   find /ruta -regex ".*\.\(log\|txt\)$"
   ```

5. **Buscar archivos modificados en los últimos N días**:
   ```bash
   find /ruta -name "*.log" -mtime -7
   ```

6. **Buscar archivos y directorios vacíos**:
   ```bash
   find /ruta -empty
   ```

7. **Buscar archivos grandes (más de 100 MB)**:
   ```bash
   find /ruta -size +100M
   ```

### Combinación de `find` con `grep`

Podemos combinar `find` y `grep` para buscar dentro de los archivos encontrados. Por ejemplo, para buscar la palabra "error" en todos los archivos `.log` en un directorio:
```bash
find /ruta -name "*.log" -exec grep "error" {} +
```

## Resumen de opciones útiles

### `grep`:

- `-i`: Ignorar mayúsculas/minúsculas.
- `-r`: Buscar recursivamente.
- `-n`: Mostrar número de línea.
- `-v`: Buscar líneas que **no** coinciden.
- `-E`: Usar expresiones regulares extendidas.

### `find`:

- `-name`: Buscar por nombre de archivo.
- `-regex`: Usar una expresión regular para el nombre.
- `-mtime`: Buscar archivos modificados recientemente.
- `-size`: Buscar por tamaño de archivo.
- `-exec`: Ejecutar un comando para cada archivo encontrado.

Estas combinaciones te permiten realizar búsquedas potentes y filtrados de texto dentro de archivos y sistemas de archivos de manera eficiente en un entorno de línea de comandos.

En los sistemas operativos basados en UNIX podemos utilizar expresiones regulares a través de la consola mediante los comandos grep y find.

- **grep**: Nos ayuda a buscar dentro de los archivos, textos muy puntuales utilizando una versión muy reducida de expresiones regulares.

- **find**: Se utiliza para encontrar archivos en un determinado directorio a partir de diversas reglas de búsqueda que incluyen expresiones regulares.

**Archivos de la clase**

[cheatsheet-1.jpg](images/cheatsheet-1.jpg)

[cheatsheet-2.jpg](images/cheatsheet-2.jpg)

[cheatsheet-3.jpg](images/cheatsheet-3.jpg)

`cat results.csv | grep ^2012`: muetra todos los datos del 2012.
`cat results.csv | grep ,3[0-9],`: Muestra los goles de 31 a 39.
`cat results.csv | grep SE$`: Muestra todos los FALSE.
`cat results.csv | grep SE`:  trae todos los elementos que contengan SE.
`cat results.csv | grep [SE]`: trae tolos los que tengan SE.
` cat results.csv | grep Brazil`: busca todos los partidos de brasil.
`cat results.csv | grep Brazil | grep Uruguay`: Busca todos los partidos de brasil contra Uruguay.
`cat results.csv | grep Brazil | grep Uruguay | grep ^1952`: Busca todos los partidos de brasil contra Uruguay en 1952.