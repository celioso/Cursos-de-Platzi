# Curso de Unit Testing en Python

## ¿Qué son las Pruebas Unitarias y por qué es importante?

Probar software no solo es una tarea técnica, es un proceso crítico que puede marcar la diferencia entre el éxito o el fracaso de un proyecto. Un pequeño error no detectado puede causar grandes problemas, como lo demuestra el caso del cohete de la Agencia Espacial Europea en 1996. Afortunadamente, en el desarrollo de software contamos con herramientas como Python y sus módulos para asegurar la calidad del código antes de que llegue a los usuarios.

### ¿Qué tipos de pruebas son necesarias para asegurar la calidad del software?

- **Pruebas unitarias**: Se encargan de validar que cada componente pequeño del código funcione correctamente de manera aislada.
- **Pruebas de integración**: Verifican que los distintos componentes trabajen bien en conjunto, evitando problemas en la interacción de partes.
- **Pruebas funcionales**: Validan que el sistema en su totalidad funcione como se espera según los requisitos.
- **Pruebas de rendimiento**: Aseguran que el software sea rápido y eficiente, evaluando su comportamiento bajo diferentes condiciones de carga.
- **Pruebas de aceptación**: Determinan si el software cumple con las expectativas del usuario final.

### ¿Qué herramientas de testing ofrece Python?

- **UnitTest**: Permite crear pruebas unitarias de manera sencilla, asegurando que todas las partes del código realicen su función correctamente.
- **PyTest**: Facilita la creación de pruebas con una configuración avanzada para cubrir diferentes escenarios.
- **DocTest**: Integra pruebas directamente en los comentarios de las funciones, permitiendo validar el código mientras se mantiene la documentación.

### ¿Cómo garantizar que todas las líneas de código están siendo probadas?

Es crucial identificar las líneas de código que no están cubiertas por pruebas. Para esto, existe **Coverage**, una herramienta que genera un reporte en HTML mostrando qué partes del código no han sido validadas, lo que permite agregar pruebas adicionales donde sea necesario.

### ¿Por qué es importante el testing en software?

El testing asegura que el software sea funcional, rápido y confiable, pero más allá de eso, puede evitar costosos errores, pérdidas financieras y en casos extremos, salvar vidas. Al probar el software antes de que llegue a producción, los desarrolladores tienen la ventaja de corregir fallos antes de que impacten a los usuarios.

**Lecturas recomendadas**

[unittest — Unit testing framework — Python 3.12.5 documentation](https://docs.python.org/3/library/unittest.html)

## ¿Qué es el Testing en Software?

Las pruebas en el desarrollo de software son esenciales para garantizar la calidad y estabilidad del código antes de lanzarlo a producción. Tanto las pruebas manuales como las automatizadas juegan un rol fundamental para detectar errores. Usar Python para automatizar estas pruebas no solo ahorra tiempo, sino que también asegura que los errores críticos se detecten antes, evitando posibles pérdidas económicas y de confianza de los usuarios.

### ¿Qué son las pruebas manuales y cómo funcionan?

Las pruebas manuales consisten en validar el funcionamiento de un cambio en el código mediante la interacción directa con la aplicación. Esto se hace, por ejemplo, al modificar una línea de código, ejecutar la aplicación y verificar si el cambio funciona correctamente. Es similar al trabajo de un mecánico que ajusta un auto y luego lo prueba encendiéndolo cada vez.

### ¿Cómo funcionan las pruebas unitarias?

Las pruebas unitarias permiten validar que pequeñas piezas de código, como funciones individuales, trabajan correctamente. En el ejemplo de un mecánico, esto sería como revisar solo un neumático: inflarlo, verificar que no tenga fugas y confirmar que esté en buen estado. En Python, estas pruebas se automatizan utilizando la palabra clave **assert**, que compara los resultados esperados con los reales.

### ¿Qué son las pruebas de integración?

Las pruebas de integración verifican que diferentes componentes de la aplicación funcionen en conjunto sin problemas. En el caso del mecánico, sería comprobar que el neumático instalado en el coche funcione bien con el resto de las piezas del vehículo. En desarrollo de software, esto se traduce a verificar, por ejemplo, que el proceso de inicio de sesión funcione correctamente, desde la entrada del usuario hasta la confirmación del acceso.

### ¿Cómo Python nos ayuda a automatizar pruebas?

Python ofrece herramientas para automatizar las pruebas, permitiendo ejecutar muchas validaciones rápidamente sin intervención manual. A través de pruebas automatizadas, podemos detectar errores que de otro modo podrían pasar desapercibidos y llegar a producción, como un fallo en el cálculo de una orden de compra. Esto es crítico para evitar situaciones como la que enfrentó CrowdStrike, donde un error no detectado en una actualización paralizó aeropuertos.

El **Testing en Software** es el proceso de evaluar y verificar que un programa o aplicación funcione de acuerdo con los requisitos especificados y no tenga errores. El objetivo principal del testing es identificar defectos o problemas en el software antes de que llegue al usuario final, garantizando su calidad, funcionalidad, seguridad y rendimiento.

### Tipos de Testing:
1. **Testing Unitario**: Verifica el correcto funcionamiento de unidades individuales de código, como funciones o métodos.
2. **Testing de Integración**: Asegura que los diferentes módulos o componentes funcionen bien en conjunto.
3. **Testing de Sistema**: Evalúa el sistema completo para asegurarse de que cumple con los requisitos especificados.
4. **Testing de Aceptación**: Verifica que el software satisface las necesidades del usuario final.
5. **Testing de Regresión**: Asegura que los cambios o actualizaciones no hayan introducido nuevos errores en funcionalidades ya existentes.

### Métodos de Testing:
- **Manual**: El tester ejecuta las pruebas sin automatización.
- **Automatizado**: Se usan herramientas o scripts para ejecutar las pruebas de manera automática.

Testing es crucial para asegurar la calidad del software y minimizar el riesgo de errores en producción.

**Lecturas recomendadas**

[unittest — Unit testing framework — Python 3.12.5 documentation](https://docs.python.org/3/library/unittest.html)
[- YouTube](https://www.youtube.com/watch?v=cSn7Ut4lysY)

## Instalación y Configuración del Entorno de Pruebas

La creación de funciones y pruebas para el código que se va a producción es clave para validar resultados correctamente. En Python, el uso de Unit Testing simplifica este proceso, permitiendo automatizar pruebas y hacerlas más legibles y eficientes, además de integrarse fácilmente con sistemas de Continuous Integration.

### ¿Cómo mejorar la legibilidad de las pruebas con Unit Testing?

Python incluye Unit Testing de forma nativa, proporcionando clases reutilizables para ejecutar pruebas de manera automática o manual. Esta herramienta no solo permite mejorar la legibilidad, sino también identificar y solucionar errores rápidamente, sin necesidad de depender de print para verificar si las pruebas se están ejecutando.

### ¿Cómo estructurar un proyecto de testing en Python?

1. **Separación de código y pruebas**: Coloca el código fuente en una carpeta src y las pruebas en una carpeta test.
2. **Entorno virtual**: Crea un entorno virtual para aislar dependencias del proyecto. Esto se hace con python -m venv, lo que genera una carpeta con binarios y librerías solo para el proyecto.
3. **Uso de gitignore**: Añade un archivo .gitignore para evitar que el entorno virtual y otros archivos no deseados se suban al repositorio.

### ¿Cómo escribir y ejecutar pruebas con Unit Test?

Para escribir pruebas, sigue estas buenas prácticas:

- Crea un archivo de pruebas, como `test_calculator.py`, y empieza importando Unit Test.
- Define clases que hereden de `unittest.TestCase`.
- Escribe métodos de prueba que validen funciones específicas usando `assertEqual` para verificar resultados.

Ejemplo básico de prueba:

```python
import unittest
from src.calculator import add, subtract

class TestCalculator(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_subtract(self):
        self.assertEqual(subtract(10, 5), 5)
```

Ejecuta las pruebas con `python -m unittest discover` para que Unit Testing encuentre y ejecute las pruebas automáticamente.

### ¿Qué hacer cuando una prueba falla?

Si una prueba falla, Unittest lo indica con una “F”, mostrando el error detallado, lo que facilita la depuración. Puedes forzar un fallo, por ejemplo, esperando que la suma de `2 + 3` sea `6` en lugar de `5`, para ver cómo se comporta.

Para configurar un entorno de **pruebas unitarias** (unit testing) en Django, debes asegurarte de que las herramientas necesarias estén instaladas y configuradas adecuadamente. A continuación te indico los pasos detallados para instalar y configurar un entorno de **pruebas unitarias** en **Django**.

### 1. **Instalar Dependencias Necesarias**

Django viene con un framework de pruebas unitarias basado en `unittest` incorporado, por lo que no necesitas instalar ninguna herramienta extra para hacer pruebas unitarias básicas. Sin embargo, a continuación hay herramientas adicionales recomendadas:

#### a. **pytest-django** (opcional)
Si prefieres usar **pytest** en lugar del framework de pruebas de Django, puedes instalar **pytest-django**, que ofrece una forma más flexible y poderosa de escribir y ejecutar pruebas.

```bash
pip install pytest pytest-django
```

#### b. **Factory Boy** (opcional)
**Factory Boy** te permite crear objetos de prueba fácilmente. Es útil cuando necesitas muchos datos de prueba en tus tests.

```bash
pip install factory-boy
```

#### c. **Coverage** (opcional)
Para medir la cobertura de tus pruebas y asegurarte de que gran parte de tu código esté siendo probado.

```bash
pip install coverage
```

### 2. **Configurar `pytest` (si se usa)**

Si decides usar `pytest`, debes crear un archivo `pytest.ini` en la raíz de tu proyecto para configurarlo correctamente:

```ini
# pytest.ini
[pytest]
DJANGO_SETTINGS_MODULE = myproject.settings  # Reemplaza con el nombre de tu proyecto
python_files = tests.py test_*.py *_tests.py
```

### 3. **Configurar la Base de Datos de Pruebas**

Django crea automáticamente una base de datos de pruebas temporal para ejecutar las pruebas. En el archivo `settings.py`, asegúrate de que la base de datos para las pruebas esté configurada de esta manera:

```python
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',  # SQLite es más rápida para las pruebas
        'NAME': ':memory:',  # Base de datos en memoria para pruebas unitarias
    }
}
```

Esto asegura que las pruebas unitarias se ejecuten en una base de datos SQLite en memoria, lo que es mucho más rápido.

### 4. **Escribir Pruebas Unitarias en Django**

Las pruebas unitarias en Django se escriben extendiendo la clase `django.test.TestCase`. Aquí tienes un ejemplo básico:

```python
# tests.py
from django.test import TestCase
from .models import MyModel

class MyModelTestCase(TestCase):
    def setUp(self):
        # Crear datos de prueba que estarán disponibles en cada prueba
        MyModel.objects.create(name="Test Object", value=100)

    def test_object_value(self):
        obj = MyModel.objects.get(name="Test Object")
        self.assertEqual(obj.value, 100)  # Asegura que el valor sea correcto
```

El método `setUp()` se ejecuta antes de cada prueba, lo que te permite preparar datos o configuraciones previas.

### 5. **Ejecutar Pruebas Unitarias**

Si estás usando el framework de pruebas incorporado en Django, puedes ejecutar las pruebas unitarias usando el siguiente comando:

```bash
python manage.py test
```

Si prefieres usar `pytest`:

```bash
pytest
```

### 6. **Ver la Cobertura del Código (opcional)**

Para ver qué partes de tu código están cubiertas por las pruebas, puedes utilizar **Coverage**. Para ejecutar tus pruebas con cobertura, utiliza los siguientes comandos:

```bash
coverage run --source='.' manage.py test
coverage report -m
```

Esto te dará un informe detallado de la cobertura de tus pruebas.

### 7. **Pruebas Unitarias en Django con `APIClient` (opcional)**

Si estás utilizando Django REST Framework, puedes usar el cliente de prueba `APIClient` para probar tus vistas de API. Aquí un ejemplo:

```python
from rest_framework.test import APITestCase

class MyAPITestCase(APITestCase):
    def setUp(self):
        # Crear datos de prueba para la API
        self.client = APIClient()

    def test_get_request(self):
        response = self.client.get('/api/my-endpoint/')
        self.assertEqual(response.status_code, 200)  # Asegura que el código de estado sea 200
```

### 8. **Organización de las Pruebas**

Es recomendable organizar las pruebas unitarias en un archivo `tests.py` o dentro de una carpeta `tests` en cada aplicación de Django. Por ejemplo:

```
myapp/
    models.py
    views.py
    tests/
        test_models.py
        test_views.py
        test_serializers.py
```

### 9. **Uso de Factories para Datos de Prueba (opcional)**

Con **Factory Boy**, puedes simplificar la creación de objetos de prueba en lugar de hacerlo manualmente:

```python
import factory
from .models import MyModel

class MyModelFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = MyModel

    name = "Test Object"
    value = 100
```

Puedes usar esta fábrica en tus pruebas:

```python
class MyModelTestCase(TestCase):
    def setUp(self):
        self.obj = MyModelFactory()

    def test_object_value(self):
        self.assertEqual(self.obj.value, 100)
```

### 10. **Integración Continua (opcional)**

Para mejorar la calidad de tu código, puedes integrar tus pruebas con herramientas de CI/CD como **GitHub Actions**, **Travis CI** o **CircleCI**, para que las pruebas se ejecuten automáticamente cada vez que haces cambios en el código.

**Nota**: para ejecutar el test `python -m unittest discover -s tests`

## Cómo Crear Pruebas Unitarias con UnitTest en Python

Las pruebas unitarias en Python son esenciales para asegurar el correcto funcionamiento del código. Utilizando la clase `TestCase` de la biblioteca `UnitTest`, podemos estructurar pruebas de manera eficiente y limpiar recursos una vez que se han ejecutado. Además, permite automatizar la validación de resultados y la captura de errores. Vamos a profundizar en cómo implementar estas pruebas y algunos métodos clave que facilitan este proceso.

### ¿Cómo configurar las pruebas en Python con TestCase?

El método `setUp()` nos permite configurar elementos antes de que cada prueba se ejecute. Imagina que tienes cinco pruebas que requieren la misma preparación: en lugar de repetir la configuración, puedes ejecutarla una sola vez aquí. Esto ahorra tiempo y esfuerzo al evitar la duplicación de código.

Por ejemplo:

- Puedes utilizar `setUp()` para crear una base común de datos, abrir archivos, o preparar datos de entrada.
- Luego, cada prueba reutiliza esta configuración, asegurando que el entorno siempre esté listo para las pruebas.

### ¿Cómo limpiar después de una prueba?

El método tearDown() sirve para limpiar los recursos utilizados en la prueba. Supongamos que has creado cientos de archivos para una prueba, este método permite eliminarlos automáticamente después de que la prueba finaliza, asegurando que el sistema no quede lleno de datos innecesarios.

Algunos ejemplos de cuándo usarlo:

- Eliminar archivos temporales creados durante las pruebas.
- Cerrar conexiones a bases de datos o liberar recursos del sistema.

### ¿Cómo ejecutar pruebas y capturar errores?

La clase `TestCase` no solo organiza las pruebas, también proporciona un método automático para ejecutar cada una de ellas. El método runTest() gestiona la ejecución de las pruebas, captura los errores y valida que todo funcione correctamente. Este proceso automatiza la validación de resultados esperados y la identificación de fallos.

Por ejemplo:

- Si tienes una lista de pruebas, este método las ejecutará una por una, asegurando que todas se validen correctamente.
- Además, capturará las excepciones que se lancen durante la ejecución.

### ¿Cómo validar excepciones en las pruebas unitarias?

Una situación común en las pruebas de una calculadora es manejar la división por cero. La mejor práctica es lanzar una excepción para evitar errores. Python permite validar que la excepción se ha lanzado correctamente en las pruebas.

Pasos clave:

- Crear una prueba donde `b = 0`.
- Utilizar `assertRaises()` para verificar que se ha lanzado la excepción `ValueError`.

**Lecturas recomendadas**

[unittest — Unit testing framework — Python 3.12.5 documentation](https://docs.python.org/3/library/unittest.html "unittest — Unit testing framework — Python 3.12.5 documentation")

## Cómo usar el método setup en tests de Python

El uso del método setup en los `tests` permite simplificar y evitar la duplicación de código en las pruebas. Al iniciar un test, `setup` se ejecuta automáticamente, preparando el entorno para cada prueba de forma eficiente. En este caso, pasamos de un proyecto de calculadora a uno de una cuenta bancaria, y veremos cómo implementar pruebas unitarias para depósitos, retiros y consultas de saldo utilizando `setup` para optimizar el código.

### ¿Cómo implementar pruebas para depósitos en una cuenta bancaria?

Primero, se crea la clase de test donde se probarán los métodos de una cuenta bancaria. Para hacer un depósito, se debe instanciar una cuenta con un saldo inicial, realizar el depósito y luego validar que el saldo ha cambiado correctamente.

Pasos:

- Crear el archivo `test_bank_account.py`.
- Instanciar una cuenta con saldo inicial.
- Probar que el método de depósito ajusta el saldo correctamente.

### ¿Cómo optimizar las pruebas con el método setup?

El método `setup` evita la creación repetitiva de instancias en cada test. Para lograr esto:

- Se crea una instancia de cuenta en `setup`.
- La cuenta creada se comparte entre todas las pruebas usando `self.`

Esto simplifica las pruebas al evitar duplicar el código de instanciación en cada método de test.

### ¿Cómo ejecutar las pruebas de retiro y consulta de saldo?

Para las pruebas de retiro y consulta de saldo:

- El método `withdraw` debe restar la cantidad del saldo y validar que el resultado sea correcto.
- El método `get_balance` simplemente valida que el saldo actual coincida con lo esperado.

Estas pruebas se benefician del uso de `setup`, ya que reutilizan la misma instancia de cuenta creada para cada prueba.

### ¿Cómo ejecutar pruebas con salida más detallada?

Al ejecutar las pruebas, es útil utilizar el comando con la opción `-b` para obtener una salida más detallada y visualizar exactamente qué pruebas se están ejecutando y dónde están ubicadas en el código. Esto ayuda a depurar y tener un mejor control sobre el flujo de las pruebas.

### ¿Cómo crear pruebas para una nueva funcionalidad de transferencia?

La tarea final consiste en agregar un método de transferencia a la clase `BankAccount`, el cual debe:

- Permitir transferir saldo entre cuentas.
- Levantar una excepción si el saldo no es suficiente para realizar la transferencia.

Luego, se deben crear dos pruebas unitarias:

1. Validar que la transferencia se realiza correctamente.
2. Validar que se lanza una excepción cuando no hay saldo suficiente para completar la transferencia.

para ver las puebas que se estan ejecutando se utiliza el siguiente codigo `python -m unittest discover -v -s tests`

## Uso de tearDown para limpieza de Pruebas Unitarias en Python

El método `teardown `es esencial para asegurar que nuestras pruebas no interfieran entre sí, y se usa para limpiar cualquier recurso temporal al final de cada prueba. En este caso, lo hemos aplicado a nuestra cuenta bancaria, donde se registra un log cada vez que se realiza una acción. Vamos a explorar cómo implementarlo correctamente y agregar funcionalidades de logging a nuestra cuenta de banco.

### ¿Qué es el método teardown y cuándo lo usamos?

El método `teardown` se ejecuta al final de cada prueba, y es utilizado para limpiar recursos como archivos temporales o cerrar conexiones. En este caso, lo usamos para eliminar el archivo de logs que se genera durante nuestras pruebas de la cuenta bancaria. De esta manera, cada prueba se ejecuta en un entorno limpio, sin interferencias de pruebas anteriores.

### ¿Cómo agregamos logging a la cuenta de banco?

Añadimos una funcionalidad de logging en el método `init`, el cual se ejecuta cada vez que se crea una instancia de nuestra clase `BankAccount`. El log incluye eventos como la creación de la cuenta, la consulta de saldo, y cuando se realizan depósitos o retiros. Esto se realiza a través del método `logTransaction`, que escribe el mensaje en un archivo de texto.

- Se define un archivo de log (`logFile`) al crear la cuenta.
- Cada vez que se realiza una transacción o se consulta el saldo, se agrega una línea al archivo.
- Para asegurar que el archivo de log se genera correctamente, creamos pruebas automatizadas.

### ¿Cómo validamos la existencia del archivo de log?

En nuestras pruebas, verificamos si el archivo de log se crea exitosamente. Utilizamos la función `os.path.exists` para validar su existencia y asegurarnos de que nuestras pruebas están funcionando correctamente.

### ¿Cómo usamos el teardown para limpiar archivos?

El `teardown` nos permite eliminar el archivo de log después de cada prueba para que no interfiera con otras. Implementamos una función que, si el archivo existe, lo borra utilizando `os.remove`. Esto asegura que las pruebas se ejecutan en un entorno limpio y los logs no se acumulan entre pruebas.

### ¿Cómo probamos que los logs contienen la información correcta?

Además de verificar que el archivo existe, es fundamental asegurarnos de que el contenido del archivo sea correcto. Para ello, creamos un método que cuenta las líneas del archivo (`countLines`). Luego, en nuestras pruebas, validamos que el número de líneas corresponde al número de transacciones realizadas.

- Contamos las líneas después de crear la cuenta (debe haber una línea).
- Hacemos un depósito y volvemos a contar las líneas (debe haber dos líneas).
- Si no limpiáramos el archivo con `teardown`, el número de líneas sería incorrecto.

### ¿Cómo crear una nueva funcionalidad de logging para transferencias fallidas?

El siguiente reto es agregar una funcionalidad para registrar un log cuando alguien intente hacer una transferencia sin saldo disponible. El log debe incluir un mensaje indicando la falta de fondos, y también se deben crear pruebas que validen que este log se genera correctamente.

- Se debe registrar el intento fallido en el archivo de log.
- Crear una prueba para asegurarse de que el mensaje “No tiene saldo disponible” aparece en el log.
- Utilizar teardown para limpiar el archivo al finalizar cada prueba.

## Cómo validar excepciones y estructuras de datos con Unittest en Python

UnitTest nos proporciona una amplia gama de métodos de aserción que mejoran la forma en que validamos nuestras pruebas. En esta clase, hemos explorado algunos de ellos y cómo utilizarlos en diferentes escenarios.

### ¿Cómo se usa el assertEqual en Unit Test?

El método `assertEqual` compara dos valores para verificar si son iguales. Acepta dos parámetros para comparar y opcionalmente un mensaje personalizado que se mostrará en la terminal si la prueba falla. Este método se integra bien con los editores, permitiendo ejecutar y depurar pruebas de manera eficiente.

- Parámetros: valor esperado, valor obtenido, mensaje de error (opcional)
- Uso típico: Validar igualdad de números, cadenas, o cualquier otro objeto comparable.

### ¿Qué otros métodos de aserción existen en Unit Test?

Además de `assertEqual`, Unit Test incluye muchos otros métodos de aserción útiles:

- `assertTrue`: Verifica que una expresión sea verdadera. No compara valores, solo evalúa si una condición es cierta.
- `assertRaises`: Valida que se lance una excepción específica dentro de un bloque de código, utilizando la palabra clave with como contexto.
- `assertIn` y `assertNotIn`: Comprueban si un elemento está o no está dentro de una secuencia, como una lista o un conjunto.

### ¿Cómo se manejan excepciones en Unit Test?

Con `assertRaises`, se puede verificar que una excepción se lance correctamente. Este método es especialmente útil para manejar errores esperados, como cuando un usuario no tiene suficientes fondos para completar una transferencia.

- Se utiliza con `with` para capturar la excepción dentro de un bloque de código.
- Ejemplo: Capturar un `ValueError` al pasar un argumento no válido a una función.

### ¿Cómo comparar listas, diccionarios y sets en Unit Test?

Unit Test ofrece métodos para comparar estructuras de datos más complejas:

- `assertDictEqual`: Compara dos diccionarios.
- `assertSetEqual`: Compara dos sets para validar que contengan los mismos elementos, independientemente del orden.
- Estos métodos también cuentan con variantes negativas, como `assertNotEqual`, para validar desigualdades.

## Control de pruebas unitarias con unittest.skip en Python

En el desarrollo de software, es común enfrentarse a situaciones donde las pruebas unitarias no pueden ejecutarse por cambios o desarrollos en curso. En estos casos, comentar el código de las pruebas no es la mejor práctica. Afortunadamente, Python y `unittest` ofrecen decoradores que nos permiten omitir pruebas temporalmente, sin comprometer el flujo de trabajo ni la integridad del proyecto. Aquí aprenderemos cómo usar decoradores como `@skip`, `@skipIf` y `@expectedFailure` para manejar estos casos de manera eficiente.

### ¿Cómo utilizar el decorador @skip?

El decorador `@skip` se utiliza cuando sabemos que una prueba no debería ejecutarse temporalmente. Esto es útil si estamos trabajando en un feature que aún no está completo y, por lo tanto, las pruebas no tienen sentido. Al aplicar `@skip`, podemos evitar la ejecución de la prueba y aún así tener visibilidad de que está pendiente de ser corregida.

- Aplicamos el decorador con una razón clara.
- Cuando ejecutamos las pruebas, en el reporte se indicará que la prueba fue saltada.

**Ejemplo de uso:**

```python
@unittest.skip("Trabajo en progreso, será habilitada nuevamente.")
def test_skip_example(self):
    self.assertEqual("hola", "chau")
```

### ¿Cuándo aplicar @skipIf?

El decorador `@skipIf` es útil cuando queremos omitir una prueba bajo una condición específica. Esto es común cuando nuestras pruebas dependen del entorno, como servidores diferentes o configuraciones específicas.

- Requiere una condición y una razón para ser aplicado.
- Se ejecutará solo si la condición es verdadera.

**Ejemplo de uso:**

```python
server = "server_b"
@unittest.skipIf(server == "server_a", "Saltada porque no estamos en el servidor correcto.")
def test_skipif_example(self):
    self.assertEqual(1000, 100)
```

### ¿Qué hace el decorador @expectedFailure?
Este decorador se usa cuando sabemos que una prueba fallará debido a un cambio en la lógica del negocio o un bug conocido, pero queremos mantener la prueba visible en el reporte de pruebas.

- Es útil para reflejar fallos esperados sin interferir con el flujo de integración continua.
- El reporte mostrará que la prueba falló como se esperaba.

**Ejemplo de uso:**

```python
@unittest.expectedFailure
def test_expected_failure_example(self):
    self.assertEqual(100, 150)
```

### ¿Cómo aplicar @skipUnless en casos avanzados?

El decorador `@skipUnless` es valioso cuando queremos ejecutar una prueba solo si se cumple una condición. Un ejemplo clásico es validar si un servicio externo, como una API, está disponible antes de ejecutar la prueba.

- Es ideal para escenarios donde dependemos de recursos externos, como API’s de terceros.

**Ejemplo de uso:**

```python
@unittest.skipUnless(api_available(), "API no disponible.")
def test_skipunless_example(self):
    self.assertEqual(get_currency_rate("USD"), 1.0)
```

### ¿Cuándo utilizar estos decoradores en desarrollo colaborativo?

El uso de decoradores como `@skip`, `@skipIf`, `@expectedFailure` y `@skipUnless` en un equipo de desarrollo asegura que las pruebas no interfieran en el flujo de trabajo, mientras mantienen la visibilidad de las pruebas pendientes. Es esencial en entornos de integración continua (CI), donde se busca que las pruebas no bloqueen el desarrollo, pero sin ignorarlas por completo.

**Lecturas recomendadas**

[unittest — Unit testing framework — Python 3.12.5 documentation](https://docs.python.org/3/library/unittest.html#unittest.skip "unittest — Unit testing framework — Python 3.12.5 documentation")

## Cómo organizar y ejecutar pruebas en Python con UnitTest

Cuando estamos desarrollando en Python, ejecutar todas las pruebas desde la terminal es común en entornos de desarrollo. Sin embargo, en producción o integración continua, este enfoque puede no ser ideal, especialmente cuando solo queremos ejecutar pruebas específicas o tener un mejor control sobre cómo organizamos y ejecutamos estas pruebas. Python y su módulo Unit Test nos ofrecen herramientas como las suites de pruebas para modularizar y seleccionar qué pruebas ejecutar.

### ¿Cómo ejecutar pruebas específicas en Python?

Para ejecutar pruebas específicas, podemos usar el comando `discover` del subcomando Unit Test. Este comando busca automáticamente todas las pruebas dentro de una carpeta y las agrupa en una suite. Sin embargo, este enfoque no es siempre ideal para entornos locales, donde podríamos querer ejecutar solo ciertas pruebas.

### ¿Qué son las suites de pruebas?

Una suite de pruebas es un grupo de pruebas que podemos ejecutar juntas. En proyectos pequeños, generalmente tenemos una sola suite, pero a medida que crece el proyecto, modularizar las pruebas por categorías o características es recomendable. Por ejemplo, podríamos tener una suite para pruebas de calculadora y otra para pruebas de banco.

### ¿Cómo crear y ejecutar una suite de pruebas manualmente?

1. **Crear una suite manualmente:**


 - Creamos un archivo nuevo, por ejemplo, `test_suites.py`.
 - Importamos `UnitTest` y definimos una suite con `suite = unittest.TestSuite()`.
 - Agregamos pruebas a la suite usando `suite.addTest()`.

2. **Agregar pruebas específicas:**

 - Podemos importar pruebas existentes y añadirlas a la suite. Por ejemplo, si ya tenemos una prueba llamada `test_deposit`, podemos agregarla a la suite con `suite.addTest(bankAccountTests('test_deposit'))`.

3. **Ejecutar las suites con un runner:**

 - Para ejecutar una suite, necesitamos un runner. Python ofrece varios tipos de runners. Un ejemplo sería el TextTestRunner, que se usa comúnmente en la terminal.
 - El código básico para ejecutar la suite sería:
 
```python
runner = unittest.TextTestRunner()
runner.run(suite)
```

### ¿Cómo configurar Visual Studio Code para ejecutar pruebas?

Visual Studio Code facilita la ejecución de pruebas con su interfaz gráfica. Podemos configurar un runner y seleccionar qué pruebas ejecutar directamente desde el editor.

1. **Configurar las pruebas en Visual Studio Code:**

 - En la configuración, seleccionamos Unit Test como el framework de pruebas y especificamos la carpeta donde se encuentran las pruebas.
 - Visual Studio Code lista automáticamente las pruebas, permitiéndonos ejecutarlas con un solo clic.

2. **Ejecutar pruebas individuales:**

 - Al hacer clic en una prueba específica, podemos ver su resultado inmediatamente, ya sea que la prueba haya pasado, fallado o se haya omitido.
 
### ¿Cómo solucionar errores comunes al ejecutar pruebas?

Durante la ejecución de las pruebas, es común encontrarse con errores como “módulo no encontrado”. Estos errores se pueden solucionar asegurándonos de que las carpetas contienen archivos `__init__.py` y configurando correctamente el `PYTHONPATH` para que Python encuentre todos los módulos necesarios.

Para que funcione el PYTHONPATH en powershell se usa `$env:PYTHONPATH = "."; python tests/test_suites.py` y se debe usar ``return suite``

```python
import unittest

from test_bank_account import BankAccountTests

def bank_account_suite():
    suite = unittest.TestSuite()
    suite.addTest(BankAccountTests("test_deposit"))
    suite.addTest(BankAccountTests("test_withdraw"))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(bank_account_suite())
```
### ¿Cómo ejecutar pruebas desde la terminal?

Podemos ejecutar una prueba específica directamente desde la terminal usando la siguiente estructura de comando:

`python -m unittest tests.test_calculator.CalculatorTest.test_sum`

Esto ejecuta únicamente la prueba `test_sum` de la clase `CalculatorTest`.

## Mejores prácticas para organizar y nombrar pruebas en Python

Nombrar pruebas correctamente es clave para el éxito en equipo, facilitando la comprensión del código y la colaboración efectiva. Aunque no es obligatorio, tener un formato claro para las pruebas es muy beneficioso.

### ¿Cómo definir el formato de los nombres de las pruebas?

- Todos los tests deben agruparse en clases, cada una relacionada con una clase de tu proyecto. Por ejemplo, si tienes una clase llamada `BankAccount`, la clase de prueba debería llamarse `BankAccountTest`.
- Cada prueba debe comenzar con `test_`, para que las herramientas de testing la identifiquen fácilmente.
- El siguiente elemento en el nombre debe ser el método que estás probando. Si es un método `deposit`, el nombre sería `test_deposit_`.

### ¿Cómo estructurar el escenario de la prueba?

- Después del método, añade el escenario. Esto se refiere a los valores o parámetros que usas en la prueba. Por ejemplo, en el caso de un valor positivo en un depósito, el escenario sería `positive_amount`.

### ¿Cómo describir el resultado esperado?

- Para finalizar el nombre, indica el resultado esperado. Si el depósito incrementa el saldo, añade algo como `increase_balance`. Así, un nombre de prueba completo sería: `test_deposit_positive_amount_increase_balance`.

### ¿Por qué es útil este formato?

- Permite a cualquier miembro del equipo entender el propósito de la prueba sin revisar el código completo.
- Facilita el mantenimiento del código y el soporte, ya que con solo leer el nombre, se entiende el objetivo de la prueba.

### ¿Qué reto se propone para mejorar las pruebas actuales?

- Refactoriza las pruebas que has creado, probando diferentes escenarios y resultados posibles.
- Imagina nuevas circunstancias para probar el código.
- Compara tus resultados con el proyecto refactorizado que estará disponible en la sección de recursos, y comparte tus ideas.

Organizar y nombrar pruebas de manera efectiva en Python es fundamental para asegurar que las pruebas sean fáciles de mantener, claras, y cubran adecuadamente el código que estás probando. Aquí te dejo algunas **mejores prácticas** recomendadas para organizar y nombrar pruebas en Python:

### 1. **Organización de las Pruebas**

#### a. **Usar un Directorio Separado para las Pruebas**
Es una buena práctica colocar todas las pruebas en un directorio llamado `tests/` en la raíz del proyecto. Esto ayuda a mantener las pruebas separadas del código fuente.

```
my_project/
├── my_app/
│   ├── module1.py
│   └── module2.py
├── tests/
│   ├── test_module1.py
│   └── test_module2.py
└── ...
```

#### b. **Mismo Esquema que el Código Fuente**
La estructura del directorio `tests/` debe reflejar la estructura del código fuente. Por ejemplo, si tienes un archivo `module1.py` en tu código fuente, las pruebas correspondientes deberían estar en `tests/test_module1.py`.

#### c. **Agrupar Pruebas en Clases**
Utiliza clases para agrupar pruebas relacionadas. Cada clase debería agrupar pruebas de un módulo o una funcionalidad específica. De esta forma, puedes aprovechar métodos como `setUp` y `tearDown` para inicializar o limpiar datos entre las pruebas.

```python
import unittest

class CalculatorTests(unittest.TestCase):
    def setUp(self):
        # Código para preparar el entorno de pruebas
        pass
    
    def tearDown(self):
        # Código para limpiar después de las pruebas
        pass

    def test_add(self):
        # Prueba para la función de suma
        pass
    
    def test_subtract(self):
        # Prueba para la función de resta
        pass
```

#### d. **Un Archivo de Prueba por Módulo**
Cada módulo o clase del código fuente debería tener su archivo de pruebas correspondiente. Por ejemplo, si tienes un archivo `module1.py`, crea `test_module1.py` con las pruebas para ese archivo.

### 2. **Nombrar las Pruebas**

#### a. **Nombrar Archivos de Pruebas**
Los archivos de pruebas deben seguir la convención de nombrarlos con el prefijo `test_` para que puedan ser detectados automáticamente por los frameworks de pruebas como `unittest` o `pytest`.

```bash
test_module1.py
test_module2.py
```

#### b. **Nombrar Métodos de Prueba**
Los métodos de prueba deben comenzar con `test_`, seguido de una descripción clara de lo que están probando. Esto asegura que las pruebas sean descubiertas automáticamente y hace que su propósito sea evidente. Utiliza nombres descriptivos para los métodos, que reflejen claramente el comportamiento que se está probando.

```python
class CalculatorTests(unittest.TestCase):
    def test_addition_with_positive_numbers(self):
        # Prueba suma con números positivos
        pass
    
    def test_division_by_zero_raises_error(self):
        # Prueba que la división por cero arroje un error
        pass
```

#### c. **Descripciones Claras**
Asegúrate de que los nombres de las pruebas describan el comportamiento esperado, como qué función estás probando y en qué condiciones. Evita nombres genéricos como `test_1`, `test_case` o `test_function`.

### 3. **Separar las Pruebas por Tipos**

#### a. **Pruebas Unitarias**
Las pruebas unitarias deben centrarse en probar unidades individuales de funcionalidad (funciones o métodos). Deben ser rápidas y no depender de recursos externos (como bases de datos o APIs).

#### b. **Pruebas de Integración**
En archivos separados o bajo una estructura diferente, incluye pruebas de integración que prueben cómo las diferentes partes del sistema interactúan entre sí.

```bash
tests/
├── unit/
│   ├── test_module1.py
│   └── test_module2.py
├── integration/
│   ├── test_integration_module1_module2.py
```

### 4. **Utilizar Métodos de Configuración y Limpieza**
Cuando varias pruebas comparten datos o configuraciones, utiliza métodos como `setUp`, `tearDown`, `setUpClass` y `tearDownClass` de `unittest` o los equivalentes de `pytest` para reducir la repetición de código.

```python
class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Código para configurar recursos una sola vez antes de todas las pruebas
        pass
    
    @classmethod
    def tearDownClass(cls):
        # Código para liberar los recursos una vez después de todas las pruebas
        pass

    def setUp(self):
        # Código que se ejecuta antes de cada prueba individual
        pass

    def tearDown(self):
        # Código que se ejecuta después de cada prueba individual
        pass
```

### 5. **Usar Nombres Descriptivos para las Clases de Prueba**
Al igual que los métodos de prueba, las clases de prueba también deben tener nombres descriptivos. Los nombres deben indicar qué clase o funcionalidad están probando.

```python
class UserAuthenticationTests(unittest.TestCase):
    def test_login_with_valid_credentials(self):
        # Prueba de inicio de sesión con credenciales válidas
        pass
```

### 6. **Documentación y Comentarios**
Incluir comentarios breves o docstrings en las pruebas puede ayudar a aclarar lo que se está probando y por qué. Esto es especialmente útil si la prueba tiene una lógica compleja o si estás probando un caso límite específico.

```python
class MathOperationsTests(unittest.TestCase):
    def test_square_root_of_negative_number_raises_error(self):
        """Debe generar un ValueError cuando se intenta calcular la raíz cuadrada de un número negativo"""
        with self.assertRaises(ValueError):
            math.sqrt(-1)
```

### 7. **Evitar Dependencias entre Pruebas**
Cada prueba debe ser independiente y no debe depender de los resultados o el estado de otras pruebas. Esto garantiza que puedan ejecutarse de forma aislada, en cualquier orden.

### 8. **Ejecutar Pruebas Automáticamente**
Utiliza herramientas como `pytest` o `unittest` para ejecutar las pruebas de forma automática y generar informes. Puedes integrarlas en un sistema de integración continua (CI) como Travis CI o GitHub Actions.

### 9. **Uso de Frameworks de Prueba**
- **`unittest`**: Es el framework de pruebas estándar en Python y tiene una buena integración con la línea de comandos.
- **`pytest`**: Proporciona una sintaxis más simple y muchas características adicionales, como fixtures y el descubrimiento automático de pruebas.

### Conclusión
Una buena organización y una convención de nombres adecuada hacen que tus pruebas sean más mantenibles y fáciles de entender. Siguiendo estas prácticas, te asegurarás de que tus pruebas sean eficaces y fáciles de trabajar a largo plazo.

## Mocking de APIs externas en Python con unittest

La simulación de servicios externos es crucial en proyectos de software para garantizar que las pruebas no dependan de APIs externas. Para lograrlo, utilizamos los Mocks, que permiten evitar las llamadas reales a servicios y, en cambio, retornan respuestas controladas en nuestras pruebas. En este caso, aprenderemos a mockear una API de geolocalización y a realizar pruebas efectivas.

### ¿Qué es un Mock y cómo nos ayuda?

Un Mock es una herramienta que nos permite simular comportamientos de funciones o servicios externos. En lugar de ejecutar una llamada real a una API, podemos definir una respuesta predefinida, lo que permite:

- Evitar depender de servicios externos en pruebas.
- Acelerar la ejecución de las pruebas.
- Controlar los resultados esperados.

### ¿Cómo integramos una API externa en Python?
Primero, se configura una función que recibe la IP del cliente y devuelve la ubicación mediante una API de terceros. Para hacer esto:

1. Se instala la librería requests con `pip install requests`.
2. Se crea un archivo `api_client.py` donde conectamos con la API utilizando requests.get.
3. Al recibir la respuesta, se convierte el resultado a JSON para obtener la información de país, ciudad y región.

### ¿Cómo probamos sin hacer llamadas reales?

El problema principal de las pruebas de integraciones con APIs es que pueden demorar, ya que las respuestas dependen de factores externos. Para evitar esto, se usan Mocks. A través del decorador `@patch` de `unittest.mock`, podemos interceptar la llamada a la API y retornar datos predefinidos.

Pasos a seguir:

- Decorar la función de prueba con `@patch.`
- Simular el valor retornado usando mock.return_value para definir qué debe devolver la llamada a la API.
- Definir tanto el código de estado como el contenido del JSON que esperamos recibir.

### ¿Cómo validar que nuestra simulación funciona correctamente?

Además de simular respuestas, debemos asegurarnos de que las pruebas validen correctamente los llamados. Se puede usar `assertCalledOnceWith` para garantizar que la URL y los parámetros pasados son los correctos.

**Lecturas recomendadas**

[unittest.mock — mock object library — Python 3.12.5 documentation](https://docs.python.org/3/library/unittest.mock.html)

## Uso de Side Effects en Mocking con Python

Mock nos permite simular comportamientos variables, una herramienta útil cuando queremos probar cómo reacciona nuestro código ante diferentes escenarios sin modificar el entorno real. Uno de los usos más poderosos es el “side effect”, que nos ayuda a hacer que un método falle en un caso y funcione en otro. Esto es clave para manejar errores temporales, como en el caso de una API de pagos que rechaza una tarjeta incorrecta, pero acepta una correcta en un segundo intento.

### ¿Cómo se define un “side effect” en Mock?

El “side effect” en Mock nos permite modificar el comportamiento de un método en distintas llamadas. Se define como una lista de comportamientos, donde cada elemento de la lista corresponde al resultado de una llamada específica. Esto permite:

- Simular fallos de manera controlada, como lanzar excepciones específicas en las pruebas.
- Probar el código bajo diferentes condiciones sin interactuar con los servicios externos.

### ¿Cómo se maneja un error y una respuesta exitosa en pruebas unitarias?

En una prueba unitaria con Mock, podemos definir comportamientos variables. Por ejemplo, para simular una excepción, usamos la estructura side_effect, donde la primera llamada lanza un error y la segunda retorna una respuesta exitosa. Esto permite cubrir ambos casos sin necesidad de realizar un request real.

- Definimos el primer comportamiento con una excepción.
- Luego, para el segundo comportamiento, usamos Mock para devolver un objeto con los valores que esperamos, simulando una respuesta exitosa.

### ¿Qué métodos auxiliares facilita Mock?

Mock facilita métodos como `raiseException`, que lanza una excepción específica, y `mock`, que permite crear objetos personalizados. Estos objetos pueden tener parámetros configurables como `status_code` y devolver datos específicos al llamar métodos como JSON. Este tipo de pruebas es crucial para validar la resiliencia del software ante errores temporales.

### ¿Cómo integrar validaciones adicionales en las pruebas?

Para reforzar las pruebas, puedes agregar validaciones adicionales, como simular el envío de una IP inválida. En este caso, si la IP es incorrecta, se debe lanzar un error, mientras que si es válida, debe retornar los datos de geolocalización. Esto se implementa agregando más casos en la lista de side effects, cubriendo así todas las situaciones posibles.

## Uso de Patching para Modificar Comportamientos en Python

En esta lección, hemos aprendido a modificar el comportamiento de objetos y funciones dentro de nuestras pruebas en Python, utilizando técnicas como el patch para simular situaciones específicas, como el control del horario de retiro en una cuenta bancaria. Esta habilidad es crucial cuando necesitamos validar restricciones temporales o cualquier otra lógica de negocio que dependa de factores externos, como el tiempo.

### ¿Cómo podemos restringir el horario de retiros en una cuenta bancaria?

Para implementar la restricción de horario, se utilizó la clase `datetime` para obtener la hora actual. Definimos que los retiros solo pueden realizarse durante el horario de oficina: entre las 8 AM y las 5 PM. Cualquier intento fuera de este horario lanzará una excepción personalizada llamada `WithdrawalError`.

- Se implementó la lógica en el método de retiro de la clase `BankAccount`.
- La restricción se basa en comparar la hora actual obtenida con `datetime.now().hour`.
- Si la hora es menor que las 8 AM o mayor que las 5 PM, se lanza la excepción.

### ¿Cómo podemos probar la funcionalidad de manera efectiva?

Las pruebas unitarias permiten simular diferentes horas del día para validar que las restricciones funcionen correctamente. Para lograrlo, usamos el decorador `patch` del módulo `unittest.mock`, el cual modifica temporalmente el comportamiento de la función `datetime.now()`.

- Con `patch`, podemos definir un valor de retorno específico para `now()`, como las 7 AM o las 10 AM.
- De esta forma, se puede validar que la excepción se lance correctamente si el retiro ocurre fuera del horario permitido.
- En caso de que el retiro sea dentro del horario, la prueba verificará que el saldo de la cuenta se actualice correctamente.

### ¿Cómo corregimos errores en la lógica de negocio?

Durante la implementación, encontramos un error en la condición lógica del horario. Inicialmente, se utilizó un operador `and` incorrecto para verificar si la hora estaba dentro del rango permitido. Este error se corrigió cambiando la condición a un `or`, asegurando que la lógica prohibiera retiros antes de las 8 AM y después de las 5 PM.

## Cómo parametrizar pruebas en Python con SubTest

El uso de SubTest en UnitTest te permite optimizar tus pruebas evitando la duplicación de código. Imagina que necesitas probar un método con varios valores diferentes. Sin SubTest, tendrías que crear varias pruebas casi idénticas, lo que resulta ineficiente. SubTest permite parametrizar pruebas, lo que significa que puedes ejecutar la misma prueba con diferentes valores sin repetir el código.

### ¿Cómo evitar la duplicación de pruebas con SubTest?

Al utilizar SubTest, puedes definir todos los valores que deseas probar en una lista o diccionario. Luego, iteras sobre estos valores mediante un bucle `for,` ejecutando la misma prueba con cada conjunto de parámetros. Así, si es necesario modificar la prueba, solo tienes que hacer cambios en un único lugar.

### ¿Cómo implementar SubTest en un caso práctico?

Para ilustrarlo, se puede crear una prueba llamada `test_deposit_various_values`. En lugar de duplicar la prueba con diferentes valores de depósito, utilizas un diccionario que contiene los valores a probar y el resultado esperado. Después, recorres estos valores con SubTest usando la estructura `with self.subTest(case=case)` y ejecutas la prueba para cada valor del diccionario. Esto asegura que cada prueba sea independiente y evita sumar valores a la cuenta de manera incorrecta.

### ¿Cómo gestionar errores con SubTest?

SubTest también es útil para identificar errores específicos. Si una prueba falla con un conjunto particular de parámetros, SubTest te permite ver fácilmente qué valores causaron el fallo. Esto facilita mucho la corrección de errores, ya que puedes aislar rápidamente los casos problemáticos y corregirlos de manera eficiente.

En Python, el uso de `subTest` dentro del módulo `unittest` permite parametrizar pruebas, facilitando la ejecución de múltiples casos de prueba dentro de una sola función de prueba. Esto es útil cuando quieres probar una función con diferentes entradas y verificar sus resultados sin necesidad de crear múltiples métodos de prueba. Aquí tienes una guía sobre cómo usar `subTest` para parametrizar pruebas.

### 1. Importar el Módulo Necesario

Primero, asegúrate de importar `unittest`.

```python
import unittest
```

### 2. Definir tu Clase de Pruebas

Crea una clase de prueba que herede de `unittest.TestCase`.

```python
class TestMiFuncion(unittest.TestCase):
    def mi_funcion(self, x):
        # Esta es la función que estás probando
        return x * 2
```

### 3. Usar `subTest` para Parametrizar las Pruebas

Dentro de tu método de prueba, utiliza un bucle para iterar sobre los casos de prueba. Usa `subTest` para encapsular cada caso.

```python
class TestMiFuncion(unittest.TestCase):
    def mi_funcion(self, x):
        return x * 2

    def test_multiplicacion(self):
        casos_de_prueba = [
            (1, 2),
            (2, 4),
            (3, 6),
            (4, 8),
            (5, 10)
        ]
        
        for entrada, resultado_esperado in casos_de_prueba:
            with self.subTest(entrada=entrada):
                resultado = self.mi_funcion(entrada)
                self.assertEqual(resultado, resultado_esperado)
```

### 4. Ejecutar las Pruebas

Finalmente, ejecuta las pruebas utilizando la línea de comando.

```bash
python -m unittest nombre_del_archivo.py
```

### Ejemplo Completo

Aquí tienes un ejemplo completo que incluye todas las partes mencionadas:

```python
import unittest

class TestMiFuncion(unittest.TestCase):
    def mi_funcion(self, x):
        return x * 2

    def test_multiplicacion(self):
        casos_de_prueba = [
            (1, 2),
            (2, 4),
            (3, 6),
            (4, 8),
            (5, 10)
        ]
        
        for entrada, resultado_esperado in casos_de_prueba:
            with self.subTest(entrada=entrada):
                resultado = self.mi_funcion(entrada)
                self.assertEqual(resultado, resultado_esperado)

if __name__ == '__main__':
    unittest.main()
```

### Ventajas de Usar `subTest`

- **Claridad**: Cada caso de prueba se ejecuta de manera aislada, lo que facilita identificar cuál falló.
- **Mantenimiento**: Evita la duplicación de código, lo que hace que sea más fácil mantener y actualizar las pruebas.
- **Resultados Detallados**: `subTest` proporciona un informe claro sobre qué subprueba falló, lo que facilita la depuración.

Usar `subTest` es una excelente manera de mantener tus pruebas organizadas y eficientes, especialmente cuando trabajas con múltiples casos de entrada y salida.

## Documentación de pruebas unitarias con Doctest en Python

El uso de Doctest es una herramienta poderosa que te permite escribir pruebas directamente en la documentación del código, lo que facilita que otros desarrolladores comprendan y verifiquen los resultados esperados. Además de los Unit Tests tradicionales, Doctest permite que tus comentarios sean interactivos, ofreciendo ejemplos funcionales que se ejecutan dentro del código de Python. Veamos cómo puedes utilizarlo de manera eficiente.

### ¿Qué es Doctest y cómo se usa?

Doctest es una librería que está incluida en Python y que permite crear pruebas en los comentarios del código. Esto lo hace práctico ya que puedes escribir pruebas de manera muy similar a una sesión interactiva de Python. Solo debes añadir los ejemplos dentro de los comentarios y ejecutarlos con el comando python -m doctest.

### ¿Cómo se estructuran las pruebas en Doctest?

Para escribir una prueba, simplemente crea un comentario que simule una sesión interactiva. Estas sesiones se caracterizan por comenzar con `>>>`. Por ejemplo, si tienes una función de suma en tu clase Calculator, podrías escribir lo siguiente:

```bash
>>> sum(5, 7)
12
```

Esto se ejecutará como si estuvieras en el shell de Python, y esperará que la salida sea `12`. Si el resultado no coincide con lo esperado, Doctest te notificará el error.

### ¿Qué hacer si hay un error en la prueba?

Si Doctest encuentra un error, revisa el mensaje de error y ajusta el código o la prueba según sea necesario. Por ejemplo, si ejecutas una prueba y esperabas `12` pero el resultado fue `11`, Doctest te informará de la discrepancia. Solucionas el error, corriges el comentario, y ejecutas nuevamente.

### ¿Cómo manejar excepciones en Doctest?

Doctest también te permite probar excepciones. Si tienes una función que lanza un `ValueError` al intentar dividir por cero, puedes capturar este comportamiento en el comentario:

```bash
>>> divide(10, 0)
Traceback (most recent call last):
  ...
ValueError: División por 0 no permitida
```

Este tipo de pruebas asegura que las excepciones se manejen correctamente y ayuda a otros desarrolladores a entender los casos de error.

### ¿Por qué es importante documentar con Doctest?

La documentación clara es clave en cualquier proyecto de software, y Doctest facilita agregar ejemplos en el código que no solo explican cómo usar las funciones, sino que además se prueban automáticamente. Esto garantiza que la documentación esté siempre alineada con el comportamiento real del código.

### ¿Qué reto implica utilizar Doctest?

El reto de este enfoque es agregar suficiente documentación con ejemplos ejecutables que cubran todos los casos, incluyendo los casos de borde, como divisiones por cero o parámetros inválidos. Este proceso no solo mejora la calidad del código, sino también la de la documentación, haciéndola más útil para todo el equipo.