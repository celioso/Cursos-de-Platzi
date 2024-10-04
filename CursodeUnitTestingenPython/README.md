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