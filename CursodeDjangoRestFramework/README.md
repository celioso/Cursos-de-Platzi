# Curso de Django Rest Framework

## Crea y escala APIs con Django REST Framework

Imagina un mundo donde las aplicaciones no pueden compartir información entre ellas. Tu app de pedidos en línea no sabría tu ubicación ni si tienes saldo para pagar. ¿Qué es lo que falta? Exacto, una API. Las APIs son las autopistas de datos que permiten a las aplicaciones intercambiar información de manera efectiva, y para ello utilizan un estilo arquitectónico llamado REST. A través de métodos como GET, POST o DELETE, REST define cómo los mensajes viajan por internet. Sin embargo, crear una API desde cero puede ser complicado, y ahí es donde entra en juego Django REST Framework.

### ¿Por qué las APIs son esenciales para las aplicaciones?

- Las APIs conectan aplicaciones permitiendo que compartan información en tiempo real.
- Sin APIs, no sería posible realizar tareas básicas como verificar tu ubicación o procesar pagos.
- Permiten la comunicación eficiente entre servidores, fundamental para la funcionalidad de cualquier aplicación moderna.

### ¿Cómo facilita Django REST Framework la creación de APIs?

- Django REST Framework permite configurar y desplegar APIs sin necesidad de crear todo desde cero.
- Se encarga de la seguridad, la comunicación y la interacción con bases de datos, ofreciendo un enfoque escalable.
- Este framework se enfoca en la simplicidad y rapidez, haciendo que el desarrollo sea eficiente y sin complicaciones.

### ¿Qué hace a Django REST Framework adecuado tanto para principiantes como para expertos?

- Empresas de todos los tamaños, desde startups hasta grandes corporaciones, usan Django REST Framework debido a su versatilidad y facilidad de uso.
- No es necesario ser un experto para empezar a trabajar con él, lo que lo convierte en una opción accesible para cualquier desarrollador.
- Al utilizar Django REST Framework, puedes concentrarte en lo que realmente importa: crear experiencias digitales de calidad.

### ¿Qué beneficios ofrece Django REST Framework en la producción de APIs?

- Ahorra tiempo al evitar el desarrollo de funciones repetitivas y básicas.
- Integra funciones clave como autenticación, manejo de datos y seguridad de forma nativa.
- Facilita la escalabilidad, permitiendo que las aplicaciones crezcan sin problemas técnicos mayores.

**Lecturas recomendadas**

[Home - Django REST framework](https://www.django-rest-framework.org/)

## Introducción a las APIs, REST y JSON

Las APIs (Application Programming Interfaces) permiten que los computadores se comuniquen entre ellos de manera estructurada, usando formatos que ambos pueden entender. Son esenciales en el desarrollo moderno, automatizando procesos y facilitando la integración entre sistemas, como el caso de las plataformas de pago o la personalización de publicidad. JSON es el formato más utilizado en estas interacciones, permitiendo compartir información como texto, arreglos y objetos. Las APIs REST, basadas en JSON y HTTP, aseguran comunicaciones predecibles entre servidores y clientes.

### ¿Qué es una API y cómo funciona?

- Las APIs permiten la comunicación entre computadores de manera estructurada.
- Se utilizan principalmente para enviar solicitudes y recibir respuestas entre servidores o entre un servidor y un cliente.
- Son fundamentales para la automatización de tareas en el desarrollo web moderno.

### ¿Cómo se usan las APIs en la vida cotidiana?

- Existen APIs comunes, como la de Facebook, que utiliza tus búsquedas para mostrarte publicidad personalizada.
- Las APIs de pago, como Stripe, permiten gestionar tarjetas de crédito de manera segura.
- Estas herramientas evitan que los desarrolladores deban implementar complejas normativas de seguridad en sus propios servidores.

### ¿Qué es el formato JSON y por qué es importante?

- JSON (JavaScript Object Notation) es el formato estándar para enviar y recibir datos a través de APIs.
- Permite almacenar y estructurar información como texto, arreglos y objetos.
- Por ejemplo, un usuario puede tener varios hobbies, y estos se almacenan en un arreglo dentro de un JSON.

### ¿Cómo se estructuran las APIs REST?

- REST (Representational State Transfer) es una arquitectura que define cómo deben enviarse los mensajes a través de HTTP usando JSON.
- Garantiza que las comunicaciones sean predecibles, lo que significa que las mismas solicitudes siempre producirán los mismos resultados.

### ¿Cuáles son los métodos principales de una API REST?

- **GET**: Se utiliza para obtener información. Puede devolver una lista de recursos o un recurso específico.
- **POST**: Permite crear nuevos recursos, como agregar un nuevo usuario.
- **DELETE**: Utilizado para eliminar un recurso existente.
- **PUT y PATCH**: Modifican la información de un recurso, ya sea un solo campo o todo el contenido.

**Lecturas recomendadas**

[JSON Online Validator and Formatter - JSON Lint](https://jsonlint.com/)

[GitHub - platzi/django-rest-framework](https://github.com/platzi/django-rest-framework)

[HTTP request methods - HTTP | MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)