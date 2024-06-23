# Curso de Docker

## Las tres áreas en el desarrollo de software profesional

Docker te permite construir, distribuir y ejecutar cualquier aplicación en cualquier lado.

**Problemática:**

**Construir:** Un entorno de desarrollo donde podamos escribir código resolviendo las siguientes problemáticas:

- Entorno de desarrollo.- paquetes y sus versiones

- Dependencias.- frameworks, bibliotecas

- Entorno de ejecución.- Versiones de node, etc.

- Equivalencia con entorno productivo.- Simular lo mas posible el entorno local al productivo.

- Servicios externos.- Comunicación con base de datos (versiones, etc)

**Distribuir**: Llevar nuestro código (artefactos) a donde tenga que llegar.

- Divergencia de repositorios.- Colocar los artefactos en su repositorio correspondiente.

- Divergencia de artefactos.- Existen variedad de artefactos en una aplicación.

- Versionamiento.- Mantener un versionado de código en las publicaciones.

**Ejecutar**: Hacer que el desarrollo o los artefactos que se programaron en local funcionen en productivo. La máquina donde se escribe el software siempre es distinta a la máquina donde se ejecuta de manera productiva.

- Compatibilidad con el entorno productivo

- Dependencias

- Disponibilidad de servicios externos

- Recursos de hardware

## Virtualización

Docker y las máquinas virtuales son dos tecnologías de virtualización que se utilizan para ejecutar aplicaciones en un entorno aislado. Sin embargo, existen algunas diferencias clave entre las dos tecnologías que pueden afectar su elección.

**Docker**

**Ventajas:** Uso de recursos más eficiente Arranque más rápido Mayor portabilidad Desventajas: No proporcionan aislamiento completo Pueden ser más difíciles de configurar Máquinas virtuales

**Ventajas:** Aislamiento completo Fácil de configurar Desventajas: Uso de recursos más intensivo Arranque más lento Menor portabilidad Cuando usar Docker

**Docker es una buena opción para las siguientes situaciones:**

Aplicaciones que requieren un uso eficiente de los recursos: Los contenedores Docker son mucho más ligeros que las máquinas virtuales, por lo que pueden ejecutarse en equipos con menos recursos. Aplicaciones que requieren un rápido arranque: Los contenedores Docker se pueden iniciar en cuestión de segundos, mientras que las máquinas virtuales pueden tardar minutos. Aplicaciones que requieren portabilidad: Las aplicaciones que se ejecutan en contenedores Docker se pueden ejecutar en cualquier entorno que admita Docker. Cuando usar máquinas virtuales

**Las máquinas virtuales son una buena opción para las siguientes situaciones:**

Aplicaciones que requieren un aislamiento completo: Las máquinas virtuales proporcionan un aislamiento completo entre las aplicaciones, lo que puede ser importante para aplicaciones que requieren un alto nivel de seguridad o que pueden interferir entre sí. Aplicaciones que requieren una configuración compleja: Las máquinas virtuales son más fáciles de configurar que los contenedores Docker, especialmente para aplicaciones que requieren un sistema operativo personalizado o configuraciones específicas de hardware.