### Curso de FastAPI

1. **¿Cuál es la mayor ventaja de usar SQLAlchemy?**
**R/:** Facilita el acceso a bases de datos relacionales mediante el mapeo de tablas a clases.

2. **¿Cómo puedes agregar SQLAlchemy a tu aplicación?**
**R/:** Ejecutando el comando "pip install sqlalchemy"

3. **¿Qué parámetro recibe la función create_engine()?**
**R/:** La URL con los datos de conexión a la base de datos

4. ¿Cómo indicamos que un atributo es una columna en un modelo de SQLAlchemy?
**R/:** Usando la clase Column

5. **¿Para qué sirve el método commit en SQLAlchemy?**
**R/: ** Confirma y actualiza los cambios pendientes realizados a la base de datos

6. **¿Cuál es la sintaxis correcta para obtener todos los datos de una tabla con SQLALchemy?**
**R/: ** db.query(Model).all()

7. **¿Cuál es la sintaxis correcta para eliminar un registro de una tabla con SQLALchemy?**
**R/: ** db.query(Model).filter(condition).delete()

8. **¿Cuál de las siguientes es una ventaja de SQLModel?**
**R/: ** Te permite crear esquemas de pydantic y modelos de bases de datos sin duplicar código

9. **¿Cuál es la sintaxis para añadir un middleware a tu aplicación?**
**R/: ** app.add_middleware(MyMiddleware)

10. **¿Cuál es la clase utilizada para crear routers en FastAPI?**
**R/: ** APIRouter

11. **¿Cuál es la sintaxis para incluir un router en tu aplicación?**
**R/: ** app.include_router(my_router)

12. **¿Con cuál de los siguientes comandos puedes ejecutar tu aplicación con PM2?**
**R/: ** `pm2 start "uvicorn main:app"`

13. **¿En qué framework está basado FastAPI?**
**R/:** Starlette

14. **¿A partir de qué versión puedes utilizar FastAPI?**
**R/:** 3.6

15. **¿Cuál es la sintaxis para añadir un parámetro a una ruta?**
**R/:** /movies/{id}

16. **¿De qué clase debe heredar un esquema de datos de Pydantic?**
**R/:** BaseModel

17. **¿Qué nombre debe tener la clase para realizar configuraciones en un esquema de Pydantic?**
**R/:** Config

18. **¿Qué parámetro puedes usar en la clase JSONResponse para enviar la información al cliente?**
**R/:**  json * Esta mal

19. **¿Cuál es el nombre correcto del módulo para manejo de tokens?**
**R/:**  pyjwt

20. **¿Cómo podemos añadir una clase como dependencia de una ruta?**
**R/:**  Usando el parámetro depends * Esta mal