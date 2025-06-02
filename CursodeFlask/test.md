# Curso de Flask

1. **¿Cuál es la principal ventaja de usar Flask como framework web en Python?**
   
**R//=** Es un microframework ligero y flexible

2. **¿Qué comando necesitas ejecutar para iniciar un servidor de desarrollo Flask?**
 
**R//=** flask run

3. **¿Cómo defines una ruta en Flask que responda tanto a GET como a POST?**
   
**R//=** @app.route(’/’ , methods=[“POST”, “GET”])

4. **¿Qué sintaxis se usa para mostrar una variable en una plantilla Jinja?**
   
**R//=** {{ variable }}

5. **¿Qué método de SQLAlchemy se usa para crear todas las tablas en la base de datos?**
    
**R//=** create_all()

6. **¿Qué método HTTP se usa para actualizar datos existentes en una aplicación RESTful?**
    
**R//=** PUT

7. **¿Qué método de SQLAlchemy se usa para eliminar un registro de la base de datos?**
    
**R//=** db.session.delete()

8. **¿Cuál es la mejor práctica para organizar una aplicación Flask de mediano tamaño?**
    
**R//=** Usar una estructura modular con blueprints

9. **¿Qué método se usa para registrar un Blueprint en una aplicación Flask?**
    
**R//=** app.register_blueprint()

10. **¿Qué función de Flask se usa para mostrar mensajes flash al usuario?**
    
**R//=** flash()

11. **¿Dónde se debe incluir el CDN de TailwindCSS en una aplicación Flask?**
    
**R//=** En el template base HTML

12. **¿Qué configuración es necesaria para usar sesiones en Flask?**
    
**R//=** SECRET_KEY

13. **¿Qué método se usa para cerrar la sesión de un usuario en Flask?**
    
**R//=** session.clear()

14. **¿Qué método se usa para validar datos de un formulario en Flask?**
    
**R//=** request.form.get()

15. **¿Qué clase base se usa para crear pruebas en Flask?**
    
**R//=** unittest.TestCase