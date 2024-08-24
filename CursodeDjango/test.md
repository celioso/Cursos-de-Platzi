### Curso de Django

1. **¿Cuál es el comando para crear un nuevo proyecto en Django?**

**R/:** django-admin startproject myproject

2. **¿Qué archivo debes modificar para registrar un modelo en el panel de administración?**

**R/:** admin.py

3. **¿Cuál es el comando para aplicar migraciones en Django?**

**R/:** python manage.py migrate

4. **¿Cómo defines una URL en Django?**

**R/:** urlpatterns = [ path('home/', views.home, name='home'), ]

5. **¿Qué archivo necesitas para definir un formulario en Django?**

**R/:** forms.py

6. **¿Cuál es el propósito de la función render en Django?**

**R/:** Renderizar una plantilla y devolver una respuesta HTTP

7. **¿Cómo configuras la base de datos en un proyecto Django?**

**R/:** En el archivo settings.py, en la sección DATABASES

8. **¿Qué comando se utiliza para iniciar el servidor de desarrollo en Django?**

**R/:** python manage.py runserver

9. **¿Cuál es el archivo de configuración principal en Django?**

**R/:** settings.py

10. **¿Qué biblioteca se utiliza para trabajar con formularios en Django?**

**R/:** django.forms

11. **¿Cómo se crea un superusuario en Django?**

**R/:** python manage.py createsuperuser

12. **¿Cuál es el propósito de MEDIA_URL y MEDIA_ROOT en Django?**

**R/:** Para manejar archivos subidos por usuarios

13. **¿En qué archivo debes registrar una aplicación nueva un proyecto Django?**

**R/:** settings.py

14. **¿Cuál es la forma correcta de importar un modelo en Django?**

**R/:** from myapp.models import MyModel

15. **¿Cómo defines una relación Many-to-One en un modelo de Django?**

**R/:** class MyModel(models.Model): related_model = models.ForeignKey(RelatedModel, on_delete=models.CASCADE)

16. **¿Cómo defines una relación Many-to-Many en un modelo de Django?**

**R/:** class MyModel(models.Model): related_model = models.ManyToManyField(RelatedModel)

17. **¿Cómo defines una relación One-to-One en un modelo de Django?**

**R/:** class MyModel(models.Model): related_model = models.OneToOneField(RelatedModel, on_delete=models.CASCADE)

18. **¿Qué comando se utiliza para crear migraciones en Django?**

**R/:** python manage.py makemigrations

19. **¿Para proteger una aplicación Django contra ataques de inyección SQL, qué medidas se deben tomar?**

**R/:** Usar consultas ORM de Django

20. **¿Cuál es la ventaja de usar vistas basadas en clases en lugar de vistas basadas en funciones en Django?**

**R/:** Reutilización de código y mayor organización

21. **¿Para manejar la autenticación de usuarios en Django, qué módulo es adecuado?**

**R/:** django.contrib.auth

22. **¿Qué patrón de diseño sigue Django para la separación de lógica y presentación?**

**R/:** Modelo-Vista-Template (MVT)

23. **¿Cómo puedes mejorar el rendimiento de una aplicación Django?**

**R/:** Usar cache y optimizar consultas de base de datos

24. **¿Qué función en Django se utiliza para redireccionar a otra URL?**

**R/:** redirect

25. **¿Qué middleware en Django se utiliza para habilitar la protección CSRF?**

**R/:** django.middleware.csrf.CsrfViewMiddleware