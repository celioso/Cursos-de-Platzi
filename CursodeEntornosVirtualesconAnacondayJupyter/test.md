# Curso de Entornos Virtuales con Anaconda y Jupyter

1. **¿Cuál es la principal ventaja de usar entornos virtuales en Python?**
   
**R//=** Evitar conflictos entre diferentes versiones de bibliotecas en distintos proyectos.

2. **¿Cuál es una diferencia clave entre Anaconda y Miniconda?**
 
**R//=** Anaconda incluye más de 250 bibliotecas preinstaladas, mientras que Miniconda tiene menos de 70.

3. **¿Cuál es el comando para crear un entorno virtual con Python 3.9 en Conda?**
   
**R//=** conda create --name mi_entorno python=3.9

4. **¿Cuál es el comando para eliminar un entorno virtual llamado mi_entorno junto con todos sus paquetes?**
   
**R//=** conda remove --name mi_entorno --all

5. **¿Cuál es el comando correcto para actualizar todos los paquetes instalados en un entorno activo?**
    
**R//=** conda update --all

6. **¿Cuál es el paso correcto para crear un nuevo entorno virtual en Anaconda Navigator?**
    
**R//=** Hacer clic en Create en la pestaña Environments.

7. **¿Cuál es la principal ventaja de usar Jupyter Notebooks en proyectos de ciencia de datos?**
    
**R//=** Combinar código, visualizaciones y documentación en un mismo archivo.

8. **¿Cuál es la diferencia entre %time y %%time?**
    
**R//=** %time mide el tiempo de ejecución de una línea de código, mientras que %%time mide el tiempo de toda la celda.

9. **¿Por qué es difícil comparar versiones de notebooks directamente con Git?**
    
**R//=** Porque los notebooks están en formato JSON, y Git compara texto línea por línea.

10. **¿Cuál es una de las principales ventajas de JupyterLab frente a Jupyter Notebooks?**
    
**R//=** JupyterLab permite trabajar con múltiples paneles y archivos al mismo tiempo.

11. **¿Qué comando usarías para dar prioridad a los paquetes del canal conda-forge en Conda?**
    
**R//=** conda config --set channel_priority strict

12. **¿Qué comando se utiliza para instalar Cookiecutter usando Conda?**
    
**R//=** conda install conda-forge::cookiecutter

13. **¿Qué archivo sería un pre-hook en Cookiecutter?**
    
**R//=** pre_gen_project.py