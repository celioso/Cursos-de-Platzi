# Curso de PostgreSQL Aplicado a Ciencia de Datos

1. **¿Cuál de las siguientes opciones es una ventaja de usar infraestructura cómo código?**
   
**R//=** Tomar ventajas de prácticas de desarrollo para manejar nuestra infraestructura.

2. **¿Cuál de las siguientes opciones puedes desplegar infraestructura como código?**
 
**R//=** Pulumi, Cloudformation, SDK, CDK, Serverless Framework, AWS SAM, Terraform.

3. **¿En cual notación se pueden crear templates de cloudformation?**
   
**R//=** YAML y JSON

4. **¿Cuál de los siguientes es obligatorio en un template de cloudformation?**
   
**R//=** Resources

5. **¿Cómo podemos exportar el valor de una propiedad específica de un recurso?**
    
**R//=** Usando solo Outputs

6. **¿Qué pasa en cloudformation cuando un recurso dentro de un stack falla en crearse?**
    
**R//=** Se hace rollback, todos los recursos creados se eliminan.

7. **¿En qué caso usarías stack sets en vez de usar stacks?**
    
**R//=** Para hacer despliegues multi-cuenta.

8. **¿Cómo se llama a dividir los stacks por recursos y orquestarlos desde un stack maestro?**
    
**R//=** Nested Stacks

9. **¿Si necesitas llamar el ARN de una función lambda qué función intrínseca usarías?**
    
**R//=** GetAtt

10. **¿Qué resultado tendría el siguiente código !GetAtt MyLambdaFunction.Arn ?**
    
**R//=** arn:aws:lambda:us-west-2:123456789012:MyLambdaName

11. **¿Qué función usarías para incluir en un template el número de cuenta de AWS donde se esta desplegando el template ?**
    
**R//=** !Sub + ${AWS::Region}

12. **¿Cuál de las siguientes opciones son funciones condicionales?**
    
**R//=** If, Or, And, Equals

13. **¿En qué casos automatizarías el despliegue de tu infraestructura? No**
    
**R//=** En todos los casos que necesite desplegar infraestructura en un Cloud Provider.

14. **¿Con qué servicios de AWS podemos crear pipelines para desplegar Infraestructura como código?**
    
**R//=** Codepipeline, Codebuild, CodeCommit, Cloudformation.

15. **¿Cuál servicio en AWS dentro de un pipeline podemos utilizar para crear el artefacto que desplegaremos en Cloudformation?**

**R//=** CodeBuild
    
16. **En tu rol de arquitecto de seguridad, descubriste que se están haciendo despliegues de infraestructura como código y ves en el repositorio de código de GitHub que tienen en un template una cadena de conexión a una BD donde se ve el password de la misma, ¿qué recomendarías hacer en ese caso?**
    
**R//=** Utilizar SecretsManager para ocularla y llamar el secret desde cloudformation.

17. **¿Qué medidas de seguridad tomarías en despliegues de infraestructura como código?**
    
**R//=** Almacenar los artefactos en S3 (Cifrado).

18. **El estado UPDATE_COMPLETE a qué hace referencia?**
    
**R//=** Rollback completo de un stack o Actualización completa de eliminación de recursos de un stack.

19. **¿Cuáles son las 2 formas de definir una función lambda en Cloudformation?**
    
**R//=** Function Lambda y Serverless function.

20. **Si queremos desplegar una serverless function como cuál de los siguientes sería correcto usar en Cloudformation?**
    
**R//=** Type: AWS::Serverless::Function

21. **¿Cuál propiedad es obligatoria al desplegar una función lambda como AWS::Lambda::Function?**
    
**R//=** Code

22. **En la consola cloudformation puedes crear stacks de forma gráfica?**
    
**R//=** Sí, en Designer

23. **Si tengo un stack set que desplega en 4 cuentas diferentes, ¿puedo desplegar un recurso con diferentes características en solo 1 de las cuentas?**
    
**R//=** Sí se puede usando parameters

24. **En la creación de un ROLE en Cloudformation, ¿qué opción debe ser habilitada en Cloudformation?**
    
**R//=** CAPABILITY_IAM and CAPABILITY_NAMED_IAM

25. **En un nested stack necesitamos crear 3 recursos en orden secuencial, ¿cómo podrías hacer esto??**
    
**R//=** Utilizando DependsOn

26. **¿Designer puede ser utilizado para validar templates en cuanto a estructura del JSON o YAML?**
    
**R//=** Si, utilizando la API ValidaTemplate

27. **Al crear un pipeline para una función lambda, ¿de dónde se llama el código de la función?**
    
**R//=** GitHub

28. **¿Es posible filtrar los stacks por estado FAILED?**
    
**R//=** Sí

29. **¿Cuáles funciones usarías para extraer solo 1 valor de un arreglo de datos?**
    
**R//=** Split y Select

30. **¿Cuál función utilizarías para obtener un true si alguno de los valores es falso en un arreglo?**
    
**R//=** AND