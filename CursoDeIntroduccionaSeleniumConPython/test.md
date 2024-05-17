### Curso de Introducción a Selenium con Python

1. **### Curso de Introducción a Selenium con Python**
**R/:** 

2. **¿Cuáles son los componentes vigentes de la suite de Selenium?**
**R/:** Selenium WebDriver, Selenium IDE y Selenium Grid

3. **¿Qué lenguaje no es soportado oficialmente con Selenium?**
**R/:** Dart

4. **Son debilidades de Selenium:**
**R/:** 

5. **¿Qué librerías complementan Selenium para generar pruebas efectivas?**
**R/:** Unittest, PyUnitReport, DDT

6. **Los métodos setUp() y tearDown() son para:**
**R/:** Realizar acciones específicas antes y después de los casos de prueba.

7. **¿Cuándo es buena idea usar XPath como selector?**
**R/:** Cuando no hay otro selector único para interactuar con el elemento.

8. **Tienes una barra de búsqueda cuyo nombre es name="q" ¿Con qué código accedes a esta?**
**R/:** driver.find_element_by_name('q')

9. **¿Qué assertion te permite validar el que el título del sitio web es el siguiente? 🚀Platzi: ‎Cursos Online Profesionales de Tecnología**
**R/:** self.assertEqual('🚀Platzi: ‎Cursos Online Profesionales de Tecnología', self.driver.title)

10. **¿Qué es y para qué nos sirven las test suites?**
**R/:** 

11. **¿Con qué me permite interactuar la clase WebDriver de Selenium?**
**R/:** Con el navegador mismo: su ventana, alerts, pop-ups y navegación.

12. **¿Con qué me permite interactuar la clase WebElement de Selenium?**
**R/:** Con los elementos del sitio web: checkbox, textbox, dropdown, radiobutton, etc.

13. **¿Cómo valido que el botón con nombre "signup" está a la vista y habilitado?**
**R/:** signup_button = driver.find_element_by_name('signup') self.assertTrue(signup_button.is_displayed() and signup_button.is_enabled())

14. **¿Cómo extraemos el valor del atributo ‘autocomplete’ del siguiente elemento?**
`<input type="search" name="search" placeholder="Search Wikipedia" title="Search Wikipedia [ctrl-option-f]" accesskey="f" id="searchInput" autocomplete="off">`
**R/:** search_bar = driver.find_element_by_name('search') search_bar.get_attribute('autocomplete')

15. **¿Qué hace el siguiente código?**
`username.send_keys('user123')`
`username.send_keys(KEYS.ENTER)`
**R/:** Introduce el texto 'user123' en el elemento de la variable 'username' y después "presiona" la tecla "ENTER"

16. **¿Qué hace el siguiente código?**
`select_amount = Select(driver.find_element_by_name('amount'))`
`select_amount.select_by_value('3')`
**R/:** Busca al elemento con nombre 'amount' y selecciona la opción cuyo valor sea igual a "3"

17. **¿Qué acciones podemos utilizar para interactuar con un alert de JavaScript?**
**R/:** Aceptar, rechazar, extraer texto y enviar texto

18. **Son todos métodos para automatizar la navegación:**
**R/:**  

19. **¿Qué hace el siguiente código?**
`driver.implicitly_wait(20)`
**R/:**  Selenium espera hasta 20 segundos a que cargue algún elemento para continuar

20. **¿Por qué debemos utilizar la menor cantidad de esperas implícitas posibles?**
**R/:**  Porque la suma de los tiempos hace que la prueba sea mucho más lenta

21. **¿Qué hace el siguiente código?**
`account = WebDriverWait(self.driver, 10).until(expected_conditions.visibility_of_element_located((By.LINK_TEXT, “ACCOUNT”)))`
`account.click()`
**R/:**  Espera hasta 10 segundos a que sea visible el elemento que incluye el texto ‘ACCOUNT’ en su link y después hace clic en él.

22. **¿Qué es una expected condition (condición esperada)?**
**R/:**  Condiciones predefinidas o personalizadas a las que el script espera se cumplan antes de continuar

23. **¿Cuándo es conveniente utilizar try y except en nuestra prueba?**
**R/:**  Cuando no conocemos cómo funciona el sitio por debajo y/o trabajamos con contenido dinámico

24. **¿Cuál es la diferencia entre DDT y TDD?**
**R/:**  DDT es testing basado en código escrito. TDD es código basado en pruebas para pasarlas positivamente.

25. **¿Cuál es el principal beneficio de Page Object Model (POM)?**
**R/:**  Permite un mejor mantenimiento de las pruebas a largo plazo y facilita su legibilidad

26. **¿Cuáles son consideraciones al presentar una prueba técnica?**
**R/:**  Tener claro los pasos a seguir y pensar como el usuario final

27. **¿Por qué no debería automatizar o hacer testing en sitios que explícitamente lo prohíben?**
**R/:**  Por respeto a su autor y cómo parte de la ética profesional