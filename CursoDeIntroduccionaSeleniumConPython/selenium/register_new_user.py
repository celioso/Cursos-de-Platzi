import unittest
from selenium import webdriver
from pyunitreport import HTMLTestRunner
from selenium.webdriver.common.by import By

class RegisterNewUserTests(unittest.TestCase):

    def setUp(self):
        chrome_driver_path = r"/chromedriver.exe"
        self.driver = webdriver.Chrome()
        driver = self.driver
        driver.implicitly_wait(15)
        driver.maximize_window()
        driver.get("http://demo-store.seleniumacademy.com/")
        
    def test_new_user(self):
        driver = self.driver
        driver.find_element(By.XPATH, '//*[@id="header"]/div/div[2]/div/a/span[2]').click()
        driver.find_element(By.LINK_TEXT,"Log In" ).click()
        create_account_button = driver.find_element(By.XPATH, '//*[@id="login-form"]/div/div[1]/div[2]/a/span/span')
        self.assertTrue(create_account_button.is_displayed() and create_account_button.is_enabled())
        create_account_button.click()

        self.assertEqual("Create New Customer Account", driver.title)

        first_name = driver.find_element(By.ID, "firstname")
        middle_name = driver.find_element(By.ID, "middlename")
        last_name = driver.find_element(By.ID, "lastname")
        email_address = driver.find_element(By.ID, "email_address")
        password = driver.find_element(By.ID, "password")
        confirm_password = driver.find_element(By.ID, "confirmation")
        news_letter_subscription = driver.find_element(By.ID, "is_subscribed")
        submit_button = driver.find_element(By.XPATH, '//*[@id="form-validate"]/div[2]/button/span/span')


        validate_from_list = [first_name.is_enabled(),
        middle_name.is_enabled(),
        last_name.is_enabled(),
        email_address.is_enabled(),
        password.is_enabled(),
        confirm_password.is_enabled(), 
        news_letter_subscription.is_enabled(),
        submit_button.is_enabled()]

        self.assertTrue(all(validate_from_list))

        first_name.send_keys("Camilo")
        middle_name.send_keys("Andres")
        last_name.send_keys("Torres")
        email_address.send_keys("camiloandres@hotmail.com")
        password.send_keys("1234567890")
        confirm_password.send_keys("1234567890")
        news_letter_subscription.click()
        submit_button.click()

        
    def tearDown(self) -> None:
        self.driver.quit()

if __name__ == "__main__":
    unittest.main(verbosity = 2, testRunner = HTMLTestRunner(output = "reportes", report_name = "register_new_user_test_report"))


# Codigo de ChatGPT
    '''
import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class NewUserRegistrationTest(unittest.TestCase):

    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.get("URL_DEL_SITIO_WEB")

    def test_new_user_registration(self):
        driver = self.driver

        # Hacer clic en el enlace de inicio de sesión
        login_link = driver.find_element(By.LINK_TEXT, "Log In")
        login_link.click()

        # Hacer clic en el botón de crear cuenta
        create_account_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="login-form"]/div/div[1]/div[2]/a/span/span'))
        )
        create_account_button.click()

        # Verificar que los campos del formulario estén visibles y habilitados
        fields = ["firstname", "middlename", "lastname", "email_address", "password", "confirmation", "is_subcribed"]
        for field_id in fields:
            field = driver.find_element(By.ID, field_id)
            self.assertTrue(field.is_displayed() and field.is_enabled(),
                            f"Campo {field_id} no está visible o habilitado.")

        # Llenar el formulario de registro
        driver.find_element(By.ID, "firstname").send_keys("Camilo")
        driver.find_element(By.ID, "middlename").send_keys("Andres")
        driver.find_element(By.ID, "lastname").send_keys("Torres")
        driver.find_element(By.ID, "email_address").send_keys("camiloandres@hotmail.com")
        driver.find_element(By.ID, "password").send_keys("1234567890")
        driver.find_element(By.ID, "confirmation").send_keys("1234567890")

        # Hacer clic en el botón de enviar
        submit_button = driver.find_element(By.XPATH, '//*[@id="form-validate"]/div[2]/button/span/span')
        submit_button.click()

        # Esperar hasta que la página de registro se cargue completamente
        WebDriverWait(driver, 10).until(
            EC.title_contains("Create New Customer Account")
        )

        # Verificar que se haya creado la cuenta exitosamente (puedes agregar más verificaciones aquí)

    def tearDown(self):
        self.driver.quit()

if __name__ == "__main__":
    unittest.main()

    '''