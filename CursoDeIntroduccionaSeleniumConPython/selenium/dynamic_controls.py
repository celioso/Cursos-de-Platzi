import unittest
from selenium import webdriver
from pyunitreport import HTMLTestRunner
from selenium.webdriver.common.by import By
#from time import sleep
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class DynamicControls(unittest.TestCase):

    def setUp(self):
        chrome_driver_path = r"/chromedriver.exe"
        self.driver = webdriver.Chrome()
        driver = self.driver
        driver.implicitly_wait(30)
        driver.maximize_window()
        driver.get("https://the-internet.herokuapp.com/")
        driver.find_element(By.LINK_TEXT, "Dynamic Controls").click()

    def test_dynamic_controls(self):
        driver = self.driver

        checkbox = driver.find_element(By.CSS_SELECTOR, '#checkbox > input[type=checkbox]')
        checkbox.click()

        remove_add_button = driver.find_element(By.CSS_SELECTOR, '#checkbox-example > button')
        remove_add_button.click()

        WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#checkbox-example > button')))
        remove_add_button.click()

        enable_disable_button = driver.find_element(By.CSS_SELECTOR, '#input-example > button')
        enable_disable_button.click()

        WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#input-example > button')))

        text_area = driver.find_element(By.CSS_SELECTOR, '#input-example > input[type=text]')
        text_area.send_keys('Platzi')

        enable_disable_button.click()



    def tearDown(self) -> None:
        self.driver.close()

if __name__ == "__main__":
    unittest.main(verbosity = 2, testRunner = HTMLTestRunner(output = "reportes", report_name = "dynamic_controls_test_report"))
