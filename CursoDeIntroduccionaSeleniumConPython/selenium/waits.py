import unittest
from selenium import webdriver
from pyunitreport import HTMLTestRunner
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class ExplicatWaitTests(unittest.TestCase):

    def setUp(self):
        chrome_driver_path = r"/chromedriver.exe"
        self.driver = webdriver.Chrome()
        driver = self.driver
        driver.implicitly_wait(30)
        driver.maximize_window()
        driver.get("http://demo-store.seleniumacademy.com/")

    def test_account_link(self):
        WebDriverWait(self.driver, 10).until(lambda s: s.find_element(By.ID, "select-language").get_attribute("length") == "3")

        account = WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.LINK_TEXT, "ACCOUNT")))                                     
        account.click()

    def test_create_new_customer(self):
        self.driver.find_element(By.LINK_TEXT, "ACCOUNT").click()

        my_acount = WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.LINK_TEXT, "My Account")))
        my_acount.click()

        create_account_button = WebDriverWait(self.driver, 20).until(EC.element_to_be_clickable((By.LINK_TEXT, "CREATE AN ACCOUNT")))
        create_account_button.click()

        WebDriverWait(self.driver, 10).until(EC.title_contains("Create New Customer Account"))

    def tearDown(self) -> None:
        self.driver.close()

if __name__ == "__main__":
    unittest.main(verbosity = 2, testRunner = HTMLTestRunner(output = "reportes", report_name = "waits_test_report"))
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class ExplicatWaitTests(unittest.TestCase):

    def setUp(self):
        chrome_driver_path = r"/chromedriver.exe"
        self.driver = webdriver.Chrome()
        driver = self.driver
        driver.implicitly_wait(30)
        driver.maximize_window()
        driver.get("http://demo-store.seleniumacademy.com/")

    def test_account_link(self):
        WebDriverWait(self.driver, 10).until(lambda s: s.find_element(By.ID, "select-language").get_attribute("length") == "3")

        account = WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.LINK_TEXT, "ACCOUNT")))                                     
        account.click()

    def test_create_new_customer(self):
        self.driver.find_element(By.LINK_TEXT, "ACCOUNT").click()

        my_acount = WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.LINK_TEXT, "My Account")))
        my_acount.click()

        create_account_button = WebDriverWait(self.driver, 20).until(EC.element_to_be_clickable((By.LINK_TEXT, "CREATE AN ACCOUNT")))
        create_account_button.click()

        WebDriverWait(self.driver, 10).until(EC.title_contains("Create New Customer Account"))

    def tearDown(self) -> None:
        self.driver.close()

if __name__ == "__main__":
    unittest.main(verbosity = 2, testRunner = HTMLTestRunner(output = "reportes", report_name = "waits_test_report"))