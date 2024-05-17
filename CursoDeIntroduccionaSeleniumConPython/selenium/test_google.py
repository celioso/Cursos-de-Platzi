import unittest
from selenium import webdriver
from pyunitreport import HTMLTestRunner
from selenium.webdriver.common.by import By
from google_page import GooglePage

class GoogleTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        chrome_driver_path = r"/chromedriver.exe"
        cls.driver = webdriver.Chrome()
        driver = cls.driver
        driver.implicitly_wait(30)
        driver.maximize_window()
        
    def test_search(self):
        google = GooglePage(self.driver)
        google.open()
        google.search("Platzi")

        self.assertEqual("Platzi", google.keyword)

    @classmethod
    def tearDown(cls) -> None:
        cls.driver.implicitly_wait(3)
        cls.driver.close()

if __name__ == "__main__":
    unittest.main(verbosity = 2, testRunner = HTMLTestRunner(output = "reportes", report_name = "select_test_google"))