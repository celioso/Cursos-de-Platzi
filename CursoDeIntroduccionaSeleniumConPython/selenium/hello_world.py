import unittest 
from pyunitreport import HTMLTestRunner
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

class HelloWorld(unittest.TestCase):
    def setUp(self) -> None:
        options = Options()
        self.driver = webdriver.Chrome(service=Service(r"C:\Users\celio\OneDrive\Escritorio\programación\platzi\CursoDeIntroduccionaSeleniumConPython\selenium\chromedriver.exe"), options=options)
        self.driver.implicitly_wait(10)
    
    def test_hello_world(self):
        driver = self.driver
        driver.get("https://www.platzi.com")

    def tearDown(self):
        self.driver.quit()

if __name__ == '__main__':
    unittest.main(verbosity = 2, testRunner = HTMLTestRunner(output = 'Reports', report_name = 'Hello_world_report'))