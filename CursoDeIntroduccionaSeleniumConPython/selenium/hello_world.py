import unittest
from pyunitreport import HTMLTestRunner
from selenium import webdriver

class HelloWorld(unittest.TestCase):

    @classmethod
    def setUp(cls):
        chrome_driver_path = r"/chromedriver.exe"
        cls.driver = webdriver.Chrome()
        driver = cls.driver
        driver.implicitly_wait(20)

    def test_hello_world(self):
        driver = self.driver
        driver.get("https://www.platzi.com")

    def test_visit_wikipedia(self):
        self.driver.get("https://www.wikipedia.org")

    def test_visit_google(self):
        self.driver.get("https://www.google.com.co")

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()
        
if __name__ == "__main__":
    unittest.main(verbosity = 2, testRunner = HTMLTestRunner(output= "reportes",report_name= "helllo_world_report"))