import unittest
from selenium import webdriver
from pyunitreport import HTMLTestRunner
from selenium.webdriver.common.by import By
#from time import sleep

class NavigationTest(unittest.TestCase):

    def setUp(self):
        chrome_driver_path = r"/chromedriver.exe"
        self.driver = webdriver.Chrome()
        driver = self.driver
        driver.implicitly_wait(30)
        driver.maximize_window()
        driver.get("http://google.com/")
        
    def test_browser_navigation(self):
        driver = self.driver

        search_fiend = driver.find_element(By.NAME, "q")
        search_fiend.clear()
        search_fiend.send_keys("platzi")
        search_fiend.submit()

        driver.back()
        #sleep(3)
        driver.forward()
        #sleep(3)
        driver.refresh()
        #sleep(3)


    def tearDown(self) -> None:
        self.driver.close()

if __name__ == "__main__":
    unittest.main(verbosity = 2, testRunner = HTMLTestRunner(output = "reportes", report_name = "automatic_navigation_test_report"))