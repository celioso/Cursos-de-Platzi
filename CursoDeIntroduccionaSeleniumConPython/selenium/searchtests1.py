import unittest
from selenium import webdriver
from pyunitreport import HTMLTestRunner
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By

class SearchTests(unittest.TestCase):

    def setUp(self):
        chrome_driver_path = r"/chromedriver.exe"
        self.driver = webdriver.Chrome()
        driver = self.driver
        driver.implicitly_wait(30)
        driver.maximize_window()
        driver.get("http://demo-store.seleniumacademy.com/")

    def test_search_tee(self):
        driver = self.driver
        search_field = driver.find_element(By.NAME, "q")
        search_field.clear()

        search_field.send_keys("tee")
        search_field.submit()

    def test_search_salt_shaker(self):
        driver = self.driver
        search_field = driver.find_element(By.NAME, "q")

        search_field.send_keys("salt shaker")
        search_field.submit()

        products = driver.find_elements(By.XPATH, '//div[@class = "product-info"]/h2[@class="product-name"]/a')
        self.assertEqual(1, len(products))

    
    def tearDown(self) -> None:
        self.driver.quit()

if __name__ == "__main__":
    unittest.main(verbosity = 2)