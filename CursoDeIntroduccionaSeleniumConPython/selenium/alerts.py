import unittest
from selenium import webdriver
from pyunitreport import HTMLTestRunner
from selenium.webdriver.common.by import By

class CompareProducts(unittest.TestCase):

    def setUp(self):
        chrome_driver_path = r"/chromedriver.exe"
        self.driver = webdriver.Chrome()
        driver = self.driver
        driver.implicitly_wait(30)
        driver.maximize_window()
        driver.get("http://demo-store.seleniumacademy.com/")
        
    def test_compare_products_removal_alert(self):
        driver = self.driver
        search_field = driver.find_element(By.NAME, "q")
        search_field.clear()

        search_field.send_keys("tee")
        search_field.submit()

        driver.find_element(By.CLASS_NAME, "link-compare").click()
        driver.find_element(By.LINK_TEXT, 'Clear All').click()

        alert = driver.switch_to.alert
        alert_text = alert.text

        self.assertEqual("Are you sure you would like to remove all products from your comparison?", alert_text)

        alert.accept()

    def tearDown(self) -> None:
        self.driver.implicitly_wait(3)
        self.driver.close()

if __name__ == "__main__":
    unittest.main(verbosity = 2, testRunner = HTMLTestRunner(output = "reportes", report_name = "alerts_test_report"))