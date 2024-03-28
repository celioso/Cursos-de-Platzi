import unittest
from selenium import webdriver
from pyunitreport import HTMLTestRunner
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

class LenguageOptions(unittest.TestCase):

    def setUp(self):
        chrome_driver_path = r"/chromedriver.exe"
        self.driver = webdriver.Chrome()
        driver = self.driver
        driver.implicitly_wait(30)
        driver.maximize_window()
        driver.get("http://demo-store.seleniumacademy.com/")
        
    def test_select_language(self):
        exp_option = ["English", "French", "German"]
        act_option = []

        select_language = Select(self.driver.find_element(By.ID, "select-language"))

        self.assertEqual(3, len(select_language.options))

        for option in select_language.options:
            act_option.append(option.text)

        self.assertListEqual(exp_option, act_option)

        self.assertEqual("English", select_language.first_selected_option.text)

        select_language.select_by_visible_text("German")

        self.assertTrue("store=german" in self.driver.current_url)

        select_language = Select(self.driver.find_element(By.ID, "select-language"))
        select_language.select_by_index(0)

    def tearDown(self) -> None:
        self.driver.implicitly_wait(3)
        self.driver.close()

if __name__ == "__main__":
    unittest.main(verbosity = 2, testRunner = HTMLTestRunner(output = "reportes", report_name = "select_lenguage_test_report"))