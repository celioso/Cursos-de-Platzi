import unittest
from selenium import webdriver
from pyunitreport import HTMLTestRunner
from selenium.webdriver.common.by import By
from time import sleep
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class AddRemoveElements(unittest.TestCase):

    def setUp(self):
        chrome_driver_path = r"/chromedriver.exe"
        self.driver = webdriver.Chrome()
        driver = self.driver
        driver.implicitly_wait(30)
        driver.maximize_window()
        driver.get("https://the-internet.herokuapp.com/")
        driver.find_element(By.LINK_TEXT, "Add/Remove Elements").click()

    def test_add_remove(self):
        driver = self.driver

        elements_added = int(input("How many elements will you add?: "))
        elements_removed = int(input("How many elements will you remove?: "))
        total_elements = elements_added - elements_removed

        add_button = driver.find_element(By.XPATH,'//*[@id="content"]/div/button')

        sleep(3)

        for i in range(elements_added):
            add_button.click()

        for i in range(elements_removed):
            try:
                delete_button = driver.find_element(By.XPATH, '//*[@id="elements"]/button[1]')
                delete_button.click()
            except:
                print("You're trying to delete more elements the existent")
                break

        if total_elements > 0:
            print(f"There are {total_elements} elements on screen")
        else:
            print("There 0 are elements on screen")

        sleep(3)

    def tearDown(self) -> None:
        self.driver.close()

if __name__ == "__main__":
    unittest.main(verbosity = 2, testRunner = HTMLTestRunner(output = "reportes", report_name = "add_remove_selements_test_report"))
