from asyncio import sleep
import unittest
from selenium import webdriver
from pyunitreport import HTMLTestRunner
from selenium.webdriver.common.by import By

class TestingMercadolibre(unittest.TestCase):

    def setUp(self):
        chrome_driver_path = r"/chromedriver.exe"
        self.driver = webdriver.Chrome()
        driver = self.driver
        driver.implicitly_wait(30)
        driver.maximize_window()
        driver.get("http://mercadolibre.com")
        
    def test_search_ps4(self):
        driver = self.driver

        country = driver.find_element(By.ID, "CO")
        country.click()

        search_field = driver.find_element(By.NAME, "as_word")
        search_field.click()
        search_field.clear()
        search_field.send_keys("Playstatin 5")
        search_field.submit()
        sleep(3)

        location = driver.find_element(By.CLASS_NAME, 'ui-search-filter-name')
        location.click()
        sleep(3)

        condition = driver.find_element(By.PARTIAL_LINK_TEXT, "Nuevo")
        condition.click()
        sleep(3)

        order_menu = driver.find_element(By.XPATH, '/html/body/main/div/div[3]/section/div[2]/div[2]/div/div/div[2]/div/div/button/span[2]')
        sleep(3)
        order_menu.click()
        higher_price = driver.find_element(By.CSS_SELECTOR, '#\:R2m55e6\:-menu-list-option-price_desc > div > div > span')
        higher_price.click()
        sleep(3)

        articles = []
        prices = []

        for i in range(5):
            article_name = driver.find_element(By.XPATH, f'//li[{i + 1}]//h2').text

            articles.append(article_name)
            
            article_price = driver.find_element(By.XPATH, f'//li[{i + 1}]//span[contains(@class, "andes-money-amount__fraction")]').text
            prices.append(article_price)

        print(articles, prices)

        # otra solucion

        '''for i in range(8):
            article_name = driver.find_element(By.XPATH, f'//li[{i + 1}]//h2').text
            articles.append(article_name)
            article_price = driver.find_element(By.XPATH, f'//li[{i + 1}]//span[contains(@class, "andes-money-amount__fraction")]').text
            prices.append(article_price)

        for i in range(8):
            print(articles[i], "| Precio:", prices[i])'''



    def tearDown(self) -> None:
        self.driver.implicitly_wait(3)
        self.driver.close()

if __name__ == "__main__":
    unittest.main(verbosity = 2, testRunner = HTMLTestRunner(output = "reportes", report_name = "select_mercaadolibre"))