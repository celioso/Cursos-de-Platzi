import unittest
from pyunitreport import HTMLtestRunner
from selenium import webdriver

class HelloWorld(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()
    
    def test_hello_world(self):
        pass

    def tearDown(self):
        return super().tearDown()
    
if __name__ == "__main__":
    unittest.main(verbosity=2, testRunner=HTMLtestRunner(output="reportes", report_name = "hello-world-report"))