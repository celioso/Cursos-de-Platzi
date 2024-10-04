import unittest
from src.calculator import sum, subtract, multiplication, division


class CalculatorTests(unittest.TestCase):

    def test_sum(self):
        assert sum(2, 3) == 5

    def test_subtract(self):
        assert subtract(10, 5) == 5

    def test_multiplication(self):
        assert multiplication(10, 5) == 50

    def test_division(self):
        assert division(10, 5) == 2

    def test_div_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            division(3, 0)