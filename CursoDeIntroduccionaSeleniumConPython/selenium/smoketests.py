import unittest
from unittest import TestLoader, TestSuite
from pyunitreport import HTMLTestRunner
from assertions import AssertionsTest
from searchtests1 import SearchTests
from searchtests import HomePageTests

assertions_test = TestLoader().loadTestsFromTestCase(AssertionsTest)
search_test = TestLoader().loadTestsFromTestCase(SearchTests)
search1_test = TestLoader().loadTestsFromTestCase(HomePageTests)

smoke_test = TestSuite([assertions_test, search_test, search1_test])

kwargs = {
    "output":"smoke-repot"
}

runner = HTMLTestRunner(**kwargs)
runner.run(smoke_test)
