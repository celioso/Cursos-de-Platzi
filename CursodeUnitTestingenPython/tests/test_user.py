import unittest, os
from faker import Faker

from src.user import User
from src.bank_account import BankAccount

class UserTest(unittest.TestCase):

    def setUp(self) -> None:
        self.faker = Faker(locale="es")
        self.user = User(self.faker.name(), self.faker.email())

    def test_user_creation(self):
        name_generated = self.faker.name()
        email_generated = self.faker.email()
        user = User(name=name_generated, email=email_generated)
        self.assertEqual(user.name, name_generated)
        self.assertEqual(user.email, email_generated)

    def test_user_with_multiple_account(self):
        user = User(self.faker.name(), self.faker.email())

        for _ in range(3):
            bank_account = BankAccount(
                balance = self.faker.random_int(min=100, max=2000, step=50),
                log_file=self.faker.file_name(extension=".txt")
            )
            self.user.add_account(account=bank_account)

        expected_value = self.user.get_total_balance()
        value = sum(account.get_balance() for account in self.user.accounts)
        self.assertEqual(value, expected_value)

    def tearDown(self) -> None:
        for account in self.user.accounts:
            os.remove(account.log_file)