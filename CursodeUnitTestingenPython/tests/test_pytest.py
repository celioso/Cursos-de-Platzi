import pytest
from src.bank_account import BankAccount


@pytest.mark.parametrize("ammount, expected", [
    (100, 1100),
    (3000, 4000),
    (4500, 5500),
])
def test_deposit_multiple_values(ammount, expected):
    account = BankAccount(balance=1000, log_file="transaction.txt")
    new_balance = account.deposit(ammount)
    assert new_balance == expected

#def test_sum():
 #   a = 4
  #  b = 4
 #   assert a + b == 8



def test_deposit_negative():
    account = BankAccount(balance=1000, log_file="transaction.txt")
    with pytest.raises(ValueError):
        account.deposit(-100)

@pytest.mark.parametrize("deposit_amount, expected_balance, raises_error", [
    (-100, 1000, ValueError),  # Caso de depósito negativo, espera excepción
    (500, 1500, None),         # Caso de depósito positivo, sin error
])
def test_deposit(deposit_amount, expected_balance, raises_error):
    account = BankAccount(balance=1000, log_file="transaction.txt")
    
    if raises_error:
        with pytest.raises(raises_error):
            account.deposit(deposit_amount)
    else:
        new_balance = account.deposit(deposit_amount)
        assert new_balance == expected_balance