from sqlmodel import Session

from db import engine
from models import Customer, Transaction

session = Session(engine)
customer = Customer(
    name = "Camilo",
    middlename = "Andres",
    last_name = "Restrepo",
    national_id_card = 54896321,
    description = "Cato",
    email = "camilorestrepo@hotmail.com",
    age = 32,
)
session.add(customer)
session.commit()

for x in range(100):
    session.add(
        Transaction(
            customer_id=customer.id,
            description=f"Test number {x}",
            ammount=10 * x,
        )
    )
session.commit()