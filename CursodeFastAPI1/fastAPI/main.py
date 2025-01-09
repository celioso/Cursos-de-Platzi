import pytz
from datetime import datetime

from fastapi import FastAPI
from models import Customer, CustomerCreate, Transaction, Invoice, CustomerCreate
from db import SessionDep

app = FastAPI()

@app.get("/")
async def root():
    return {"message":"Hola, Luis!"}

country_timezones = {
    "CO": "America/Bogota",
    "MX": "America/Mexico_City",
    "AR": "America/Argentina/Buenos_Aires",
    "BR": "America/Sao_Paulo",
    "PE": "America/Lima"
}

@app.get("/time/{iso_code}/{format}")
async def time(iso_code: str, format: int):
    iso = iso_code.upper()
    timezone_str = country_timezones.get(iso)
    tz = pytz.timezone(timezone_str)
    date = datetime.now(tz)
    
    if format == 12:
        formatted_date = date.strftime("%Y-%m-%d %I:%M:%S %p")
    else:
        formatted_date = date.strftime("%Y-%m-%d %H:%M:%S")
    
    return {"time": formatted_date}

db_customers: list[Customer] = []

@app.post("/customers", response_model=Customer)
async def create_customer(customer_data: CustomerCreate, session: SessionDep):
    customer = Customer.model_validate(customer_data.model_dump())
    # Ausmiendo que hace base de datos
    customer.id = len(db_customers)
    db_customers.append(customer)
    return customer

@app.get("/customers", response_model = list[Customer])
async def list_customer():
    return db_customers

@app.get("/customers/{id}")
async def read_customer(id: int):
    for i in db_customers:
        if i.id == id:
            return i

@app.post("/transactions")
async def create_transaction(transaction_data: Transaction):
    return transaction_data

@app.post("/invoices")
async def create_Invoice(invoice_data: Invoice):
    return invoice_data