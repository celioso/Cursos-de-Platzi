import pytz
from datetime import datetime

from fastapi import FastAPI
from models import Customer, Invoice
from db import SessionDep, create_all_tables
from .routers import customers, transactions, invoices, plans

app = FastAPI(lifespan=create_all_tables)
app.include_router(customers.router)
app.include_router(transactions.router)
app.include_router(invoices.router)
app.include_router(plans.router)

@app.get("/")
async def root():
    return {"message":"Hola, Mario!"}

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