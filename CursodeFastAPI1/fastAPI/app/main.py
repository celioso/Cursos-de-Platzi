from typing import Annotated
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import pytz
import time
import zoneinfo
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Request, status
from models import Customer, Invoice
from db import SessionDep, create_all_tables
from .routers import customers, transactions, invoices, plans

app = FastAPI(lifespan=create_all_tables)
app.include_router(customers.router)
app.include_router(transactions.router)
app.include_router(invoices.router)
app.include_router(plans.router)

@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"Request: {request.url} completed in: {process_time:.4f} seconds")

    return response

@app.middleware("http") 
async def log_request_headers(request: Request, call_next):
    
    print("Request Headers:")
    for header, value in request.headers.items():
        print(f"{header}: {value}")

    response = await call_next(request) 

    return response

security = HTTPBasic()

@app.get("/")
async def root(credentails: Annotated[HTTPBasicCredentials, Depends(security)]):
    print(credentails)
    if credentails.username == "Haru" and credentails.password == "hola":
        return {"message":f"Hola, {credentails.username}!"}
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

country_timezones = {
    "CO": "America/Bogota",
    "MX": "America/Mexico_City",
    "AR": "America/Argentina/Buenos_Aires",
    "BR": "America/Sao_Paulo",
    "PE": "America/Lima"
}

@app.get("/time/{iso_code}/{format}")
async def get_time_by_iso_code(iso_code: str, format: int):
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