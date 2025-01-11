from models import Invoice
from fastapi import APIRouter

router = APIRouter()

@router.post("/invoices", tags=["Invoices"])
async def create_Invoice(invoice_data: Invoice):
    breakpoint()
    return invoice_data