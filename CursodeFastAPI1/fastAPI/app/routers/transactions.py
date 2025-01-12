from fastapi import APIRouter, HTTPException, Query, status
from sqlmodel import select

from db import SessionDep
from models import Customer, Transaction, TransactionCreate

router = APIRouter()

@router.post("/transactions", status_code=status.HTTP_201_CREATED, tags=["Transactions"])
async def create_transaction(transaction_data: TransactionCreate, session: SessionDep):
    transaction_data_dict = transaction_data.model_dump()
    customer = session.get(Customer, transaction_data_dict.get('customer_id'))
    if not customer:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer doesn't exit")
    transaction_db = Transaction.model_validate(transaction_data_dict)
    session.add(transaction_db)
    session.commit()
    session.refresh(transaction_db)

    return transaction_data

@router.get("/transactions", tags=["Transactions"])
async def list_transaction(session: SessionDep, skip: int = Query(0, description="Registros a omitir"), limit: int = Query(10, description = "Número de registros")):
    query = select(Transaction).offset(skip).limit(limit)
    len_transations = len(session.exec(select(Transaction)).all())
    transactions = session.exec(query).all()
    
    total_pages = len_transations // limit
    current_page = (skip // limit) + 1
    
    message = {"total_pages": total_pages, "current_page": current_page}
    
    return {"transactions":transactions, "message":message}