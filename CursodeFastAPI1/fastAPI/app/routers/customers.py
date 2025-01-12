from fastapi import APIRouter,status, HTTPException, Query
from sqlmodel import select

from models import Customer, CustomerCreate, CustomerPlan, CustomerUpdate, Plan, StatusEnum
from db import SessionDep

router = APIRouter()


@router.post("/customers", response_model=Customer, status_code=status.HTTP_201_CREATED,tags=['Customers'])
async def create_customer(customer_data: CustomerCreate, session: SessionDep):
    customer = Customer.model_validate(customer_data.model_dump())
    session.add(customer)
    session.commit()
    session.refresh(customer)
    return customer

@router.get("/customers/{customer_id}", response_model=Customer, tags=['Customers'])
async def read_customer(customer_id: int, session:SessionDep):
    customer_db = session.get(Customer, customer_id)
    if not customer_db :
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer doesn't exits")
    return customer_db

@router.patch("/customers/{customer_id}", response_model=Customer, status_code=status.HTTP_201_CREATED, tags=['Customers'])
async def update_customer(customer_id: int, customer_data: CustomerUpdate, session:SessionDep):
    customer_db = session.get(Customer, customer_id)
    if not customer_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Customer doesn't exits"
        )
    customer_data_dict = customer_data.model_dump(exclude_unset=True)
    customer_db.sqlmodel_update(customer_data_dict)
    session.add(customer_db)
    session.commit()
    session.refresh(customer_db)
    return customer_db

@router.delete("/customers/{customer_id}", response_model=Customer, status_code=status.HTTP_201_CREATED, tags=['Customers'])
async def delete_customer(customer_id: int, session:SessionDep):
    customer_db = session.get(Customer, customer_id)
    if not customer_db :
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer doesn't exits")
    session.delete(customer_db)
    session.commit()
    return {"status": "ok", "detail": f"Customer with ID {customer_id} was successfully deleted"}


@router.get("/customers", response_model = list[Customer], tags=['Customers'])
async def list_customer(session: SessionDep):
    return session.exec(select(Customer)).all()

'''@router.get("/customers/{id}")
async def read_customer(id: int):
    for i in db_customers:
        if i.id == id:
            return i'''

@router.post("/customers/{customer_id}/plans/{plan_id}", status_code=status.HTTP_201_CREATED, tags=['Customers'])
async def subcribe_customer_to_plan(customer_id: int, plan_id: int,  session:SessionDep, plan_status: StatusEnum = Query()):
    customer_db = session.get(Customer, customer_id)
    plan_db = session.get(Plan, plan_id)

    if not customer_db or not plan_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="the Customer or plan doesn't exist")
    customer_plan_db = CustomerPlan(plan_id=plan_db.id, customer_id=customer_db.id, status=plan_status)

    session.add(customer_plan_db)
    session.commit()
    session.refresh(customer_plan_db)
    return customer_plan_db

@router.get("/customers/{customer_id}/plans", status_code=status.HTTP_201_CREATED, tags=['Customers'])
async def subcribe_customer_to_plan(customer_id: int, session: SessionDep,  plan_status: StatusEnum = Query()):
    customer_db = session.get(Customer, customer_id)
    if not customer_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    
    query = (
        select(CustomerPlan)
        .Where(CustomerPlan.customer_id == customer_id)
        .where(CustomerPlan.status == plan_status)
        )
    
    plans = session.exec(query).all()
    return plans

