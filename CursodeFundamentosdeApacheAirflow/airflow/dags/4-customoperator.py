from airflow import DAG
from datetime import datetime
from hellooperator import HelloOperator

with DAG(dag_id="customoperator",
        description="Nuestro primer customeperator",
        start_date=datetime(2024, 12, 25)) as dag:
    
    t1 = HelloOperator(task_id="Hello",
                    name="Mario")
    