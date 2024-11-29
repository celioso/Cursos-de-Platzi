from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def print_hello():
    print("Hola gente de platzi")

with DAG (dag_id="pythonoperator",
        description="Nuestro primer DAG utilizando Python Operaator",
        schedule_interval="@once",
        start_date=datetime(2024, 11, 25)) as dag:
    
    t1 = PythonOperator(task_id="hello_with_python",
                        python_callable=print_hello)