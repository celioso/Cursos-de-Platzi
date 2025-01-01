from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from datetime import datetime, date

default_args = {
'start_date': datetime(2024, 12, 20),
'end_date': datetime(2025, 1, 3)
}

def _choose(**context):

    if context["logical_date"].date() < date(2024, 12, 31):
        return "finish_22_december"
    
    return "start_1_january"

with DAG(dag_id="10-branching",
        schedule_interval="@daily",
        default_args=default_args) as dag:
    
    branching = BranchPythonOperator(task_id="branch",
                                    python_callable=_choose)

    finish_22 = BashOperator(task_id="finish_22_december", 
                            bash_command="echo 'Running {{ds}}'")

    start_1 = BashOperator(task_id="start_1_january",
                            bash_command="echo 'Running {{ds}}'")

    branching >> [finish_22, start_1]