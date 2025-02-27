from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor

with DAG(dag_id="7.3-filesensor",
        description="FileSensor",
        schedule_interval="@daily",
        start_date=datetime(2024, 1, 10),
        end_date=datetime(2025, 8, 25),
        max_active_runs=1) as dag:
    
    t1 = BashOperator(
        task_id="creating_file",
        bash_command="sleep 10 && mkdir -p /opt/airflow/tmp && touch /opt/airflow/tmp/file.txt"
    )
    
    t2 = FileSensor(
        task_id="waiting_file",
        filepath="/opt/airflow/tmp/file.txt",
        poke_interval=30,
        timeout=600
    )
    
    t3 = BashOperator(
        task_id="end_task",
        bash_command="echo 'El fichero ha llegado'"
    )

    t1 >> t2 >> t3