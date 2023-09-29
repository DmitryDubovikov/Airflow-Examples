"""
Test documentation
"""
import time
from datetime import datetime, timedelta
from textwrap import dedent

# Для объявления DAG нужно импортировать класс из airflow
from airflow import DAG

# Операторы - это кирпичики DAG, они являются звеньями в графе
# Будем иногда называть операторы тасками (tasks)
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

with DAG(
    "another_dag",
    # Параметры по умолчанию для тасок
    default_args={
        # Если прошлые запуски упали, надо ли ждать их успеха
        "depends_on_past": False,
        # Кому писать при провале
        "email": ["airflow@example.com"],
        # А писать ли вообще при провале?
        "email_on_failure": False,
        # Писать ли при автоматическом перезапуске по провалу
        "email_on_retry": False,
        # Сколько раз пытаться запустить, далее помечать как failed
        "retries": 1,
        # Сколько ждать между перезапусками
        "retry_delay": timedelta(minutes=5),  # timedelta из пакета datetime
    },
    # Описание DAG (не тасок, а самого DAG)
    description="A simple tutorial DAG",
    # Как часто запускать DAG
    schedule_interval=timedelta(days=1),
    # С какой даты начать запускать DAG
    # Каждый DAG "видит" свою "дату запуска"
    # это когда он предположительно должен был
    # запуститься. Не всегда совпадает с датой на вашем компьютере
    start_date=datetime(2023, 1, 1),
    # Запустить за старые даты относительно сегодня
    # https://airflow.apache.org/docs/apache-airflow/stable/dag-run.html
    catchup=False,
    # теги, способ помечать даги
    tags=["example"],
) as dag:
    # t1, t2, t3 - это операторы (они формируют таски, а таски формируют даг)
    t1 = BashOperator(
        task_id="print_date",  # id, будет отображаться в интерфейсе
        bash_command="date",  # какую bash команду выполнить в этом таске
    )
    t2 = BashOperator(
        task_id="sleep",
        depends_on_past=False,  # переопределили настройку из DAG
        bash_command="sleep 5",
        retries=3,  # тоже переопределили retries (было 1)
    )

    t1.doc_md = dedent(
        """\
    #### Task Documentation
    You can document your task using the attributes `doc_md` (markdown),
    `doc` (plain text), `doc_rst`, `doc_json`, `doc_yaml` which gets
    rendered in the UI's Task Instance Details page.
    ![img](http://montcs.bloomu.edu/~bobmon/Semesters/2012-01/491/import%20soul.png)
    """
    )  # dedent - это особенность Airflow, в него нужно оборачивать всю доку

    dag.doc_md = __doc__  # Можно забрать докстрингу из начала файла вот так
    dag.doc_md = """
    This is a documentation placed anywhere
    """  # а можно явно написать
    # формат ds: 2022-12-25
    templated_command = dedent(
        """
    {% for i in range(5) %}
        echo "{{ ds }}"
        echo "{{ macros.ds_add(ds, 7)}}"
    {% endfor %}
    """
    )  # поддерживается шаблонизация через Jinja
    # https://airflow.apache.org/docs/apache-airflow/stable/concepts/operators.html#concepts-jinja-templating

    t3 = BashOperator(
        task_id="templated",
        depends_on_past=False,
        bash_command=templated_command,
    )

    def print_context(ds, **kwargs):
        """Print the Airflow context and ds variable from the context."""
        print(kwargs)
        print(ds)
        return "Whatever you return gets printed in the logs"

    t4 = PythonOperator(
        task_id="print_the_context",
        python_callable=print_context,
    )

    def my_sleeping_function(seconds):
        time.sleep(seconds)

    # генерируем таски в цикле - так тоже можно
    for i in range(5):
        task = PythonOperator(
            task_id="sleep_for_" + str(i),
            python_callable=my_sleeping_function,
            op_kwargs={"seconds": float(i) / 10},
        )

    # А вот так в Airflow указывается последовательность задач
    t1 >> [t2, t3] >> t4 >> task
    # будет выглядеть вот так
    #    -> t2 ->

    # t1 |      | -> t4 -> [tasks...]
    #    -> t3 ->
