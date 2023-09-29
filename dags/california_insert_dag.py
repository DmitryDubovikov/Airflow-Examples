import io
import json
import logging
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.configuration import conf
from sqlalchemy import create_engine


DEFAULT_ARGS = {
    "owner": "Dmitry Dubovikov",
    "email": "dmitry.dubovikov@gmail.com",
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())


def init() -> None:
    _LOG.info("DAG started.")

    engine = create_engine(conf.get("database", "sql_alchemy_conn"))
    df = pd.read_sql_query("SELECT * FROM california_housing", engine)
    _LOG.info(df.head())


def insert_data_to_postgres() -> None:
    # Получим датасет California housing
    data = fetch_california_housing()

    # Объединим фичи и таргет в один np.array
    dataset = np.concatenate(
        [data["data"], data["target"].reshape([data["target"].shape[0], 1])], axis=1
    )

    # Преобразуем в dataframe.
    dataset = pd.DataFrame(
        dataset, columns=data["feature_names"] + data["target_names"]
    )

    # Создадим подключение к базе данных postgres.
    engine = create_engine(conf.get("database", "sql_alchemy_conn"))

    # Сохраним датасет в базу данных
    dataset.to_sql("california_housing", engine)

    # Для проверки можно сделать:
    df = pd.read_sql_query("SELECT * FROM california_housing", engine)

    _LOG.info(df.head())


with DAG(
    "california_insert",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS,
) as dag:
    task_init = PythonOperator(task_id="init", python_callable=init)

    task_insert_data_to_postgres = PythonOperator(
        task_id="insert", python_callable=insert_data_to_postgres
    )

    task_init >> task_insert_data_to_postgres
