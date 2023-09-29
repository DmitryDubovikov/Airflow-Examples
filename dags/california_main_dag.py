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

BUCKET = "dda-mlops"
DATA_PATH = "datasets/california_housing.pkl"

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"

DEFAULT_ARGS = {
    "owner": "Dmitry Dubovikov",
    "email": "dmitry.dubovikov@gmail.com",
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())


def init() -> None:
    _LOG.info("DAG started.")


def get_data_from_postgres() -> None:
    # pg_hook = PostgresHook("pg_connection")
    # conn = pg_hook.get_conn()
    # df = pd.read_sql_query("SELECT * FROM california_housing", conn)

    engine = create_engine(conf.get("database", "sql_alchemy_conn"))
    df = pd.read_sql_query("SELECT * FROM california_housing", engine)
    _LOG.info(df.head())

    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("ru-central1")  # ru-central1
    resource = session.resource("s3")

    pickle_byte_obj = pickle.dumps(df)
    resource.Object(BUCKET, DATA_PATH).put(Body=pickle_byte_obj)

    _LOG.info("Data download completed.")


def prepare_data() -> None:
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(key=DATA_PATH, bucket_name=BUCKET)
    df = pd.read_pickle(file)

    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.fit_transform(X_test)

    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        X_train_fitted,
        X_test_fitted,
        y_train,
        y_test,
    ):
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET, f"dataset/{name}.pkl").put(Body=pickle_byte_obj)

    _LOG.info("Data preparation completed.")


def train_model() -> None:
    s3_hook = S3Hook("s3_connection")
    data = {}

    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(key=f"dataset/{name}.pkl", bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    model = RandomForestRegressor()
    model.fit(data["X_train"], data["y_train"])
    prediction = model.predict(data["X_test"])

    result = {
        "r2_score": r2_score(data["y_test"], prediction),
        "rmse": mean_squared_error(data["y_test"], prediction) ** 0.5,
        "mae": median_absolute_error(data["y_test"], prediction),
    }

    date = datetime.now().strftime("%Y_%m_%d_%H")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")
    json_byte_obj = json.dumps(result)
    resource.Object(BUCKET, f"results/{date}.json").put(Body=json_byte_obj)

    _LOG.info("Model training completed.")


def save_results() -> None:
    _LOG.info("Success. Results saved.")


def all_in_one() -> None:
    engine = create_engine(conf.get("database", "sql_alchemy_conn"))
    df = pd.read_sql_query("SELECT * FROM california_housing", engine)

    ##############################################
    # prepare data

    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.fit_transform(X_test)

    ##############################################
    # train model

    model = RandomForestRegressor()
    model.fit(X_train_fitted, y_train)
    prediction = model.predict(X_test_fitted)

    result = {
        "r2_score": r2_score(y_test, prediction),
        "rmse": mean_squared_error(y_test, prediction) ** 0.5,
        "mae": median_absolute_error(y_test, prediction),
    }

    _LOG.info(result)


with DAG(
    "california_main",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS,
) as dag:
    task_init = PythonOperator(task_id="init", python_callable=init)

    # task_get_data = PythonOperator(
    #     task_id="read_data", python_callable=get_data_from_postgres
    # )
    #
    # task_prepare_data = PythonOperator(task_id="prepare", python_callable=prepare_data)
    #
    # task_train_model = PythonOperator(
    #     task_id="train_model", python_callable=train_model
    # )
    #
    # task_save_results = PythonOperator(
    #     task_id="save_results", python_callable=save_results
    # )
    #
    # (
    #     task_init
    #     >> task_get_data
    #     >> task_prepare_data
    #     >> task_train_model
    #     >> task_save_results
    # )

    task_all_in_one = PythonOperator(task_id="all_in_one", python_callable=all_in_one)

    task_init >> task_all_in_one
