import io
import json
import logging
import numpy as np
import pandas as pd
import pickle
import functools

from typing import Literal, Dict, Any

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
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
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

models = {
    "rf": RandomForestRegressor(),
    "lr": LinearRegression(),
    "hgb": HistGradientBoostingRegressor(),
}


def init() -> Dict[str, Any]:
    metrics = {
        "start_timestamp": datetime.now().strftime("%Y_%m_%d %H:%M:%S"),
    }
    _LOG.info("DAG started.")
    return metrics


def get_data_from_postgres(**kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="init")

    # закомментируем, т.к. ранее уже положили датасет на S3

    # engine = create_engine(conf.get("database", "sql_alchemy_conn"))
    # df = pd.read_sql_query("SELECT * FROM california_housing", engine)
    # _LOG.info(df.head())
    #
    # s3_hook = S3Hook("s3_connection")
    # session = s3_hook.get_session(s3_hook.conn_config.region_name)
    # resource = session.resource("s3", endpoint_url=s3_hook.conn_config.endpoint_url)
    #
    # pickle_byte_obj = pickle.dumps(df)
    # resource.Object(BUCKET, DATA_PATH).put(Body=pickle_byte_obj)

    _LOG.info("Data download completed.")

    return metrics


def prepare_data(**kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="read_data")

    _LOG.info("Data preparation completed.")

    _LOG.info(f"metrics: {metrics}")

    return metrics


def train_model(model_name, **kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="prepare_data")

    s3_hook = S3Hook("s3_connection")

    data = {}

    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(key=f"dataset/{name}.pkl", bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    model = models[model_name]
    metrics["train_start"] = datetime.now().strftime("%Y_%m_%d %H:%M:%S")

    model.fit(data["X_train"], data["y_train"])
    prediction = model.predict(data["X_test"])
    metrics["train_end"] = datetime.now().strftime("%Y_%m_%d %H:%M:%S")

    metrics["r2_score"] = r2_score(data["y_test"], prediction)
    metrics["rmse"] = mean_squared_error(data["y_test"], prediction) ** 0.5
    metrics["mae"] = median_absolute_error(data["y_test"], prediction)

    _LOG.info("Models training completed.")

    session = s3_hook.get_session(s3_hook.conn_config.region_name)
    resource = session.resource("s3", endpoint_url=s3_hook.conn_config.endpoint_url)

    date = datetime.now().strftime("%Y_%m_%d_%H_%M")
    json_byte_obj = json.dumps(metrics)
    resource.Object(BUCKET, f"results/{model_name}_{date}.json").put(Body=json_byte_obj)

    _LOG.info("Success. Results saved.")


with DAG(
    "california_multy_models",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS,
) as dag:
    task_init = PythonOperator(task_id="init", python_callable=init)

    task_get_data = PythonOperator(
        task_id="read_data",
        python_callable=get_data_from_postgres,
        provide_context=True,
    )

    task_prepare_data = PythonOperator(
        task_id="prepare_data", python_callable=prepare_data, provide_context=True
    )

    task_train_model_rf = PythonOperator(
        task_id="task_train_model_rf",
        python_callable=functools.partial(train_model, model_name="rf"),
        provide_context=True,
    )

    task_train_model_lr = PythonOperator(
        task_id="task_train_model_lr",
        python_callable=functools.partial(train_model, model_name="lr"),
        provide_context=True,
    )

    task_train_model_hgb = PythonOperator(
        task_id="task_train_model_hgb",
        python_callable=functools.partial(train_model, model_name="hgb"),
        provide_context=True,
    )

    (
        task_init
        >> task_get_data
        >> task_prepare_data
        >> [task_train_model_rf, task_train_model_lr, task_train_model_hgb]
    )
