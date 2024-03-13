import os

import typer
from celery import Celery
from config import Config
from waffle_hub.hub import Hub

celery_app = Celery(
    "tasks",
    backend=f"redis://{Config.REDIS_HOST}:{Config.REDIS_PORT}/0",
    broker=f"redis://{Config.REDIS_HOST}:{Config.REDIS_PORT}/0",
)


@celery_app.task
def task_long(n: int):
    import time

    print("Strat")
    if n == -1:
        time.sleep(5)
        print("Fail")
        raise ValueError("Error")
    for i in range(n):
        print(f"Processing {i}")
        time.sleep(1)
    print("End")
    return {
        "state": "success",
    }


@celery_app.task
def train(train_args: dict):
    print(train_args)
    import time

    time.sleep(5)
    return train_args
