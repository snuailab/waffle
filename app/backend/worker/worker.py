import os

import typer
from celery import Celery
from config import Config
from waffle_hub.hub import Hub
from waffle_utils.file import io

celery_app = Celery(
    "tasks",
    backend=f"redis://{Config.REDIS_HOST}:{Config.REDIS_PORT}/0",
    broker=f"redis://{Config.REDIS_HOST}:{Config.REDIS_PORT}/0",
)

############## task
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
def check_gpu():
    import torch

    return torch.cuda.is_available()


############## hub
@celery_app.task
def train_task(hub_name: str, hub_root_dir: str, train_args: dict):
    hub = Hub.load(hub_name, hub_root_dir)
    if hub.artifact_dir.exists():
        hub.delete_artifact()
    hub.train(**train_args)


@celery_app.task
def evaluate_task(hub_name: str, hub_root_dir: str, evaluate_args: dict):
    hub = Hub.load(hub_name, hub_root_dir)
    hub.evaluate(**evaluate_args)


@celery_app.task
def inference_task(hub_name: str, hub_root_dir: str, inference_args: dict):
    hub = Hub.load(hub_name, hub_root_dir)
    if hub.inference_dir.exists():
        io.remove_directory(hub.inference_dir, recursive=True)
    hub.inference(**inference_args)


@celery_app.task
def export_onnx_task(hub_name: str, hub_root_dir: str, export_onnx_args: dict):
    hub = Hub.load(hub_name, hub_root_dir)
    hub.export_onnx(**export_onnx_args)


@celery_app.task
def export_waffle_task(hub_name: str, hub_root_dir: str):
    hub = Hub.load(hub_name, hub_root_dir)
    hub.export_waffle()
