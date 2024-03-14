import os
import time
from tempfile import NamedTemporaryFile

import torch
from src.schema.task import TaskType
from src.service.api_service import api_service
from waffle_hub.hub import Hub
from waffle_hub.schema.running_status import (
    EvaluatingStatus,
    ExportingOnnxStatus,
    ExportingWaffleStatus,
    InferencingStatus,
    TrainingStatus,
)
from waffle_utils.file import io


def get_parse_root_dir():
    return Hub.parse_root_dir(os.getenv("WAFFLE_HUB_ROOT_DIR", None))


def get_hub_list(root_dir) -> list[str]:
    return Hub.get_hub_list(root_dir=root_dir)


def get_available_backends() -> list[str]:
    return list({str(back).upper() for back in Hub.get_available_backends()})


def get_available_tasks(backend: str) -> list[str]:
    return Hub.get_available_tasks(backend=backend)


def get_available_model_types(backend: str, task: str) -> list[str]:
    return Hub.get_available_model_types(backend=backend, task=task)


def get_available_model_sizes(backend: str, task: str, model_type: str) -> list[str]:
    return Hub.get_available_model_sizes(backend=backend, task=task, model_type=model_type)


def get_model_config_dict(hub: Hub) -> dict:
    if hub is not None:
        return hub.get_model_config().to_dict()
    else:
        return None


def get_category_names(hub: Hub) -> list[str]:
    if hub is not None:
        return hub.get_category_names()
    else:
        return []


def get_default_train_params(hub: Hub) -> dict:
    if hub is not None:
        return Hub.get_default_train_params(
            backend=hub.backend, task=hub.task, model_type=hub.model_type, model_size=hub.model_size
        ).to_dict()
    else:
        return None


def get_default_advanced_train_params(hub: Hub) -> dict:
    try:
        return hub.get_default_advance_train_params()
    except Exception as e:
        return None


def get_train_status(hub: Hub) -> TrainingStatus:
    if hub is not None:
        return hub.get_training_status()
    else:
        return None


def get_evaluate_status(hub: Hub) -> EvaluatingStatus:
    if hub is not None:
        return hub.get_evaluating_status()
    else:
        return None


def get_inference_status(hub: Hub) -> InferencingStatus:
    if hub is not None:
        return hub.get_inferencing_status()
    else:
        return None


def get_export_onnx_status(hub: Hub) -> ExportingOnnxStatus:
    if hub is not None:
        return hub.get_exporting_onnx_status()
    else:
        return None


def get_export_waffle_status(hub: Hub) -> ExportingWaffleStatus:
    if hub is not None:
        return hub.get_exporting_waffle_status()
    else:
        return None


def get_status(run_type: str, hub: Hub) -> dict:
    if run_type == TaskType.TRAIN:
        return get_train_status(hub)
    elif run_type == TaskType.EVALUATE:
        return get_evaluate_status(hub)
    elif run_type == TaskType.INFERENCE:
        return get_inference_status(hub)
    elif run_type == TaskType.EXPORT_ONNX:
        return get_export_onnx_status(hub)
    elif run_type == TaskType.EXPORT_WAFFLE:
        return get_export_waffle_status(hub)
    else:
        return None


METRICS_MAP = {
    "autocare_dlt": {
        "SEMANTIC_SEGMENTATION": {
            "train_loss": ["train/loss"],
            "val_loss": ["val/loss"],
            "metric": ["val/mpa"],
        },
        "OBJECT_DETECTION": {
            "train_loss": ["train/loss"],
            "val_loss": [],
            "metric": ["val/COCOAP50", "val/COCOAP50_95"],
        },
        "CLASSIFICATION": {
            "train_loss": ["train/loss"],
            "val_loss": [],
            "metric": ["val/f1", "val/precision", "val/recall", "val/accuracy"],
        },
        "TEXT_RECOGNITION": {
            "train_loss": ["train/loss"],
            "val_loss": [],
            "metric": ["val/norm_ED", "val/accuracy"],
        },
    },
    "ultralytics": {
        "INSTANCE_SEGMENTATION": {
            "train_loss": ["train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss"],
            "val_loss": ["val/box_loss", "val/seg_loss", "val/cls_loss", "val/dfl_loss"],
            "metric": [
                "metrics/precision(M)",
                "metrics/recall(M)",
                "metrics/mAP50(M)",
                "metrics/mAP50-95(M)",
            ],
        },
        "OBJECT_DETECTION": {
            "train_loss": ["train/box_loss", "train/cls_loss", "train/dfl_loss"],
            "val_loss": ["val/box_loss", "val/cls_loss", "val/dfl_loss"],
            "metric": [
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            ],
        },
        "CLASSIFICATION": {
            "train_loss": ["train/loss"],
            "val_loss": ["val/loss"],
            "metric": ["metrics/accuracy_top1", "metrics/accuracy_top5"],
        },
    },
    "transformers": {
        "CLASSIFICATION": {
            "train_loss": ["loss"],
            "val_loss": ["eval_loss"],
            "metric": ["eval_accuracy"],
        },
        "OBJECT_DETECTION": {
            "train_loss": ["loss"],
            "val_loss": ["eval_loss"],
            "metric": [],
        },
    },
}


def get_metrics(hub: Hub) -> tuple[dict]:
    if hub is not None:
        backend = hub.backend.lower()
        task = hub.task.upper()
        metrics = hub.get_metrics()
        train_loss = dict()
        for tag_name in METRICS_MAP[backend][task]["train_loss"]:
            train_loss[tag_name] = [
                tags["value"] for metric in metrics for tags in metric if tags["tag"] == tag_name
            ]
        val_loss = dict()
        for tag_name in METRICS_MAP[backend][task]["val_loss"]:
            val_loss[tag_name] = [
                tags["value"] for metric in metrics for tags in metric if tags["tag"] == tag_name
            ]
        metric = dict()
        for tag_name in METRICS_MAP[backend][task]["metric"]:
            metric[tag_name] = [
                tags["value"] for metric in metrics for tags in metric if tags["tag"] == tag_name
            ]

        return train_loss, val_loss, metric
    else:
        return ({}, {}, {})


def get_evaluate_result(hub: Hub) -> dict:
    if hub is not None:
        return hub.get_evaluate_result()
    else:
        return None


def get_train_config(hub: Hub) -> dict:
    if hub is not None:
        return hub.get_train_config().to_dict()
    else:
        return None


def new(
    name: str,
    backend: str,
    task: str,
    model_type: str,
    model_size: str,
    categories: list[str] = None,
    hub_root_dir: str = None,
) -> Hub:
    return Hub.new(
        name=name,
        backend=backend,
        task=task,
        model_type=model_type,
        model_size=model_size,
        categories=categories if categories else None,
        root_dir=hub_root_dir,
    )


def from_waffle(name: str, hub_root_dir: str, waffle_file) -> Hub:
    temp_waffle_file = NamedTemporaryFile(suffix=".waffle")
    temp_waffle_file.write(waffle_file.read())

    return Hub.from_waffle_file(
        name=name,
        waffle_file=str(temp_waffle_file.name),
        root_dir=hub_root_dir,
    )


def load(hub_name: str, root_dir: str = None) -> Hub:
    if hub_name in get_hub_list(root_dir=root_dir):
        return Hub.load(hub_name, root_dir=root_dir)
    else:
        return None


def delete_hub(hub: Hub) -> None:
    if hub is not None:
        return hub.delete_hub()
    else:
        return None


def delete_artifact(hub: Hub) -> None:
    if hub is not None:
        return hub.delete_artifact()
    else:
        return None


def delete_status(hub: Hub, run_type: str) -> None:
    if hub is not None:
        if run_type == TaskType.TRAIN:
            if hub.training_status_file.exists():
                io.remove_file(hub.training_status_file)
        elif run_type == TaskType.EVALUATE:
            if hub.evaluating_status_file.exists():
                io.remove_file(hub.evaluating_status_file)
        elif run_type == TaskType.INFERENCE:
            if hub.inferencing_status_file.exists():
                io.remove_file(hub.inferencing_status_file)
        elif run_type == TaskType.EXPORT_ONNX:
            if hub.exporting_onnx_status_file.exists():
                io.remove_file(hub.exporting_onnx_status_file)
        elif run_type == TaskType.EXPORT_WAFFLE:
            if hub.exporting_waffle_status_file.exists():
                io.remove_file(hub.exporting_waffle_status_file)


def delete_evaluate_result(hub: Hub) -> None:
    delete_status(run_type=TaskType.EVALUATE, hub=hub)
    if hub.evaluate_file.exists():
        io.remove_file(hub.evaluate_file)


def delete_inference_result(hub: Hub) -> None:
    delete_status(run_type=TaskType.INFERENCE, hub=hub)
    if hub.inference_dir.exists():
        io.remove_directory(hub.inference_dir, recursive=True)


def delete_export_onnx_result(hub: Hub) -> None:
    delete_status(run_type=TaskType.EXPORT_ONNX, hub=hub)
    if hub.onnx_file.exists():
        io.remove_file(hub.onnx_file)


def delete_export_waffle_result(hub: Hub) -> None:
    delete_status(run_type=TaskType.EXPORT_WAFFLE, hub=hub)
    if hub.waffle_file.exists():
        io.remove_file(hub.waffle_file)


def delete_result(hub: Hub, run_type: str) -> None:
    if run_type == TaskType.EVALUATE:
        delete_evaluate_result(hub)
    elif run_type == TaskType.INFERENCE:
        delete_inference_result(hub)
    elif run_type == TaskType.EXPORT_ONNX:
        delete_export_onnx_result(hub)
    elif run_type == TaskType.EXPORT_WAFFLE:
        delete_export_waffle_result(hub)


def train(hub: Hub, args: dict) -> None:
    if hub.artifact_dir.exists():
        hub.delete_artifact()
    delete_status(run_type=TaskType.TRAIN, hub=hub)

    args.update({"hub_name": hub.name, "hub_root_dir": str(hub.root_dir)})

    task_name = f"{hub.name}_{TaskType.TRAIN}"
    api_service.add_task(name=task_name, task_type=TaskType.TRAIN, args=args)


def is_trained(hub: Hub) -> bool:
    if hub is not None:
        status = hub.get_training_status()
        if status is not None:
            return status["status_desc"] in ["SUCCESS", "STOPPED"]
    return False


def evaluate(hub: Hub, args: dict) -> None:
    delete_status(run_type=TaskType.EVALUATE, hub=hub)
    args.update({"hub_name": hub.name, "hub_root_dir": str(hub.root_dir)})

    task_name = f"{hub.name}_{TaskType.EVALUATE}"
    api_service.add_task(name=task_name, task_type=TaskType.EVALUATE, args=args)


def is_evaluated(hub: Hub) -> bool:
    if hub is not None:
        status = hub.get_evaluating_status()
        if status is not None:
            return status["status_desc"] == "SUCCESS"
    return False


def inference(hub: Hub, args: dict) -> None:
    if hub.inference_dir.exists():
        io.remove_directory(hub.inference_dir, recursive=True)
    delete_status(run_type=TaskType.INFERENCE, hub=hub)

    args.update({"hub_name": hub.name, "hub_root_dir": str(hub.root_dir)})

    task_name = f"{hub.name}_{TaskType.INFERENCE}"
    api_service.add_task(name=task_name, task_type=TaskType.INFERENCE, args=args)


def is_inferenced(hub: Hub) -> bool:
    if hub is not None:
        status = hub.get_inferencing_status()
        if status is not None:
            return status["status_desc"] == "SUCCESS"
    return False


def export_onnx(hub: Hub, args: dict):
    delete_status(run_type=TaskType.EXPORT_ONNX, hub=hub)

    args.update({"hub_name": hub.name, "hub_root_dir": str(hub.root_dir)})

    task_name = f"{hub.name}_{TaskType.EXPORT_ONNX}"
    api_service.add_task(name=task_name, task_type=TaskType.EXPORT_ONNX, args=args)


def is_exported_onnx(hub: Hub) -> bool:
    if hub is not None:
        status = hub.get_exporting_onnx_status()
        if status is not None:
            return status["status_desc"] == "SUCCESS"
    return False


def export_waffle(hub: Hub):
    delete_status(run_type=TaskType.EXPORT_WAFFLE, hub=hub)

    task_name = f"{hub.name}_{TaskType.EXPORT_WAFFLE}"
    args = {"hub_name": hub.name, "hub_root_dir": str(hub.root_dir)}
    api_service.add_task(name=task_name, task_type=TaskType.EXPORT_WAFFLE, args=args)


def is_exported_waffle(hub: Hub) -> bool:
    if hub is not None:
        status = hub.get_exporting_waffle_status()
        if status is not None:
            return status["status_desc"] == "SUCCESS"
    return False


def get_export_onnx_path(hub: Hub) -> str:
    return str(hub.onnx_file)


def get_export_waffle_path(hub: Hub) -> str:
    return str(hub.waffle_file)
