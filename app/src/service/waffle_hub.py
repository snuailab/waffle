import os
import time
from tempfile import NamedTemporaryFile

import torch
from src.schema.run import RunType
from waffle_hub.hub import Hub
from waffle_hub.schema.result import EvaluateResult, InferenceResult, TrainResult
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
    if run_type == RunType.TRAIN:
        return get_train_status(hub)
    elif run_type == RunType.EVALUATE:
        return get_evaluate_status(hub)
    elif run_type == RunType.INFERENCE:
        return get_inference_status(hub)
    elif run_type == RunType.EXPORT_ONNX:
        return get_export_onnx_status(hub)
    elif run_type == RunType.EXPORT_WAFFLE:
        return get_export_waffle_status(hub)
    else:
        return None


METRICS_MAP = {
    "autocare_dlt": {
        "SEMANTIC_SEGMENTATION": {
            "train_loss": ["Loss", "mIoU", "Pixel Accuracy"],
            "metric": ["mIoU", "Pixel Accuracy"],
        },
        "OBJECT_DETECTION": {
            "train_loss": ["Loss", "mAP", "Precision", "Recall"],
            "metric": ["mAP", "Precision", "Recall"],
        },
        "CLASSIFICATION": {
            "train_loss": ["Loss", "Accuracy"],
            "metric": ["Accuracy"],
        },
    },
    "ultralytics": {
        "INSTANCE_SEGMENTATION": {
            "train_loss": ["Loss", "mIoU", "Pixel Accuracy"],
            "metric": ["mIoU", "Pixel Accuracy"],
        },
        "OBJECT_DETECTION": {
            "train_loss": ["Loss", "mAP", "Precision", "Recall"],
            "metric": ["mAP", "Precision", "Recall"],
        },
        "CLASSIFICATION": {
            "train_loss": ["Loss", "Accuracy"],
            "metric": ["Accuracy"],
        },
    },
    "transformers": {
        "CLASSIFICATION": {
            "train_loss": ["Loss", "Accuracy"],
            "metric": ["Accuracy"],
        },
        "OBJECT_DETECTION": {
            "train_loss": ["Loss", "mAP", "Precision", "Recall"],
            "metric": ["mAP", "Precision", "Recall"],
        },
    },
}


def get_metrics(hub: Hub) -> list[list[dict]]:
    if hub is not None:
        return hub.get_metrics()
    else:
        return []


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
        hub_root_dir=hub_root_dir,
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
        if run_type == RunType.TRAIN:
            if hub.training_status_file.exists():
                io.remove_file(hub.training_status_file)
        elif run_type == RunType.EVALUATE:
            if hub.evaluating_status_file.exists():
                io.remove_file(hub.evaluating_status_file)
        elif run_type == RunType.INFERENCE:
            if hub.inferencing_status_file.exists():
                io.remove_file(hub.inferencing_status_file)
        elif run_type == RunType.EXPORT_ONNX:
            if hub.exporting_onnx_status_file.exists():
                io.remove_file(hub.exporting_onnx_status_file)
        elif run_type == RunType.EXPORT_WAFFLE:
            if hub.exporting_waffle_status_file.exists():
                io.remove_file(hub.exporting_waffle_status_file)


def delete_evaluate_result(hub: Hub) -> None:
    delete_status(run_type=RunType.EVALUATE, hub=hub)
    if hub.evaluate_file.exists():
        io.remove_file(hub.evaluate_file)


def delete_inference_result(hub: Hub) -> None:
    delete_status(run_type=RunType.INFERENCE, hub=hub)
    if hub.inference_dir.exists():
        io.remove_directory(hub.inference_dir, recursive=True)


def delete_export_onnx_result(hub: Hub) -> None:
    delete_status(run_type=RunType.EXPORT_ONNX, hub=hub)
    if hub.onnx_file.exists():
        io.remove_file(hub.onnx_file)


def delete_export_waffle_result(hub: Hub) -> None:
    delete_status(run_type=RunType.EXPORT_WAFFLE, hub=hub)
    if hub.waffle_file.exists():
        io.remove_file(hub.waffle_file)


def delete_result(hub: Hub, run_type: str) -> None:
    if run_type == RunType.EVALUATE:
        delete_evaluate_result(hub)
    elif run_type == RunType.INFERENCE:
        delete_inference_result(hub)
    elif run_type == RunType.EXPORT_ONNX:
        delete_export_onnx_result(hub)
    elif run_type == RunType.EXPORT_WAFFLE:
        delete_export_waffle_result(hub)


def train(hub: Hub, args: dict) -> TrainResult:
    if hub.artifact_dir.exists():
        hub.delete_artifact()
    return hub.train(**args)


def is_trained(hub: Hub) -> bool:
    if hub is not None:
        status = hub.get_training_status()
        if status is not None:
            return status["status_desc"] == "SUCCESS"
    return False


def evaluate(hub: Hub, args: dict) -> EvaluateResult:
    return hub.evaluate(**args)


def is_evaluated(hub: Hub) -> bool:
    if hub is not None:
        status = hub.get_evaluating_status()
        if status is not None:
            return status["status_desc"] == "SUCCESS"
    return False


def inference(hub: Hub, args: dict) -> InferenceResult:
    if hub.inference_dir.exists():
        io.remove_directory(hub.inference_dir, recursive=True)
    return hub.inference(**args)


def is_inferenced(hub: Hub) -> bool:
    if hub is not None:
        status = hub.get_inferencing_status()
        if status is not None:
            return status["status_desc"] == "SUCCESS"
    return False


def export_onnx(hub: Hub, args: dict):
    return hub.export_onnx(**args)


def is_exported_onnx(hub: Hub) -> bool:
    if hub is not None:
        status = hub.get_exporting_onnx_status()
        if status is not None:
            return status["status_desc"] == "SUCCESS"
    return False


def export_waffle(hub: Hub):
    return hub.export_waffle()


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
