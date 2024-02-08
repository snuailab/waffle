import os

import torch
from waffle_hub.hub import Hub
from waffle_hub.schema.result import EvaluateResult, InferenceResult, TrainResult


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


def get_model_config_dict(hub_name: str, root_dir: str = None) -> dict:
    if hub_name in Hub.get_hub_list(root_dir=root_dir):
        hub = Hub.load(hub_name, root_dir=root_dir)
        return hub.get_model_config().to_dict()
    else:
        return None


def get_category_names(hub_name: str, root_dir: str = None) -> list[str]:
    if hub_name in Hub.get_hub_list(root_dir=root_dir):
        hub = Hub.load(hub_name, root_dir=root_dir)
        return hub.get_category_names()
    else:
        return []


def get_train_status(hub_name: str, root_dir: str = None) -> dict:
    if hub_name in Hub.get_hub_list(root_dir=root_dir):
        hub = Hub.load(hub_name, root_dir=root_dir)
        return hub.get_training_status()
    else:
        return None


def get_default_train_params(hub_name: str, root_dir: str = None) -> dict:
    if hub_name in Hub.get_hub_list(root_dir=root_dir):
        hub = Hub.load(hub_name, root_dir=root_dir)
        return Hub.get_default_train_params(
            backend=hub.backend, task=hub.task, model_type=hub.model_type, model_size=hub.model_size
        ).to_dict()
    else:
        return None


def get_default_advanced_train_params(hub_name: str, root_dir: str = None) -> dict:
    try:
        hub = Hub.load(hub_name, root_dir=root_dir)
        return hub.get_default_advance_train_params()
    except Exception as e:
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


def get_metrics(hub_name: str, root_dir: str = None) -> list[list[dict]]:
    if hub_name in Hub.get_hub_list(root_dir=root_dir):
        hub = Hub.load(hub_name, root_dir=root_dir)
        return hub.get_metrics()
    else:
        return []


def get_evaluate_result(hub_name: str, root_dir: str = None) -> dict:
    if hub_name in Hub.get_hub_list(root_dir=root_dir):
        hub = Hub.load(hub_name, root_dir=root_dir)
        return hub.get_evaluate_result()
    else:
        return None


def get_train_config(hub_name: str, root_dir: str = None) -> dict:
    if hub_name in Hub.get_hub_list(root_dir=root_dir):
        hub = Hub.load(hub_name, root_dir=root_dir)
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


def load(hub_name: str, root_dir: str = None) -> Hub:
    if hub_name in get_hub_list(root_dir=root_dir):
        return Hub.load(hub_name, root_dir=root_dir)
    else:
        return None


def delete_hub(hub_name: str, root_dir: str = None) -> None:
    if hub_name in get_hub_list(root_dir=root_dir):
        hub = Hub.load(hub_name, root_dir=root_dir)
        return hub.delete_hub()
    else:
        return None


def delete_artifact(hub_name: str, root_dir: str = None) -> None:
    if hub_name in get_hub_list(root_dir=root_dir):
        hub = Hub.load(hub_name, root_dir=root_dir)
        return hub.delete_artifact()
    else:
        return None


def train(hub_name: str, args: dict, root_dir: str = None) -> TrainResult:
    hub = Hub.load(hub_name, root_dir=root_dir)
    if hub.artifact_dir.exists():
        hub.delete_artifact()
    return hub.train(**args)


def is_trained(hub_name: str, root_dir: str = None) -> bool:
    if hub_name in get_hub_list(root_dir=root_dir):
        hub = Hub.load(hub_name, root_dir=root_dir)
        status = hub.get_training_status()
        if status is not None:
            return status["status_desc"] == "SUCCESS"
    return False


def evaluate(hub_name: str, args: dict, root_dir: str = None) -> EvaluateResult:
    hub = Hub.load(hub_name, root_dir=root_dir)
    hub.evaluate(
        dataset=args["dataset"],
        root_dir=args["root_dir"],
        set_name=args["set_name"],
        batch_size=args["batch_size"],
        image_size=args["image_size"],
        letter_box=args["letter_box"],
        confidence_threshold=args["confidence_threshold"],
        iou_threshold=args["iou_threshold"],
        half=args["half"],
        workers=args["workers"],
        device=args["device"],
        draw=args["draw"],
    )
    return hub.evaluate(**args)


def is_evaluated(hub_name: str, root_dir: str = None) -> bool:
    if hub_name in get_hub_list(root_dir=root_dir):
        hub = Hub.load(hub_name, root_dir=root_dir)
        status = hub.get_evaluating_status()
        if status is not None:
            return status["status_desc"] == "SUCCESS"
    return False


def inference(hub_name: str, args: dict, root_dir: str = None) -> InferenceResult:
    hub = Hub.load(hub_name, root_dir=root_dir)
    hub.inference(
        source=args["source"],
        recursive=args["recursive"],
        image_size=args["image_size"],
        letter_box=args["letter_box"],
        batch_size=args["batch_size"],
        confidence_threshold=args["confidence_threshold"],
        iou_threshold=args["iou_threshold"],
        half=args["half"],
        workers=args["workers"],
        device=args["device"],
        draw=args["draw"],
        show=args["show"],
    )
    return hub.inference(**args)


def export_onnx(hub_name: str, args: dict, root_dir: str = None):
    hub = Hub.load(hub_name, root_dir=root_dir)
