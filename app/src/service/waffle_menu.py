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
