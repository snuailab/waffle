from pathlib import Path
from typing import Union

from waffle_hub.hub import Hub
from waffle_menu.active_learning import (
    EntropySampling,
    PL2NSampling,
    RandomSampling,
)
from waffle_utils.file import io

METHOD_MAP = {
    "Random": [
        "CLASSIFICATION",
        "OBJECT_DETECTION",
        "INSTANCE_SEGMENTATION",
        "SEMANTIC_SEGMENTATION",
        "KEYPOINT_DETECTION",
        "TEXT_RECOGNITION",
        "REGRESSION",
    ],  # all tasks
    "Entropy": ["CLASSIFICATION"],
    "PL2N": ["CLASSIFICATION", "OBJECT_DETECTION"],
}


def get_available_tasks(method: str):
    if method in METHOD_MAP.keys():
        return METHOD_MAP[method]
    else:
        raise ValueError(f"Invalid method: {method}")


def random_sampling(
    image_dir: Union[str, Path], num_samples: int, result_dir: Union[str, Path], seed: int
):
    RandomSampling(seed=seed).sample(
        image_dir=str(image_dir),
        num_images=num_samples,
        result_dir=str(result_dir),
        save_images=True,
    )


def entropy_sampling(
    image_dir: Union[str, Path],
    num_samples: int,
    result_dir: Union[str, Path],
    hub: Hub,
    image_size: list[int],
    batch_size: int,
    device: str,
    num_workers: int,
):
    result = EntropySampling(
        hub=hub,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    ).sample(
        image_dir=str(image_dir),
        num_images=num_samples,
        result_dir=str(result_dir),
        save_images=True,
    )


def pl2n_sampling(
    image_dir: Union[str, Path],
    num_samples: int,
    result_dir: Union[str, Path],
    hub: Hub,
    image_size: list[int],
    batch_size: int,
    device: str,
    num_workers: int,
    diversity_sampling: bool,
):
    result = PL2NSampling(
        hub=hub,
        diversity_sampling=diversity_sampling,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    ).sample(
        image_dir=str(image_dir),
        num_images=num_samples,
        result_dir=str(result_dir),
        save_images=True,
    )


def get_result_json(result_dir: Union[str, Path]) -> dict:
    return io.load_json(Path(result_dir) / "result.json")


def get_sample_json(result_dir: Union[str, Path]) -> dict:
    return io.load_json(Path(result_dir) / "sampled.json")


def get_total_json(result_dir: Union[str, Path]) -> dict:
    return io.load_json(Path(result_dir) / "total.json")
