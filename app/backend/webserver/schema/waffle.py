from typing import Optional, Union

from pydantic import BaseModel, Field


class WaffleTrainParams(BaseModel):
    hub_name: str
    hub_root_dir: Optional[str]
    dataset: str
    dataset_root_dir: Optional[str]
    epochs: Optional[int]
    learning_rate: Optional[float]
    batch_size: Optional[int]
    image_size: Optional[Union[int, list[int]]]
    letter_box: Optional[bool]
    device: Optional[str]
    workers: Optional[int]
    seed: Optional[int]
    advance_params: Optional[dict]


class WaffleEvaluateParams(BaseModel):
    hub_name: str
    hub_root_dir: Optional[str]
    dataset: str
    dataset_root_dir: Optional[str]
    batch_size: Optional[int]
    image_size: Optional[Union[int, list[int]]]
    letter_box: Optional[bool]
    half: Optional[bool]
    device: Optional[str]
    workers: Optional[int]


class WaffleInferenceParams(BaseModel):
    hub_name: str
    hub_root_dir: Optional[str]
    source: str
    batch_size: Optional[int]
    image_size: Optional[Union[int, list[int]]]
    letter_box: Optional[bool]
    half: Optional[bool]
    device: Optional[str]
    workers: Optional[int]
    recursive: Optional[bool]
    draw: Optional[bool]


class WaffleExportOnnxParams(BaseModel):
    hub_name: str
    hub_root_dir: Optional[str]
    batch_size: Optional[int]
    image_size: Optional[Union[int, list[int]]]
    half: Optional[bool]
    device: Optional[str]
    opset_version: Optional[int]


class WaffleExportWaffleParams(BaseModel):
    hub_name: str
    hub_root_dir: Optional[str]
