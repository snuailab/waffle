import os
import random
from collections import OrderedDict
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Union

from waffle_hub.dataset import Dataset
from waffle_hub.schema.fields import Image
from waffle_utils.file import io, search

SET_CODES = {
    "total": None,
    "train": 0,
    "val": 1,
    "test": 2,
    "unlabeled": 3,
}


def get_parse_root_dir() -> Path:
    return Dataset.parse_root_dir(os.getenv("WAFFLE_DATASET_ROOT_DIR", None))


def get_dataset_list(root_dir: str = None, task: str = None) -> list[str]:
    dataset_list = Dataset.get_dataset_list(root_dir=root_dir)
    temp_list = dataset_list.copy()
    if task:
        for dataset_name in dataset_list:
            dataset = Dataset.load(dataset_name, root_dir=root_dir)
            if dataset.task.lower() != task.lower():
                temp_list.remove(dataset_name)

    return temp_list


def get_dataset_info_dict(dataset_name: str, root_dir: str = None) -> dict:
    dataset = Dataset.load(dataset_name, root_dir=root_dir)
    return dataset.get_dataset_info().to_dict()


def get_category_names(dataset_name: str, root_dir: str = None) -> list[str]:
    dataset = Dataset.load(dataset_name, root_dir=root_dir)
    return dataset.get_category_names()


def get_images(dataset_name: str, set_name: str = "total", root_dir: str = None) -> list[Image]:
    dataset = Dataset.load(dataset_name, root_dir=root_dir)
    if set_name == "total":
        return dataset.get_images()
    else:
        image_ids = dataset.get_split_ids()[SET_CODES[set_name]]
        return dataset.get_images(image_ids) if image_ids else []


def get_statistics(dataset_name: str, set_name: str = "total", root_dir: str = None) -> dict:
    dataset = Dataset.load(dataset_name, root_dir=root_dir)
    if set_name == "total":
        images = dataset.get_images()
    else:
        image_ids = dataset.get_split_ids()[SET_CODES[set_name]]
        images = dataset.get_images(image_ids) if image_ids else []

    image_ids = [image.image_id for image in images]
    num_images = len(images)

    categories = dataset.get_categories()
    num_categories = len(categories)

    image_to_annotations = dataset.image_to_annotations
    image_to_annotations = {image_id: image_to_annotations[image_id] for image_id in image_ids}

    num_annotations = 0
    num_images_per_category = OrderedDict({category.category_id: set() for category in categories})
    num_instances_per_category = OrderedDict({category.category_id: 0 for category in categories})
    for image_id, annotations in image_to_annotations.items():
        num_annotations += len(annotations)
        for annotation in annotations:
            num_images_per_category[annotation.category_id].add(image_id)
            num_instances_per_category[annotation.category_id] += 1
    num_images_per_category = OrderedDict(
        {category_id: len(image_ids) for category_id, image_ids in num_images_per_category.items()}
    )

    return {
        "images": images,
        "categories": categories,
        "num_images": num_images,
        "num_categories": num_categories,
        "image_to_annotations": image_to_annotations,
        "num_annotations": num_annotations,
        "num_images_per_category": num_images_per_category,
        "num_instances_per_category": num_instances_per_category,
    }


def get_split_list(dataset_name: str, root_dir: str = None) -> list[str]:
    dataset = Dataset.load(dataset_name, root_dir=root_dir)
    set_names = ["train", "val", "test", "unlabeled"]
    split_ids = dataset.get_split_ids()
    return [set_name for set_name in set_names if len(split_ids[SET_CODES[set_name]]) > 0]


def get_sample_image_paths(
    dataset_name: str,
    sample_num: int = 100,
    draw: bool = False,
    set_name: str = "total",
    root_dir: str = None,
) -> list[Path]:
    dataset = Dataset.load(dataset_name, root_dir=root_dir)
    if set_name == "total":
        images = dataset.get_images()
        image_ids = [image.image_id for image in images]
    else:
        image_ids = dataset.get_split_ids()[SET_CODES[set_name]]

    sample_ids = random.sample(
        image_ids,
        sample_num if len(image_ids) > sample_num else len(image_ids),
    )

    if draw:
        if dataset.draw_dir.exists():
            io.remove_directory(dataset.draw_dir, recursive=True)
        dataset.draw_annotations(sample_ids)
        return [path for path in dataset.draw_dir.rglob("*") if path.is_file()]
    else:
        image_dir = dataset.raw_image_dir
        return [image_dir / image["file_name"] for image in dataset.get_images(sample_ids)]


# waffle dataset methods
def from_coco(dataset_name: str, root_dir: str, task: str, image_zip_file, json_files):
    temp_image_zip_file = NamedTemporaryFile(suffix=".zip")
    temp_image_zip_file.write(image_zip_file.read())

    temp_json_files = [NamedTemporaryFile(suffix=".json") for _ in range(len(json_files))]
    for i, json_file in enumerate(json_files):
        temp_json_files[i].write(json_file.read())

    with TemporaryDirectory() as temp_dir:
        io.unzip(temp_image_zip_file.name, temp_dir, create_directory=True)

        Dataset.from_coco(
            name=dataset_name,
            task=task,
            coco_root_dir=temp_dir,
            coco_file=[temp_json_file.name for temp_json_file in temp_json_files],
            root_dir=root_dir,
        )


def from_yolo(dataset_name: str, root_dir: str, task: str, yolo_root_zip_file):
    temp_root_zip_file = NamedTemporaryFile(suffix=".zip")
    temp_root_zip_file.write(yolo_root_zip_file.read())

    with TemporaryDirectory() as temp_dir:
        io.unzip(temp_root_zip_file.name, temp_dir, create_directory=True)
        yaml_file = search.get_files(temp_dir, extension=".yaml")
        if yaml_file:
            if len(yaml_file) > 1:
                raise ValueError("There are multiple yaml files in the root directory.")
            temp_yaml_file = yaml_file[0]
        else:
            if task != "classification":
                raise ValueError(f"{task} requires a yaml file.")
            temp_yaml_file = None

        Dataset.from_yolo(
            name=dataset_name,
            task=task,
            yolo_root_dir=temp_dir,
            yaml_path=temp_yaml_file,
            root_dir=root_dir,
        )


def load(dataset_name: str, root_dir: str = None) -> Dataset:
    return Dataset.load(dataset_name, root_dir=root_dir)


def split(
    dataset_name: str, train_ratio: float, val_ratio: float, test_ratio: float, root_dir: str = None
):
    dataset = Dataset.load(dataset_name, root_dir=root_dir)
    dataset.split(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)


def export(dataset_name: str, data_type: str, root_dir: str = None) -> str:
    dataset = Dataset.load(dataset_name, root_dir=root_dir)
    return dataset.export(data_type=data_type)


def delete(dataset_name: str, root_dir: str = None):
    dataset = Dataset.load(dataset_name, root_dir=root_dir)
    dataset.delete()
