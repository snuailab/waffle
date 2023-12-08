import logging
import os
from collections import OrderedDict, defaultdict
from tempfile import NamedTemporaryFile, TemporaryDirectory

import streamlit as st
from src.util.plot import plot_bar
from streamlit_image_viewer import image_viewer
from waffle_utils.file import io

from waffle_hub.dataset import Dataset
from waffle_hub.schema.fields import Image

from .base_page import BasePage

logger = logging.getLogger(__name__)


SET_CODES = {
    "total": None,
    "train": 0,
    "val": 1,
    "test": 2,
    "unlabeled": 3,
}


class DatasetPage(BasePage):
    # utils
    def get_images(self, dataset: Dataset, set_name: str = "total") -> list[Image]:
        if set_name == "total":
            return dataset.get_images()

        try:
            image_ids = dataset.get_split_ids()[SET_CODES[set_name]]
            return dataset.get_images(image_ids) if image_ids else []
        except Exception as e:
            logger.error(e)
            return []

    def get_statistics(self, dataset: Dataset, set_name: str = "total") -> dict:
        images = self.get_images(dataset, set_name)
        image_ids = [image.image_id for image in images]
        num_images = len(images)

        categories = dataset.get_categories()
        num_categories = len(categories)

        image_to_annotations = dataset.image_to_annotations
        image_to_annotations = {image_id: image_to_annotations[image_id] for image_id in image_ids}

        num_annotations = 0
        num_images_per_category = OrderedDict(
            {category.category_id: set() for category in categories}
        )
        num_instances_per_category = OrderedDict(
            {category.category_id: 0 for category in categories}
        )
        for image_id, annotations in image_to_annotations.items():
            num_annotations += len(annotations)
            for annotation in annotations:
                num_images_per_category[annotation.category_id].add(image_id)
                num_instances_per_category[annotation.category_id] += 1
        num_images_per_category = OrderedDict(
            {
                category_id: len(image_ids)
                for category_id, image_ids in num_images_per_category.items()
            }
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

    # render
    def render_import_dataset(self):
        st.subheader("Import Dataset")

        data_type = st.radio(
            label="Data Type",
            options=["coco", "yolo"],
            index=0,
            key="import_dataset_data_type",
        )

        st.text_input("Dataset Name", key="import_dataset_name")
        if data_type == "coco":
            st.selectbox(
                "Task Type",
                ["classification", "object_detection", "instance_segmentation"],
                key="import_dataset_task_type",
            )
            st_image_zip_file = st.file_uploader(
                "image_zip_file",
                type=["zip"],
                key="import_dataset_images",
                accept_multiple_files=False,
            )
            st_json_files = st.file_uploader(
                "annotations",
                type=["json"],
                key="import_dataset_annotations",
                accept_multiple_files=True,
            )

            if st.button("Import"):
                image_zip_file = NamedTemporaryFile(suffix=".zip")
                image_zip_file.write(st_image_zip_file.read())

                json_files = [NamedTemporaryFile(suffix=".json") for _ in range(len(st_json_files))]
                for i, st_json_file in enumerate(st_json_files):
                    json_files[i].write(st_json_file.read())

                with TemporaryDirectory() as temp_dir:
                    io.unzip(image_zip_file.name, temp_dir, create_directory=True)

                    Dataset.from_coco(
                        name=st.session_state.import_dataset_name,
                        task=st.session_state.import_dataset_task_type,
                        coco_root_dir=temp_dir,
                        coco_file=[json_file.name for json_file in json_files],
                        root_dir=st.session_state.waffle_dataset_root_dir,
                    )

                del image_zip_file
                del json_files

                st.success("Import done!")
                st.session_state.select_dataset_name = st.session_state.import_dataset_name

        elif data_type == "yolo":
            st.text("NOT IMPLEMENTED YET")

    def render_select_dataset(self):
        st.subheader("Select Dataset")
        dataset_list = Dataset.get_dataset_list(root_dir=st.session_state.waffle_dataset_root_dir)

        filter_maps = defaultdict(set)
        dataset_infos = []
        dataset_captions = []
        for name in dataset_list:
            dataset = Dataset.load(name, root_dir=st.session_state.waffle_dataset_root_dir)

            dataset_info = dataset.get_dataset_info().to_dict()
            dataset_infos.append(dataset_info)
            for key, value in dataset_info.items():
                if isinstance(value, (str, int, float)) and key != "name":
                    filter_maps[key].add(value)

            dataset_captions.append(
                f"Task: {dataset.task.upper():>24}, Categories: {str(dataset.get_category_names()):>20}, Created: {dataset.created:>20}"
            )

        for key, value in filter_maps.items():
            filter_maps[key] = list(set(value))

        filter_key = st.selectbox("filter key", ["All"] + list(filter_maps.keys()), key="filter_key")
        if filter_key and filter_key != "All":
            values = list(filter_maps[st.session_state.filter_key])
            st.multiselect("filter value", values, default=values, key="filter_value")

            filtered_dataset_index = []
            for i, dataset_info in enumerate(dataset_infos):
                if dataset_info[st.session_state.filter_key] in st.session_state.filter_value:
                    filtered_dataset_index.append(i)

            dataset_list = [dataset_list[i] for i in filtered_dataset_index]
            dataset_captions = [dataset_captions[i] for i in filtered_dataset_index]

        st.radio(
            "Select Dataset", dataset_list, 0, captions=dataset_captions, key="select_dataset_name"
        )

    def render_actions(self):
        dataset = Dataset.load(
            st.session_state.select_dataset_name, root_dir=st.session_state.waffle_dataset_root_dir
        )
        st.subheader("Dataset Actions")

        st.subheader("Split")
        train_ratio = st.slider("split ratio", 0.0, 1.0, 0.8, 0.05, key="split_ratio")
        if st.button("Split"):
            dataset.split(train_ratio)
            st.success("Split done!")

        st.divider()

        st.subheader("Download Dataset")
        data_type = st.selectbox("format", ["coco", "yolo"])
        if st.button("Export"):
            exported_directory = dataset.export(data_type)
            st.success("Export done!")
            st.text_area("Exported Path", exported_directory)
            with NamedTemporaryFile(suffix=".zip") as zip_file:
                io.zip(exported_directory, zip_file.name)
                with open(zip_file.name, "rb") as f:
                    data = f.read()
                    st.download_button(
                        "Download", data, f"{data_type}.zip", key="download_exported_dataset"
                    )

        st.divider()

        agree = st.checkbox("I agree to delete this dataset. This action cannot be undone.")
        if st.button("Delete", disabled=not agree):
            dataset.delete()
            st.success("Delete done!")
            st.experimental_rerun()

    def render_dataset_info(self):
        dataset = Dataset.load(
            st.session_state.select_dataset_name, root_dir=st.session_state.waffle_dataset_root_dir
        )

        st.subheader("Dataset Info")
        st.write(dataset.get_dataset_info().to_dict())

    def render_dataset_statistics(self):
        dataset = Dataset.load(
            st.session_state.select_dataset_name, root_dir=st.session_state.waffle_dataset_root_dir
        )

        st.subheader("Statistics")
        set_name = st.radio(
            "", ["total", "train", "val", "test", "unlabeled"], 0, key="dataset_info_select_split"
        )
        statistics = self.get_statistics(dataset, set_name)
        st.write(
            {
                "num_images": statistics["num_images"],
                "num_categories": statistics["num_categories"],
                "num_instances": statistics["num_annotations"],
            }
        )

        columns = st.columns(2)

        with columns[0]:
            st.pyplot(
                plot_bar(
                    list(statistics["num_instances_per_category"].keys()),
                    list(statistics["num_instances_per_category"].values()),
                    title="num_instances_per_category",
                    names=[category.name for category in statistics["categories"]],
                    xlabel="num_instances",
                    ylabel="category_id",
                    figsize=(10, 5),
                    legend=True,
                )
            )

        with columns[1]:
            st.pyplot(
                plot_bar(
                    list(statistics["num_images_per_category"].keys()),
                    list(statistics["num_images_per_category"].values()),
                    title="num_images_per_category",
                    names=[category.name for category in statistics["categories"]],
                    xlabel="num_images",
                    ylabel="category_id",
                    figsize=(10, 5),
                    legend=True,
                )
            )

        st.divider()

        st.subheader("Sample Images")
        if st.checkbox("Show Annotations"):
            if not dataset.draw_dir.exists():
                dataset.draw_annotations()
                st.experimental_rerun()
            image_dir = dataset.draw_dir
        else:
            image_dir = dataset.raw_image_dir
        image_paths = [image_dir / image["file_name"] for image in statistics["images"]]
        image_viewer(image_paths)

    def render_content(self):
        with st.expander("Import New Dataset"):
            self.render_import_dataset()
        st.divider()
        self.render_select_dataset()
        if st.session_state.select_dataset_name:
            st.divider()

            col1, col2 = st.columns([0.6, 0.4], gap="medium")
            with col1:
                self.render_dataset_info()
            with col2:
                self.render_actions()

            with st.expander("Dataset Statistics", expanded=True):
                self.render_dataset_statistics()
