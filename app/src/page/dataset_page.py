import decimal
import logging
import os
from collections import OrderedDict, defaultdict
from tempfile import NamedTemporaryFile, TemporaryDirectory

import streamlit as st

# from waffle_hub.dataset import Dataset
from src.service import waffle_dataset as wd
from src.utils.plot import plot_bar
from streamlit_image_viewer import image_viewer
from waffle_hub.schema.fields import Image
from waffle_utils.file import io

from .base_page import BasePage

logger = logging.getLogger(__name__)


class DatasetPage(BasePage):
    # render
    def render_import_dataset(self):
        st.subheader("Import Dataset")

        data_type = st.radio(
            label="Data Type",
            options=["coco", "yolo"],
            index=0,
            key="import_dataset_data_type",
            horizontal=True,
        )

        st.text_input("Dataset Name", key="import_dataset_name")
        if data_type == "coco":
            st.selectbox(
                "Task Type",
                [
                    "classification",
                    "object_detection",
                    "instance_segmentation",
                    "semantic_segmentation",
                ],
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
                wd.from_coco(
                    dataset_name=st.session_state.import_dataset_name,
                    task=st.session_state.import_dataset_task_type,
                    image_zip_file=st_image_zip_file,
                    json_files=st_json_files,
                    root_dir=st.session_state.waffle_dataset_root_dir,
                )

                st.success("Import done!")
                st.session_state.select_dataset_name = st.session_state.import_dataset_name

        elif data_type == "yolo":
            st.text("NOT IMPLEMENTED YET")

    def render_select_dataset(self):
        st.subheader("Select Dataset")
        dataset_list = wd.get_dataset_list(root_dir=st.session_state.waffle_dataset_root_dir)

        filter_maps = defaultdict(set)
        dataset_infos = []
        dataset_captions = []
        for name in dataset_list:
            dataset_info = wd.get_dataset_info_dict(
                dataset_name=name, root_dir=st.session_state.waffle_dataset_root_dir
            )
            dataset_infos.append(dataset_info)
            for key, value in dataset_info.items():
                if isinstance(value, (str, int, float)) and key != "name":
                    filter_maps[key].add(value)

            dataset_captions.append(
                f"Task: {dataset_info['task'].upper():>24}, Categories: {str([category['name'] for category in dataset_info['categories']]):>20}, Created: {dataset_info['created']:>20}"
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
        st.subheader("Dataset Actions")

        st.subheader("Split")
        train_ratio = st.slider("train ratio", 0.0, 1.0, 0.8, 0.1, key="split_train_ratio")
        val_ratio = st.slider("val ratio", 0.0, 1.0, 0.1, 0.1, key="split_val_ratio")
        test_ratio = st.slider("test ratio", 0.0, 1.0, 0.1, 0.1, key="split_test_ratio")

        # make the sum of ratios to 1
        ratio_sum = train_ratio + val_ratio + test_ratio
        train_ratio = float(decimal.Decimal(train_ratio / ratio_sum))
        val_ratio = float(decimal.Decimal(val_ratio / ratio_sum))
        test_ratio = float(decimal.Decimal(test_ratio / ratio_sum))

        st.write(
            f"train ratio: {train_ratio:>.2}, val ratio: {val_ratio:>.2}, test ratio: {test_ratio:>.2}, total: {train_ratio + val_ratio + test_ratio}"
        )
        split_button_disabled = False
        if train_ratio + val_ratio + test_ratio != 1.0:
            st.error("The sum of ratios must be 1.0")
            split_button_disabled = True

        if st.button("Split", disabled=split_button_disabled):
            wd.split(
                dataset_name=st.session_state.select_dataset_name,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                root_dir=st.session_state.waffle_dataset_root_dir,
            )
            st.success("Split done!")

        st.divider()

        st.subheader("Download Dataset")
        data_type = st.selectbox("format", ["coco", "yolo"])
        if st.button("Export"):
            exported_directory = wd.export(
                dataset_name=st.session_state.select_dataset_name,
                data_type=data_type,
                root_dir=st.session_state.waffle_dataset_root_dir,
            )
            st.success("Export done!")
            st.info(f"Exported Path: {exported_directory}")
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
            wd.delete(
                dataset_name=st.session_state.select_dataset_name,
                root_dir=st.session_state.waffle_dataset_root_dir,
            )
            st.success("Delete done!")
            st.experimental_rerun()

    def render_dataset_info(self):
        st.subheader("Dataset Info")
        dataset_info = wd.get_dataset_info_dict(
            dataset_name=st.session_state.select_dataset_name,
            root_dir=st.session_state.waffle_dataset_root_dir,
        )
        st.write(dataset_info)

    def render_dataset_statistics(self):

        st.subheader("Statistics")
        col1, col2 = st.columns([0.3, 0.7], gap="medium")
        with col1:
            set_name = st.radio(
                "split_radio",
                ["total", "train", "val", "test", "unlabeled"],
                0,
                key="dataset_info_select_split",
            )
        with col2:
            statistics = wd.get_statistics(
                dataset_name=st.session_state.select_dataset_name,
                set_name=set_name,
                root_dir=st.session_state.waffle_dataset_root_dir,
            )
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

        draw = st.checkbox("Show Annotations")
        with st.spinner("Drawing..."):
            image_paths = wd.get_sample_image_paths(
                dataset_name=st.session_state.select_dataset_name,
                sample_num=105,
                draw=draw,
                set_name=set_name,
                root_dir=st.session_state.waffle_dataset_root_dir,
            )

        image_viewer(image_paths, ncol=5, nrow=3)

    def render_content(self):
        with st.expander("Import New Dataset"):
            self.render_import_dataset()
        st.divider()
        self.render_select_dataset()
        if st.session_state.select_dataset_name:
            st.divider()

            col1, col2 = st.columns([0.5, 0.5], gap="medium")
            with col1:
                self.render_dataset_info()
            with col2:
                self.render_actions()

            with st.expander("Dataset Statistics", expanded=True):
                self.render_dataset_statistics()
