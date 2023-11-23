from collections import OrderedDict
import PIL.Image as Image

from .base_page import BasePage

import streamlit as st
from streamlit_image_viewer import image_viewer

from waffle_hub.dataset import Dataset


class DatasetPage(BasePage):
    def import_dataset(self):
        st.subheader("Import Dataset")

    def select_dataset(self):
        st.selectbox(
            label="Select Dataset",
            options=[None] + Dataset.get_dataset_list(),
            index=0,
            key="select_dataset_name"
        )
    
    def render_actions(self):
        dataset = Dataset.load(st.session_state.select_dataset_name)

        st.subheader("Split")
        train_ratio = st.slider("split ratio", 0.0, 1.0, 0.8, 0.1, key="split_ratio")
        if st.button("Split"):
            dataset.split(train_ratio)
            st.success("Split done!")
        
        st.subheader("export")
        data_format = st.selectbox("format", ["coco", "yolo"])
        if st.button("Export"):
            dataset.export(data_format)
            st.success("Export done!")
    
    def render_dataset_info(self):
        dataset = Dataset.load(st.session_state.select_dataset_name)

        st.subheader("Dataset Info")
        st.write(dataset.get_dataset_info().to_dict())

        st.subheader("Annotation Info")
        st.bar_chart(dataset.get_num_images_per_category())

        try:
            train_ids, val_ids, test_ids, unlabeled_ids = dataset.get_split_ids()
            st.subheader("Split Info")
            st.bar_chart(OrderedDict({
                "Train": len(train_ids),
                "Validation": len(val_ids),
                "Test": len(test_ids),
                "Unlabeled": len(unlabeled_ids),
            }))
        except:
            pass
    
    def render_sample_images(self):
        dataset = Dataset.load(st.session_state.select_dataset_name)

        st.subheader("Sample Images")

        split_name = st.radio("select split", ["train", "val", "test", "unlabeled"], 0, key="select_split")
        try:
            train_ids, val_ids, test_ids, unlabeled_ids = dataset.get_split_ids()
            if split_name == "train":
                image_ids = train_ids if train_ids else []
            elif split_name == "val":
                image_ids = val_ids if val_ids else []
            elif split_name == "test":
                image_ids = test_ids if test_ids else []
            elif split_name == "unlabeled":
                image_ids = unlabeled_ids if unlabeled_ids else []
        except:
            image_ids = None

        images = dataset.get_images(image_ids=image_ids)
        image_dir = dataset.raw_image_dir
        image_paths = [image_dir / image["file_name"] for image in images]

        image_viewer(image_paths)

    def render_content(self):
        self.import_dataset()
        self.select_dataset()
        if st.session_state.select_dataset_name:
            self.render_actions()
            self.render_dataset_info()
            self.render_sample_images()