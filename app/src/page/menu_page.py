import logging
from collections import defaultdict
from pathlib import Path

import streamlit as st
import streamlit_shadcn_ui as ui
from src.component.auto_component import generate_component
from src.schema.run import RunType
from src.service import waffle_dataset as wd
from src.service import waffle_hub as wh
from src.service.run_service import run_service
from src.utils.plot import plot_graphs
from src.utils.resource import get_available_devices
from streamlit_image_viewer import image_viewer
from streamlit_tags import st_tags
from waffle_utils.file import io, search

from .base_page import BasePage

logger = logging.getLogger(__name__)


class MenuPage(BasePage):
    # render
    def render_sampling(self):
        st.header("Sampling")
        st.subheader("Select Sampling Method")
        st.session_state.sampling_button_disabled = False
        # result_dir = st.text_input("Sampling Experiment Name", key="sampling_result_dir")
        method = st.selectbox(
            "Sampling Method", ["Random", "Entropy", "PL2N"], index=0, key="sampling_method"
        )

        self.render_upload_unlabeled()
        col1, col2 = st.columns([0.5, 0.5], gap="medium")
        if method != "Random":
            with col2:
                col = st.columns([0.6, 0.4], gap="small")
                with col[0]:
                    self.render_select_hub()
                with col[1]:
                    self.render_hub_info()

        with col1:
            self.render_sampling_config(method)

    def render_upload_unlabeled(self):
        st.subheader("Upload Unlabeled Data")
        data_type = st.radio(
            "Unlabled Data Type",
            ["Image files", "Zip files", "Video file"],
            index=0,
            key="unlabeled_data_type",
            horizontal=True,
        )
        if data_type == "Image files":
            st_data = st.file_uploader(
                "Upload Image files",
                type=["jpg", "jpeg", "png"],
                key="unlabeled_images",
                accept_multiple_files=True,
            )
        elif data_type == "Zip files":
            st_data = st.file_uploader(
                "Upload Image Zip files",
                type=["zip"],
                key="unlabeled_image_zip",
                accept_multiple_files=True,
            )
        elif data_type == "Video file":
            st_data = st.file_uploader(
                "Upload Video", type=["mp4", "avi", "mkv"], key="unlabeled_video"
            )
        if not st_data:
            st.error("Please upload a file")
            st.session_state.sampling_button_disabled = True

    def render_sampling_config(self, method: str) -> dict:
        st.subheader(f"Sampling Config ({method})")
        st.number_input("num_samples", value=100, key="sampling_num_samples")
        if method == "Random":
            st.number_input("seed", value=42, key="sampling_seed")
        else:
            default_params = wh.get_default_train_params(st.session_state.select_waffle_hub)
            sub_col = st.columns(2, gap="medium")
            with sub_col[0]:
                st.number_input(
                    "image_width",
                    value=int(default_params["image_size"][0]),
                    key="sampling_image_width",
                )
            with sub_col[1]:
                st.number_input(
                    "image_height",
                    value=int(default_params["image_size"][1]),
                    key="sampling_image_height",
                )

            sub_col = st.columns(2, gap="medium")
            with sub_col[0]:
                container = st.container(border=True)
                container.checkbox(
                    "letter_box", value=bool(default_params["letter_box"]), key="sampling_letterbox"
                )
            with sub_col[1]:
                if method == "PL2N":
                    container = st.container(border=True)
                    container.checkbox(
                        "Diversity Sampling", value=True, key="sampling_diversity_sampling"
                    )
            st.multiselect("device", get_available_devices(), key="sampling_device")
            st.number_input(
                "batch_size", value=int(default_params["batch_size"]), key="sampling_batch_size"
            )
            st.number_input("num_workers", value=0, key="sampling_num_workers")

    def render_select_hub(self):
        st.subheader("Select Hub")
        model_list = wh.get_hub_list(root_dir=st.session_state.waffle_hub_root_dir)

        model_infos = []
        model_captions = []
        for name in model_list:
            hub = wh.load(name, root_dir=st.session_state.waffle_hub_root_dir)
            model_status = wh.get_train_status(hub)

            if model_status.status_desc is None or model_status.status_desc in ["RUNNING", "FAILED"]:
                continue

            model_info = wh.get_model_config_dict(hub)
            if model_info["task"]:  # TODO
                pass

            model_info["status"] = model_status.status_desc
            model_infos.append(model_info)

            model_captions.append(
                f"Backend: {model_info['backend'].upper():>24}, Task: {model_info['task'].upper():>24}, Categories: {str([category['name'] for category in model_info['categories']]):>20}, Status: {model_status.status_desc if model_status else 'INIT'}"
            )

        hub_name = st.radio(
            "Select Hub", model_list, 0, captions=model_captions, key="select_waffle_hub_name"
        )
        st.session_state.select_waffle_hub = wh.load(
            hub_name, root_dir=st.session_state.waffle_hub_root_dir
        )

    def render_hub_info(self):
        st.subheader("Hub Info")
        st.write(wh.get_model_config_dict(st.session_state.select_waffle_hub))

    def render_content(self):
        tab = ui.tabs(["Sampling", "Dimensional Reduction"])
        if tab == "Sampling":
            self.render_sampling()
        elif tab == "Dimensional Reduction":
            st.info("To-do")
            st.info("Dimensional Reduction is not implemented yet.")
