import logging
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import streamlit as st
import streamlit_shadcn_ui as ui
from src.component.auto_component import generate_component
from src.schema.run import RunType
from src.service import waffle_hub as wh
from src.service import waffle_menu as wm
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
            "Sampling Method", ["PL2N", "Random", "Entropy"], index=0, key="sampling_method"
        )

        self.render_upload_unlabeled()
        col1, col2 = st.columns([0.5, 0.5], gap="medium")
        if method != "Random":
            with col2:
                col = st.columns([0.6, 0.4], gap="small")
                with col[0]:
                    self.render_select_hub(method)
                with col[1]:
                    self.render_hub_info()

        with col1:
            self.render_sampling_config(method)

        self.render_sampling_button(method)

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
                accept_multiple_files=False,
            )
        elif data_type == "Video file":
            st_data = st.file_uploader(
                "Upload Video", type=["mp4", "avi", "mkv"], key="unlabeled_video"
            )
            st.info("To-do")
        if not st_data:
            st.error("Please upload a file")
            st.session_state.sampling_button_disabled = True

    def render_sampling_config(self, method: str):
        if st.session_state.select_waffle_hub is None:
            return
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
            if st.session_state.sampling_device == []:
                st.error("Please select a device")
                st.session_state.sampling_button_disabled = True

    def render_select_hub(self, method: str):
        st.subheader("Select Hub")
        model_list = wh.get_hub_list(root_dir=st.session_state.waffle_hub_root_dir)

        model_name_list = []
        model_infos = []
        model_captions = []
        for name in model_list:
            hub = wh.load(name, root_dir=st.session_state.waffle_hub_root_dir)
            model_status = wh.get_train_status(hub)

            if model_status is None or model_status.status_desc in ["RUNNING", "FAILED"]:
                continue

            model_info = wh.get_model_config_dict(hub)
            if not model_info["task"] in wm.get_available_tasks(method):
                continue

            model_name_list.append(name)
            model_info["status"] = model_status.status_desc
            model_infos.append(model_info)

            model_captions.append(
                f"Backend: {model_info['backend'].upper():>24}, Task: {model_info['task'].upper():>24}, Categories: {str([category['name'] for category in model_info['categories']]):>20}, Status: {model_status.status_desc if model_status else 'INIT'}"
            )

        hub_name = st.radio(
            "Select Hub", model_name_list, 0, captions=model_captions, key="select_waffle_hub_name"
        )
        st.session_state.select_waffle_hub = wh.load(
            hub_name, root_dir=st.session_state.waffle_hub_root_dir
        )

    def render_hub_info(self):
        st.subheader("Hub Info")
        st.write(wh.get_model_config_dict(st.session_state.select_waffle_hub))

    def render_sampling_button(self, method: str):
        if st.button(
            "Sampling", key="sampling_button", disabled=st.session_state.sampling_button_disabled
        ):
            with st.spinner("Sampling..."):
                with TemporaryDirectory() as temp_dir:
                    image_dir = Path(temp_dir) / "images"
                    result_dir = Path(temp_dir) / "result"
                    sample_zip_file = Path(temp_dir) / "sample.zip"
                    io.make_directory(image_dir)
                    io.make_directory(result_dir)
                    if st.session_state.unlabeled_data_type == "Image files":
                        for i, file in enumerate(st.session_state.unlabeled_images):
                            file_path = Path(image_dir) / f"image_{i}{Path(file.name).suffix}"
                            with open(file_path, "wb") as f:
                                f.write(file.read())
                    elif st.session_state.unlabeled_data_type == "Zip files":
                        temp_unlabeled_image_zip_file = NamedTemporaryFile(suffix=".zip")
                        temp_unlabeled_image_zip_file.write(
                            st.session_state.unlabeled_image_zip.read()
                        )
                        io.unzip(
                            temp_unlabeled_image_zip_file.name, image_dir, create_directory=True
                        )
                    elif st.session_state.unlabeled_data_type == "Video file":
                        st.info("To-do")
                        return

                    if method == "Random":
                        wm.random_sampling(
                            image_dir=image_dir,
                            num_samples=st.session_state.sampling_num_samples,
                            result_dir=result_dir,
                            seed=st.session_state.sampling_seed,
                        )
                    elif method == "Entropy":
                        device = (
                            "cpu"
                            if "cpu" in st.session_state.sampling_device
                            else ",".join(st.session_state.sampling_device)
                        )
                        wm.entropy_sampling(
                            image_dir=image_dir,
                            num_samples=st.session_state.sampling_num_samples,
                            result_dir=result_dir,
                            hub=st.session_state.select_waffle_hub,
                            image_size=[
                                st.session_state.sampling_image_width,
                                st.session_state.sampling_image_height,
                            ],
                            batch_size=st.session_state.sampling_batch_size,
                            device=device,
                            num_workers=st.session_state.sampling_num_workers,
                        )
                    elif method == "PL2N":
                        device = (
                            "cpu"
                            if "cpu" in st.session_state.sampling_device
                            else ",".join(st.session_state.sampling_device)
                        )
                        wm.pl2n_sampling(
                            image_dir=image_dir,
                            num_samples=st.session_state.sampling_num_samples,
                            result_dir=result_dir,
                            hub=st.session_state.select_waffle_hub,
                            image_size=[
                                st.session_state.sampling_image_width,
                                st.session_state.sampling_image_height,
                            ],
                            batch_size=st.session_state.sampling_batch_size,
                            device=device,
                            num_workers=st.session_state.sampling_num_workers,
                            diversity_sampling=st.session_state.sampling_diversity_sampling,
                        )

                    st.subheader("Results")
                    image_list = search.get_image_files(directory=result_dir / "images")
                    if len(image_list) > 1000:
                        image_list = image_list[:1000]
                    image_viewer(image_list, ncol=5, nrow=2, image_name_visible=False)

                    io.zip(result_dir, sample_zip_file, recursive=True)
                    with open(sample_zip_file, "rb") as f:
                        zip_bytes = f.read()
                    st.subheader("Download Results")
                    st.download_button(
                        label="Download Sample",
                        data=zip_bytes,
                        file_name="sample.zip",
                        key="download_sample",
                    )
                    st.write(wm.get_result_json(result_dir))
                    st.write(wm.get_sample_json(result_dir))
                    st.write(wm.get_total_json(result_dir))

    def render_content(self):
        tab = ui.tabs(["Sampling", "Dimensional Reduction"])
        if tab == "Sampling":
            self.render_sampling()
        elif tab == "Dimensional Reduction":
            st.info("To-do")
            st.info("Dimensional Reduction is not implemented yet.")
