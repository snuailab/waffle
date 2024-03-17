import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import PIL
import streamlit as st
from streamlit_ace import st_ace

from .base_page import BasePage

logger = logging.getLogger(__name__)


class PlayGround(BasePage):
    @property
    def example_code_dir(self):
        return "example_code"

    def get_code_list(self):
        return list(filter(lambda x: Path(x).is_dir(), Path(self.example_code_dir).glob("*")))

    def _image_viewer(self, title, image):
        st.subheader(title)
        h, w = image.shape[:2]
        st.image(image, width=min(700, w), channels="BGR")

    def render_upload_image(self):
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"], key="playground_uploaded_image"
        )
        if uploaded_file is not None:
            st.session_state.playground_input_image = cv2.cvtColor(
                np.array(PIL.Image.open(uploaded_file)), cv2.COLOR_RGB2BGR
            )
        else:
            st.session_state.playground_input_image = cv2.imread("samples/people.jpg")

    def render_input_image(self):
        if "playground_input_image" not in st.session_state:
            return
        self._image_viewer("Input Image", st.session_state.playground_input_image)

    def render_code_input(self):
        st.subheader("Code Input")

        with st.expander("Example Codes"):
            code_dir_list = self.get_code_list()
            code_name_list = [code.name for code in code_dir_list]
            code_name = st.selectbox("Select Example Code", options=code_name_list)
            if code_name:
                code_path = Path(self.example_code_dir) / code_name / "code.py"
                input_image_path = Path(self.example_code_dir) / code_name / "input.jpg"
                output_image_path = Path(self.example_code_dir) / code_name / "output.jpg"
                code_base = code_path.read_text()
                st.text_area("Code", value=code_base, height=300, key="playground_code_example")
                columns = st.columns(2)
                with columns[0]:
                    st.image(str(input_image_path), use_column_width=True, channels="BGR")
                with columns[1]:
                    st.image(str(output_image_path), use_column_width=True, channels="BGR")

            if st.button("Apply Example Code"):
                st.session_state.playground_code = st.session_state.playground_code_example

        st.text_area("Code", height=300, key="playground_code")

        if st.button("Run"):
            try:
                image = st.session_state.playground_input_image
                data = {"image": image}
                exec(st.session_state.playground_code, data)
                st.session_state.playground_result_image = data["image"]
            except Exception as e:
                st.error(e)

    def render_result_image(self):
        if "playground_result_image" not in st.session_state:
            return
        self._image_viewer("Result Image", st.session_state.playground_result_image)

    def render_register_code(self):
        if "playground_result_image" not in st.session_state:
            return
        st.text_input(
            "Register Code Name",
            value=f"code_{len(self.get_code_list())}",
            key="playground_code_name",
        )
        if st.button("Register"):
            code_path = (
                Path(self.example_code_dir) / st.session_state.playground_code_name / "code.py"
            )
            input_image_path = (
                Path(self.example_code_dir) / st.session_state.playground_code_name / "input.jpg"
            )
            output_image_path = (
                Path(self.example_code_dir) / st.session_state.playground_code_name / "output.jpg"
            )
            code_path.parent.mkdir(parents=True, exist_ok=True)
            with open(code_path, "w") as f:
                f.write(st.session_state.playground_code)
            cv2.imwrite(str(input_image_path), st.session_state.playground_input_image)
            cv2.imwrite(str(output_image_path), st.session_state.playground_result_image)
            st.rerun()

    def render_content(self):
        self.render_upload_image()
        self.render_input_image()
        self.render_code_input()
        self.render_result_image()
        self.render_register_code()
