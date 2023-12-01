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
        st_ace(
            value="# change `image`(cv2, BGR) as you want.\nimage = image",
            height=300,
            language="python",
            key="playground_code",
        )

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

    def render_content(self):
        self.render_upload_image()
        self.render_input_image()
        self.render_code_input()
        self.render_result_image()
