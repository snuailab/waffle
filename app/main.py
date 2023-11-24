import os

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from src.page.nav import get_page_list, nav
from waffle_utils.log import initialize_logger

from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub

# initialize_logger("./logs/app.log", "INFO")


st.set_page_config(
    "Waffle App",
    page_icon="🧇",
    layout="wide",
)

st.sidebar.title("Waffle App")
current_page = st.sidebar.selectbox(
    label="Select Page",
    options=get_page_list(),
)

st.sidebar.divider()
st.sidebar.subheader("Settings")
st.sidebar.text_input(
    "Dataset Root Dir",
    value=Dataset.parse_root_dir(os.getenv("WAFFLE_DATASET_ROOT_DIR", None)),
    key="waffle_dataset_root_dir",
)
st.sidebar.text_input(
    "Hub Root Dir",
    value=Hub.parse_root_dir(os.getenv("WAFFLE_HUB_ROOT_DIR", None)),
    key="waffle_hub_root_dir",
)

st.sidebar.divider()

page = nav(current_page)
