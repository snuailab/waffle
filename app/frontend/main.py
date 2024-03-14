from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from src.page.nav import get_page_list, nav
from src.service import waffle_dataset as wd
from src.service import waffle_hub as wh
from waffle_utils.logger import initialize_logger

initialize_logger(
    "logs/app.log", root_level="WARNING", console_level="WARNING", file_level="WARNING"
)

st.set_page_config(
    "Waffle App",
    page_icon="🧇",
    layout="wide",
)

DATASET_ROOT_PATH = Path("/workspace/datasets")
HUB_ROOT_PATH = Path("/workspace/hubs")
SHARE_TEMP_ROOT_PATH = Path("/workspace/share_temp")
st.session_state.share_temp_root_path = SHARE_TEMP_ROOT_PATH

st.sidebar.title("Waffle App")
current_page = st.sidebar.selectbox(
    label="Select Page",
    options=get_page_list(),
)

st.sidebar.divider()
st.sidebar.subheader("Settings")
st_waffle_dataset_root_dir = st.sidebar.text_input(
    "Dataset Root Dir",
    value=wd.get_parse_root_dir(),
)
st.session_state.waffle_dataset_root_dir = DATASET_ROOT_PATH / st_waffle_dataset_root_dir
st_waffle_hub_root_dir = st.sidebar.text_input(
    "Hub Root Dir",
    value=wh.get_parse_root_dir(),
)
st.session_state.waffle_hub_root_dir = HUB_ROOT_PATH / st_waffle_hub_root_dir
st.sidebar.divider()

page = nav(current_page)
