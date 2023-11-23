from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from src.page.nav import get_page_list, nav
from waffle_utils.log import initialize_logger

initialize_logger("./logs/app.log", "INFO")


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

page = nav(current_page)
