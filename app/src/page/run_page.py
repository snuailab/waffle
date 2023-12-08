from collections import defaultdict

import streamlit as st
from src.service.run_service import run_service
from src.util.resource import cpu_check, gpu_check, memory_check
from streamlit_autorefresh import st_autorefresh

from waffle_hub.hub import Hub
from waffle_hub.schema.running_status import TrainingStatus

from .base_page import BasePage


class RunPage(BasePage):
    def render_cpu_resource(self):
        st.subheader("CPU Resource")
        st.code(cpu_check())

    def render_memory_resource(self):
        st.subheader("Memory Resource")
        st.code(memory_check())

    def render_gpu_resource(self):
        st.subheader("GPU Resource")
        st.code(gpu_check())

    def render_train_list(self):
        st.subheader("Train List")
        hub_list = Hub.get_hub_list(root_dir=st.session_state.waffle_hub_root_dir)
        info_dict = defaultdict(list)
        info_dict["name"] = hub_list
        for hub_name in hub_list:
            hub = Hub.load(hub_name, root_dir=st.session_state.waffle_hub_root_dir)
            training_status = hub.get_training_status()
            if training_status is None:
                continue
            for key, value in training_status.to_dict().items():
                info_dict[key].append(value)

            train_config = hub.get_train_config()
            device = train_config.device
            info_dict["device"].append(device)
        st.table(info_dict)

    def render_train_kill(self):
        st.subheader("Kill")

        st.selectbox("Select", options=run_service.get_run_list(), key="run_page_train_kill_select")
        st.button(
            "Kill Selected",
            on_click=lambda: run_service.kill(st.session_state.run_page_train_kill_select),
        )

    def render_content(self):
        st_autorefresh(interval=1000 * 0.5)

        st.subheader("Resource")
        cols = st.columns(3)
        with cols[0]:
            self.render_cpu_resource()
        with cols[1]:
            self.render_memory_resource()
        with cols[2]:
            self.render_gpu_resource()

        st.divider()

        self.render_train_list()
        self.render_train_kill()

        st.divider()
