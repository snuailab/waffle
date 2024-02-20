from collections import defaultdict
from dataclasses import asdict

import streamlit as st
from src.schema.run import RunType
from src.service import waffle_hub as wh
from src.service.run_service import run_service
from src.utils.resource import cpu_check, gpu_check, memory_check
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
        st.subheader("Train Process List")
        train_run_list = run_service.get_run_list(RunType.TRAIN)
        info_dict = defaultdict(list)
        for run_name in train_run_list:
            run = run_service.get_run(run_name)
            run_info = run["run_info"]
            for key, value in asdict(run_info).items():
                info_dict[key].append(value)
        option_set = set(run_service.get_run_list(RunType.TRAIN)) - set(
            run_service.get_running_process_name_list(RunType.TRAIN)
        )

        col1, col2 = st.columns([0.8, 0.2], gap="medium")
        with col1:
            st.table(info_dict)
        with col2:
            st.selectbox(
                "Remove Train Status",
                options=list(option_set),
                key="run_page_train_remove_status_select",
            )
            st.button(
                "Delete Train Status",
                on_click=lambda: run_service.del_run_list(
                    st.session_state.run_page_train_remove_status_select
                ),
            )

    def render_train_kill(self):
        st.subheader("Kill Train Process")
        st.selectbox(
            "Select",
            options=run_service.get_running_process_name_list(RunType.TRAIN),
            key="run_page_train_kill_select",
        )
        st.button(
            "Train Kill",
            on_click=lambda: run_service.kill(st.session_state.run_page_train_kill_select),
        )

    def render_eval_list(self):
        st.subheader("Evaluate Process List")
        eval_run_list = run_service.get_run_list(RunType.EVALUATE)
        info_dict = defaultdict(list)
        for run_name in eval_run_list:
            run = run_service.get_run(run_name)
            run_info = run["run_info"]
            for key, value in asdict(run_info).items():
                info_dict[key].append(value)
        option_set = set(run_service.get_run_list(RunType.EVALUATE)) - set(
            run_service.get_running_process_name_list(RunType.EVALUATE)
        )

        col1, col2 = st.columns([0.8, 0.2], gap="medium")
        with col1:
            st.table(info_dict)
        with col2:
            st.selectbox(
                "Remove Evaluate Status",
                options=list(option_set),
                key="run_page_eval_remove_status_select",
            )
            st.button(
                "Delete Evaluate Status",
                on_click=lambda: run_service.del_run_list(
                    st.session_state.run_page_eval_remove_status_select
                ),
            )

    def render_eval_kill(self):
        st.subheader("Kill Evaluate Process")
        st.selectbox(
            "Select",
            options=run_service.get_running_process_name_list(RunType.EVALUATE),
            key="run_page_eval_kill_select",
        )
        st.button(
            "Evaluate Kill",
            on_click=lambda: run_service.kill(st.session_state.run_page_eval_kill_select),
        )

    def render_infer_list(self):
        st.subheader("Inference Process List")
        infer_run_list = run_service.get_run_list(RunType.INFERENCE)
        info_dict = defaultdict(list)
        for run_name in infer_run_list:
            run = run_service.get_run(run_name)
            run_info = run["run_info"]
            for key, value in asdict(run_info).items():
                info_dict[key].append(value)
        option_set = set(run_service.get_run_list(RunType.INFERENCE)) - set(
            run_service.get_running_process_name_list(RunType.INFERENCE)
        )

        col1, col2 = st.columns([0.8, 0.2], gap="medium")
        with col1:
            st.table(info_dict)
        with col2:
            st.selectbox(
                "Remove Inference Status",
                options=list(option_set),
                key="run_page_infer_remove_status_select",
            )
            st.button(
                "Delete Inference Status",
                on_click=lambda: run_service.del_run_list(
                    st.session_state.run_page_infer_remove_status_select
                ),
            )

    def render_infer_kill(self):
        st.subheader("Kill Inference Process")
        st.selectbox(
            "Select",
            options=run_service.get_running_process_name_list(RunType.INFERENCE),
            key="run_page_infer_kill_select",
        )
        st.button(
            "Inference Kill",
            on_click=lambda: run_service.kill(st.session_state.run_page_infer_kill_select),
        )

    def render_export_onnx_list(self):
        st.subheader("Export Onnx Process List")
        export_onnx_run_list = run_service.get_run_list(RunType.EXPORT_ONNX)
        info_dict = defaultdict(list)
        for run_name in export_onnx_run_list:
            run = run_service.get_run(run_name)
            run_info = run["run_info"]
            for key, value in asdict(run_info).items():
                info_dict[key].append(value)
        option_set = set(run_service.get_run_list(RunType.EXPORT_ONNX)) - set(
            run_service.get_running_process_name_list(RunType.EXPORT_ONNX)
        )

        col1, col2 = st.columns([0.8, 0.2], gap="medium")
        with col1:
            st.table(info_dict)
        with col2:
            st.selectbox(
                "Remove Export Onnx Status",
                options=list(option_set),
                key="run_page_export_onnx_remove_status_select",
            )
            st.button(
                "Delete Export Onnx Status",
                on_click=lambda: run_service.del_run_list(
                    st.session_state.run_page_export_onnx_remove_status_select
                ),
            )

    def render_export_onnx_kill(self):
        st.subheader("Kill Export Onnx Process")
        st.selectbox(
            "Select",
            options=run_service.get_running_process_name_list(RunType.EXPORT_ONNX),
            key="run_page_export_onnx_kill_select",
        )
        st.button(
            "Export Onnx Kill",
            on_click=lambda: run_service.kill(st.session_state.run_page_export_onnx_kill_select),
        )

    def render_export_waffle_list(self):
        st.subheader("Export Waffle Process List")
        export_waffle_run_list = run_service.get_run_list(RunType.EXPORT_WAFFLE)
        info_dict = defaultdict(list)
        for run_name in export_waffle_run_list:
            run = run_service.get_run(run_name)
            run_info = run["run_info"]
            for key, value in asdict(run_info).items():
                info_dict[key].append(value)
        option_set = set(run_service.get_run_list(RunType.EXPORT_WAFFLE)) - set(
            run_service.get_running_process_name_list(RunType.EXPORT_WAFFLE)
        )

        col1, col2 = st.columns([0.8, 0.2], gap="medium")
        with col1:
            st.table(info_dict)
        with col2:
            st.selectbox(
                "Remove Export Waffle Status",
                options=list(option_set),
                key="run_page_export_waffle_remove_status_select",
            )
            st.button(
                "Delete Export Waffle Status",
                on_click=lambda: run_service.del_run_list(
                    st.session_state.run_page_export_waffle_remove_status_select
                ),
            )

    def render_export_waffle_kill(self):
        st.subheader("Kill Export Waffle Process")
        st.selectbox(
            "Select",
            options=run_service.get_running_process_name_list(RunType.EXPORT_WAFFLE),
            key="run_page_export_waffle_kill_select",
        )
        st.button(
            "Export Waffle Kill",
            on_click=lambda: run_service.kill(st.session_state.run_page_export_waffle_kill_select),
        )

    def render_content(self):
        st_autorefresh(interval=1000)

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
        self.render_eval_list()
        self.render_infer_list()
        self.render_export_onnx_list()
        self.render_export_waffle_list()

        st.divider()
        cols = st.columns(5)
        with cols[0]:
            self.render_train_kill()
        with cols[1]:
            self.render_eval_kill()
        with cols[2]:
            self.render_infer_kill()
        with cols[3]:
            self.render_export_onnx_kill()
        with cols[4]:
            self.render_export_waffle_kill()

        st.divider()
