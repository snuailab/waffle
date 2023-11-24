import logging
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import streamlit as st
from src.component.auto_component import generate_component
from src.util.plot import plot_bar
from streamlit_image_viewer import image_viewer
from streamlit_tags import st_tags
from waffle_utils.file import io

from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub
from waffle_hub.schema.configs import TrainConfig
from waffle_hub.schema.fields import Image

from .base_page import BasePage

logger = logging.getLogger(__name__)


class HubPage(BasePage):
    @property
    def app_artifact_dir(self):
        artifact_dir = str(Path(self.get_hub().hub_dir) / "app")
        Path(artifact_dir).mkdir(exist_ok=True, parents=True, mode=0o777)
        return artifact_dir

    @property
    def train_run_file(self):
        return str(Path(self.app_artifact_dir) / "train_run.py")

    @property
    def train_config_file(self):
        return str(Path(self.app_artifact_dir) / "train_config.yaml")

    # utils
    def get_hub(self) -> Hub:
        model_name = st.session_state.select_waffle_model
        if model_name not in Hub.get_hub_list(root_dir=st.session_state.waffle_hub_root_dir):
            return None
        return Hub.load(model_name, root_dir=st.session_state.waffle_hub_root_dir)

    def generate_train_code(self, hub: Hub, dataset_name: Dataset, train_config: TrainConfig) -> str:
        train_dict = train_config.to_dict()
        # remove invalid values
        for key, value in train_dict.items():
            if value:
                pass
            else:
                train_dict[key] = None
        train_dict = {name: value for name, value in train_dict.items() if value is not None}

        return "\n".join(
            [
                "from waffle_hub.hub import Hub",
                f"Hub.load(",
                f'\t"{hub.name}",',
                f'\troot_dir="{hub.root_dir}"',
                f").train(",
                f'\tdataset="{dataset_name}",',
                f"\t**{train_dict},",
                f")",
            ]
        )

    def get_train_status(self) -> dict:
        hub = self.get_hub()
        if hub.training_info_file.exists():
            return self.get_hub().get_training_info()
        else:
            return None

    # render
    def render_new_model(self):
        st.subheader("New Model")

        model_name = st.text_input("Model Name", key="waffle_model_name")
        backend = st.selectbox("Backend", Hub.get_available_backends(), key="waffle_model_backend")
        task_type = st.selectbox(
            "Task Type", Hub.get_available_tasks(backend=backend), key="waffle_model_task_type"
        )
        model_type = st.selectbox(
            "Model Type",
            Hub.get_available_model_types(backend=backend, task=task_type),
            key="waffle_model_model_type",
        )
        model_size = st.selectbox(
            "Model Size",
            Hub.get_available_model_sizes(backend=backend, task=task_type, model_type=model_type),
            key="waffle_model_model_size",
        )
        categories = st_tags(
            label="Categories", text="Press enter to add more", key="waffle_model_categories"
        )

        if st.button("Create"):
            if not model_name:
                st.error("Model Name is required")
                return
            Hub.new(
                name=model_name,
                backend=backend,
                task=task_type,
                model_type=model_type,
                model_size=model_size,
                categories=categories if categories else None,
                hub_root_dir=st.session_state.waffle_hub_root_dir,
            )
            st.session_state.select_waffle_model = model_name

    def render_select_model(self):
        model_list = Hub.get_hub_list(root_dir=st.session_state.waffle_hub_root_dir)
        st.selectbox("Select Model", model_list, index=0, key="select_waffle_model")

    def render_model_info(self):
        hub = self.get_hub()
        st.write(hub.get_model_config().to_dict())

    def render_train(self):
        hub = self.get_hub()
        dataset_list = []
        for dataset_name in Dataset.get_dataset_list(
            root_dir=st.session_state.waffle_dataset_root_dir
        ):
            dataset = Dataset.load(dataset_name, root_dir=st.session_state.waffle_dataset_root_dir)
            if dataset.task.lower() == hub.task.lower():
                dataset_list.append(dataset_name)
        dataset_name = st.selectbox("Dataset", dataset_list, key="waffle_model_train_dataset")

        default_params = hub.get_default_train_params(
            backend=hub.backend, task=hub.task, model_type=hub.model_type, model_size=hub.model_size
        ).to_dict()
        if Path(self.train_config_file).exists():
            default_params.update(TrainConfig.load(self.train_config_file).to_dict())

        for name, type in TrainConfig.__annotations__.items():
            generate_component(
                name,
                type,
                key=f"waffle_model_train_config_{name}",
                default=default_params.get(name, None),
            )

        if st.button("Generate Code"):
            train_config = TrainConfig(
                **{
                    name: st.session_state[f"waffle_model_train_config_{name}"]
                    for name in TrainConfig.__annotations__.keys()
                    if hasattr(st.session_state, f"waffle_model_train_config_{name}")
                }
            )
            train_config.save_yaml(self.train_config_file)

            code = self.generate_train_code(hub, dataset_name, train_config)
            with open(self.train_run_file, "w") as f:
                f.write(code)

        initialized = Path(self.train_run_file).exists()
        if initialized:
            st.code(open(self.train_run_file).read())

        train_status = self.get_train_status()
        if train_status:
            st.subheader("Train Status")
            st.write(train_status.to_dict())

        trainable = initialized and (not train_status or train_status.status == "INIT")

        if st.button("Run", disabled=not trainable):
            import subprocess
            import sys

            subprocess.Popen([sys.executable, self.train_run_file])

    def render_evaluate(self):
        if st.button("Evaluate"):
            pass

    def render_predict(self):
        if st.button("Predict"):
            pass

    def render_export(self):
        if st.button("Export"):
            pass

    def render_content(self):
        with st.expander("Create new Model"):
            self.render_new_model()

        self.render_select_model()

        if st.session_state.select_waffle_model:
            with st.expander("Model Info"):
                self.render_model_info()

            with st.expander("Train"):
                self.render_train()

            train_status = self.get_train_status()
            if train_status and train_status.status == "SUCCESS":
                with st.expander("Evaluate"):
                    self.render_evaluate()

                with st.expander("Predict"):
                    self.render_predict()

                with st.expander("Export"):
                    self.render_export()
