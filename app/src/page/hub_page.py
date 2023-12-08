import logging
from collections import defaultdict
from pathlib import Path

import streamlit as st
from src.component.auto_component import generate_component
from src.schema.run import RunType
from src.service.run_service import run_service
from streamlit_tags import st_tags

from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub
from waffle_hub.schema.configs import TrainConfig

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
        if hub.training_status_file.exists():
            return hub.get_training_status()
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
        st.subheader("Select Model")
        model_list = Hub.get_hub_list(root_dir=st.session_state.waffle_hub_root_dir)

        filter_maps = defaultdict(list)
        model_infos = []
        model_captions = []
        for name in model_list:
            hub = Hub.load(name, root_dir=st.session_state.waffle_hub_root_dir)

            model_info = hub.get_model_config().to_dict()
            model_infos.append(model_info)
            for key, value in model_info.items():
                if isinstance(value, (str, int, float)) and key != "name":
                    filter_maps[key].append(value)

            model_captions.append(
                f"Task: {hub.task.upper():>24}, Categories: {str(hub.categories):>20}"
            )

        for key, value in filter_maps.items():
            filter_maps[key] = list(set(value))

        filter_key = st.selectbox("filter key", ["All"] + list(filter_maps.keys()), key="filter_key")
        if filter_key and filter_key != "All":
            values = list(filter_maps[st.session_state.filter_key])
            st.multiselect("filter value", values, default=values, key="filter_value")

            filtered_model_index = []
            for i, model_info in enumerate(model_infos):
                if model_info[st.session_state.filter_key] in st.session_state.filter_value:
                    filtered_model_index.append(i)

            model_list = [model_list[i] for i in filtered_model_index]
            model_captions = [model_captions[i] for i in filtered_model_index]

        st.radio("Select Model", model_list, 0, captions=model_captions, key="select_waffle_model")

    def render_model_info(self):
        hub = self.get_hub()
        st.subheader("Model Info")
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

        trainable = initialized and (not train_status or train_status.status_desc == "INIT")

        if st.button("Run", disabled=not trainable):
            run_service.add_run(hub.name, RunType.TRAIN, self.train_run_file)

    def render_train_results(self):
        hub = self.get_hub()

        st.write(hub.get_training_status().to_dict())
        st.write(hub.get_evaluate_result())

    def render_evaluate(self):
        if st.button("Evaluate"):
            pass

    def render_predict(self):
        hub = self.get_hub()
        pass

    def render_export(self):
        if st.button("Export"):
            pass

    def render_delete(self):
        hub = self.get_hub()
        agree = st.checkbox("I agree to delete this model. This action cannot be undone.")
        if st.button("Delete", disabled=not agree):
            hub.delete_hub()
            st.success("Delete done!")
            st.experimental_rerun()

    def render_content(self):
        with st.expander("Create new Model"):
            self.render_new_model()
        st.divider()
        self.render_select_model()
        if st.session_state.select_waffle_model:
            st.divider()

            col1, col2 = st.columns([0.4, 0.6], gap="medium")
            with col1:
                self.render_model_info()
            with col2:
                with st.expander("Train"):
                    self.render_train()

                train_status = self.get_train_status()
                if train_status:
                    with st.expander("Train Results"):
                        self.render_train_results()

                    if train_status.status_desc == "SUCCESS":
                        with st.expander("Evaluate"):
                            self.render_evaluate()

                        with st.expander("Predict"):
                            self.render_predict()

                        with st.expander("Export"):
                            self.render_export()

                self.render_delete()
