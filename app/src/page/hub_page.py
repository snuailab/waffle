import logging
from collections import defaultdict
from pathlib import Path

import streamlit as st
from src.component.auto_component import generate_component
from src.schema.run import RunType
from src.service import waffle_dataset as wd
from src.service import waffle_hub as wh
from src.service.run_service import run_service
from src.utils.plot import plot_graphs
from src.utils.resource import get_available_devices
from streamlit_tags import st_tags
from waffle_hub.schema.configs import TrainConfig

from .base_page import BasePage

logger = logging.getLogger(__name__)


class HubPage(BasePage):
    # @property
    # def app_artifact_dir(self):
    #     artifact_dir = str(Path(self.get_hub().hub_dir) / "app")
    #     Path(artifact_dir).mkdir(exist_ok=True, parents=True, mode=0o777)
    #     return artifact_dir

    # @property
    # def train_run_file(self):
    #     return str(Path(self.app_artifact_dir) / "train_run.py")

    # @property
    # def train_config_file(self):
    #     return str(Path(self.app_artifact_dir) / "train_config.yaml")

    # def generate_train_code(self, hub: Hub, dataset_name: Dataset, train_config: TrainConfig) -> str:
    #     train_dict = train_config.to_dict()
    #     # remove invalid values
    #     for key, value in train_dict.items():
    #         if value:
    #             pass
    #         else:
    #             train_dict[key] = None
    #     train_dict = {name: value for name, value in train_dict.items() if value is not None}

    #     return "\n".join(
    #         [
    #             "from waffle_hub.hub import Hub",
    #             f"Hub.load(",
    #             f'\t"{hub.name}",',
    #             f'\troot_dir="{hub.root_dir}"',
    #             f").train(",
    #             f'\tdataset="{dataset_name}",',
    #             f"\t**{train_dict},",
    #             f")",
    #         ]
    #     )

    # render
    def render_new_hub(self):
        st.subheader("New Hub")

        model_name = st.text_input("Model Name", key="waffle_model_name")
        backend = st.selectbox("Backend", wh.get_available_backends(), key="waffle_model_backend")
        task_type = st.selectbox(
            "Task Type", wh.get_available_tasks(backend=backend), key="waffle_model_task_type"
        )
        model_type = st.selectbox(
            "Model Type",
            wh.get_available_model_types(backend=backend, task=task_type),
            key="waffle_model_model_type",
        )
        model_size = st.selectbox(
            "Model Size",
            wh.get_available_model_sizes(backend=backend, task=task_type, model_type=model_type),
            key="waffle_model_model_size",
        )
        categories = st_tags(
            label="Categories", text="Press enter to add more", key="waffle_model_categories"
        )

        if st.button("Create"):
            if not model_name:
                st.error("Model Name is required")
                return
            wh.new(
                name=model_name,
                backend=backend,
                task=task_type,
                model_type=model_type,
                model_size=model_size,
                categories=categories if categories else None,
                hub_root_dir=st.session_state.waffle_hub_root_dir,
            )
            st.session_state.select_waffle_model = model_name

    def render_select_hub(self):
        st.subheader("Select Hub")
        model_list = wh.get_hub_list(root_dir=st.session_state.waffle_hub_root_dir)

        filter_maps = defaultdict(list)
        filter_map_keys = ["backend", "task", "status"]
        model_infos = []
        model_captions = []
        for name in model_list:
            model_info = wh.get_model_config_dict(
                name, root_dir=st.session_state.waffle_hub_root_dir
            )
            model_status = wh.get_train_status(name, root_dir=st.session_state.waffle_hub_root_dir)
            model_info["status"] = model_status.status_desc if model_status else "INIT"
            model_infos.append(model_info)
            for key in filter_map_keys:
                filter_maps[key].append(model_info[key])

            model_captions.append(
                f"Task: {model_info['task'].upper():>24}, Categories: {str([category['name'] for category in model_info['categories']]):>20}, Status: {model_status.status_desc if model_status else 'INIT'}"
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

    def render_hub_info(self):
        st.subheader("Hub Info")
        st.write(
            wh.get_model_config_dict(
                st.session_state.select_waffle_model, root_dir=st.session_state.waffle_hub_root_dir
            )
        )

    def render_train_config(self):
        hub_model_config = wh.get_model_config_dict(
            hub_name=st.session_state.select_waffle_model,
            root_dir=st.session_state.waffle_hub_root_dir,
        )
        if hub_model_config is None:
            return
        dataset_list = wd.get_dataset_list(
            root_dir=st.session_state.waffle_dataset_root_dir,
            task=hub_model_config["task"],
        )
        st.selectbox("Dataset", dataset_list, key="waffle_model_train_dataset")

        default_params = wh.get_default_train_params(
            hub_name=st.session_state.select_waffle_model,
            root_dir=st.session_state.waffle_hub_root_dir,
        )
        # if Path(self.train_config_file).exists():
        #     default_params.update(TrainConfig.load(self.train_config_file).to_dict())
        col = st.columns(3, gap="medium")
        with col[0]:
            st.number_input("num_epochs", value=int(default_params["epochs"]), key="train_epochs")
            st.number_input(
                "learning_rate",
                value=float(default_params["learning_rate"]),
                step=0.001,
                format="%f",
                key="train_learning_rate",
            )
            st.number_input(
                "batch_size", value=int(default_params["batch_size"]), key="train_batch_size"
            )
        with col[1]:

            sub_col = st.columns(2, gap="medium")
            with sub_col[0]:
                st.number_input(
                    "image_width",
                    value=int(default_params["image_size"][0]),
                    key="train_image_width",
                )
            with sub_col[1]:
                st.number_input(
                    "image_height",
                    value=int(default_params["image_size"][1]),
                    key="train_image_height",
                )

            container = st.container(border=True)
            container.checkbox(
                "letter_box", value=bool(default_params["letter_box"]), key="train_letterbox"
            )
            with st.expander("Advance"):
                st.info("To-do: Advance Configs")  # TODO: Advance Configs
                default_advance_params = wh.get_default_advanced_train_params(
                    hub_name=st.session_state.select_waffle_model,
                    root_dir=st.session_state.waffle_hub_root_dir,
                )
                if default_advance_params:
                    for key, value in default_advance_params.items():
                        st.write(f"{key}: {value} ({type(value)})")

        with col[2]:
            st.multiselect("device", get_available_devices(), key="train_device")
            st.number_input("num_workers", value=0, key="train_num_workers")
            st.number_input("seed", value=42, key="train_seed")

    def render_train(self):
        self.render_train_config()
        st.session_state.train_button_disabled = False
        if not st.session_state.waffle_model_train_dataset:
            st.error("Please select a dataset")
            st.session_state.train_button_disabled = True

        if st.session_state.train_device == []:
            st.error("Please select a device")
            st.session_state.train_button_disabled = True

        if st.session_state.waffle_model_train_dataset:
            c = wh.get_category_names(
                st.session_state.select_waffle_model, st.session_state.waffle_hub_root_dir
            )
            if c != []:
                if set(c) != set(
                    wd.get_category_names(
                        st.session_state.waffle_model_train_dataset,
                        st.session_state.waffle_dataset_root_dir,
                    )
                ):
                    st.error("Dataset and hub categories are not matched")
                    st.session_state.train_button_disabled = True

        if "cpu" in st.session_state.train_device:
            st.info("CPU is selected. If you want to use GPU, please select only GPU number.")

        if st.button("Train", disabled=st.session_state.train_button_disabled):
            device = (
                "cpu"
                if "cpu" in st.session_state.train_device
                else ",".join(st.session_state.train_device)
            )
            kwargs = {
                "dataset": st.session_state.waffle_model_train_dataset,
                "dataset_root_dir": st.session_state.waffle_dataset_root_dir,
                "epochs": st.session_state.train_epochs,
                "learning_rate": st.session_state.train_learning_rate,
                "batch_size": st.session_state.train_batch_size,
                "image_size": [
                    st.session_state.train_image_width,
                    st.session_state.train_image_height,
                ],
                "letter_box": st.session_state.train_letterbox,
                "device": device,
                "workers": st.session_state.train_num_workers,
                "seed": st.session_state.train_seed,
                "advance_params": None,  # TODO: Advance Configs
            }
            run_args = {
                "hub_name": st.session_state.select_waffle_model,
                "args": kwargs,
                "root_dir": st.session_state.waffle_hub_root_dir,
            }
            run_service.add_run(
                st.session_state.select_waffle_model, RunType.TRAIN, wh.train, run_args
            )
            st.write("Train Process is registered.")

        # for name, type in TrainConfig.__annotations__.items():
        #     print(name, type)
        #     generate_component(
        #         name,
        #         type,
        #         key=f"waffle_model_train_config_{name}",
        #         default=default_params.get(name, None),
        #     )

        return
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

    def render_train_result(self):
        if wh.is_trained(
            st.session_state.select_waffle_model, root_dir=st.session_state.waffle_hub_root_dir
        ):
            st.subheader("Train Results")
            metrics = wh.get_metrics(
                st.session_state.select_waffle_model, root_dir=st.session_state.waffle_hub_root_dir
            )
            x = [i + 1 for i in range(len(metrics))]
            y = [[metric[i]["value"] for metric in metrics] for i in range(1, len(metrics[0]))]
            labels = [metrics[0][i]["tag"] for i in range(1, len(metrics[0]))]
            st.pyplot(plot_graphs(x, y, labels, "Metrics"))

    def render_evaluate(self):
        if st.button("Evaluate"):
            pass

    def render_evaluate_result(self):
        if wh.is_evaluated(
            st.session_state.select_waffle_model, root_dir=st.session_state.waffle_hub_root_dir
        ):
            st.subheader("Evaluate Results")
            st.write(
                wh.get_evaluate_result(
                    st.session_state.select_waffle_model,
                    root_dir=st.session_state.waffle_hub_root_dir,
                )
            )

    def render_predict(self):
        hub = self.get_hub()
        pass

    def render_export(self):
        if st.button("Export"):
            pass

    def render_delete(self):
        agree = st.checkbox("I agree to delete this model. This action cannot be undone.")
        if st.button("Delete", disabled=not agree):
            wh.delete_hub(
                hub_name=st.session_state.select_waffle_model,
                root_dir=st.session_state.waffle_hub_root_dir,
            )
            st.success("Delete done!")
            st.rerun()

    def render_content(self):
        with st.expander("Create new Hub"):
            self.render_new_hub()
        st.divider()
        col1, col2 = st.columns([0.6, 0.4], gap="medium")
        with col1:
            self.render_select_hub()
            with st.expander("Hub delete"):
                self.render_delete()
        with col2:
            self.render_hub_info()

        st.divider()

        train_tab, eval_tab, infer_tab, export_tab = st.tabs(
            ["Train", "Evaluate", "Inference", "Export"]
        )
        with train_tab:
            st.subheader("Train")
            self.render_train()
            st.divider()
            self.render_train_result()
        with eval_tab:
            st.subheader("Evaluate")
            if wh.is_trained(
                st.session_state.select_waffle_model, root_dir=st.session_state.waffle_hub_root_dir
            ):
                self.render_evaluate()
                st.divider()
                self.render_evaluate_result()
            else:
                st.warning("This hub is not trained yet.")

        with infer_tab:
            st.subheader("Inference")
            if wh.is_trained(
                st.session_state.select_waffle_model, root_dir=st.session_state.waffle_hub_root_dir
            ):
                pass
            else:
                st.warning("This hub is not trained yet.")

        with export_tab:
            st.subheader("Export Onnx")
            if wh.is_trained(
                st.session_state.select_waffle_model, root_dir=st.session_state.waffle_hub_root_dir
            ):
                pass
            else:
                st.warning("This hub is not trained yet.")

        # if st.session_state.select_waffle_model:
        #     st.divider()

        #     col1, col2 = st.columns([0.4, 0.6], gap="medium")
        #     with col1:
        #         self.render_model_info()
        #     with col2:
        #         with st.expander("Train"):
        #             self.render_train()

        #         train_status = self.get_train_status()
        #         if train_status:
        #             with st.expander("Train Results"):
        #                 self.render_train_results()

        #             if train_status.status_desc == "SUCCESS":
        #                 with st.expander("Evaluate"):
        #                     self.render_evaluate()

        #                 with st.expander("Predict"):
        #                     self.render_predict()

        #                 with st.expander("Export"):
        #                     self.render_export()

        #         self.render_delete()
