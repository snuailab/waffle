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
    # render
    def render_new_hub(self):
        st.subheader("New Hub")

        model_name = st.text_input("Model Name", key="waffle_hub_name")
        backend = st.selectbox("Backend", wh.get_available_backends(), key="waffle_hub_backend")
        task_type = st.selectbox(
            "Task Type", wh.get_available_tasks(backend=backend), key="waffle_hub_task_type"
        )
        model_type = st.selectbox(
            "Model Type",
            wh.get_available_model_types(backend=backend, task=task_type),
            key="waffle_hub_model_type",
        )
        model_size = st.selectbox(
            "Model Size",
            wh.get_available_model_sizes(backend=backend, task=task_type, model_type=model_type),
            key="waffle_hub_model_size",
        )
        categories = st_tags(
            label="Categories", text="Press enter to add more", key="waffle_hub_categories"
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
            st.session_state.select_waffle_hub = model_name

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

        st.radio("Select Model", model_list, 0, captions=model_captions, key="select_waffle_hub")

    def render_hub_info(self):
        st.subheader("Hub Info")
        st.write(
            wh.get_model_config_dict(
                st.session_state.select_waffle_hub, root_dir=st.session_state.waffle_hub_root_dir
            )
        )

    def render_train_config(self):
        hub_model_config = wh.get_model_config_dict(
            hub_name=st.session_state.select_waffle_hub,
            root_dir=st.session_state.waffle_hub_root_dir,
        )
        if hub_model_config is None:
            return
        dataset_list = wd.get_dataset_list(
            root_dir=st.session_state.waffle_dataset_root_dir,
            task=hub_model_config["task"],
        )
        st.selectbox("Dataset", dataset_list, key="waffle_hub_train_dataset")

        default_params = wh.get_default_train_params(
            hub_name=st.session_state.select_waffle_hub,
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
                    hub_name=st.session_state.select_waffle_hub,
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
        if not st.session_state.waffle_hub_train_dataset:
            st.error("Please select a dataset")
            st.session_state.train_button_disabled = True

        if st.session_state.train_device == []:
            st.error("Please select a device")
            st.session_state.train_button_disabled = True

        if st.session_state.waffle_hub_train_dataset:
            c = wh.get_category_names(
                st.session_state.select_waffle_hub, st.session_state.waffle_hub_root_dir
            )
            if c != []:
                if set(c) != set(
                    wd.get_category_names(
                        st.session_state.waffle_hub_train_dataset,
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
                "dataset": st.session_state.waffle_hub_train_dataset,
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
                "hub_name": st.session_state.select_waffle_hub,
                "args": kwargs,
                "root_dir": st.session_state.waffle_hub_root_dir,
            }
            run_name = st.session_state.select_waffle_hub + str(RunType.TRAIN)
            run_service.add_run(run_name, RunType.TRAIN, wh.train, run_args)
            st.info("Train Process is registered.")

        # for name, type in TrainConfig.__annotations__.items():
        #     print(name, type)
        #     generate_component(
        #         name,
        #         type,
        #         key=f"waffle_hub_train_config_{name}",
        #         default=default_params.get(name, None),
        #     )

    def render_train_result(self):
        if wh.is_trained(
            st.session_state.select_waffle_hub, root_dir=st.session_state.waffle_hub_root_dir
        ):
            st.subheader("Train Results")
            metrics = wh.get_metrics(
                st.session_state.select_waffle_hub, root_dir=st.session_state.waffle_hub_root_dir
            )
            st.write(metrics[-1])
            # x = [i + 1 for i in range(len(metrics))]
            # y = [[metric[i]["value"] for metric in metrics] for i in range(1, len(metrics[0]))]
            # labels = [metrics[0][i]["tag"] for i in range(1, len(metrics[0]))]
            # st.pyplot(plot_graphs(x, y, labels, "Metrics"))

    def render_func_config(self) -> dict:
        train_config = wh.get_train_config(
            hub_name=st.session_state.select_waffle_hub,
            root_dir=st.session_state.waffle_hub_root_dir,
        )
        col = st.columns(3, gap="medium")
        with col[0]:
            batch_size = st.number_input("batch_size", value=int(train_config["batch_size"]))
            sub_col = st.columns(2, gap="medium")
            with sub_col[0]:
                image_width = st.number_input(
                    "image_width", value=int(train_config["image_size"][0])
                )
            with sub_col[1]:
                image_height = st.number_input(
                    "image_height", value=int(train_config["image_size"][1])
                )

            sub_col = st.columns(2, gap="medium")
            with sub_col[0]:
                container = st.container(border=True)
                letter_box = container.checkbox("letter_box", value=bool(train_config["letter_box"]))
            with sub_col[1]:
                container = st.container(border=True)
                half = container.checkbox("half", value=False)
        with col[1]:
            confidence_threshold = st.slider(
                "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25
            )
            iou_threshold = st.slider("IoU Threshold", min_value=0.0, max_value=1.0, value=0.5)
            with st.expander("Advance"):
                st.info("To-do: Advance Configs")  # TODO: Advance Configs
                if train_config["advance_params"] != {}:
                    for key, value in train_config["advance_params"].items():
                        st.write(f"{key}: {value} ({type(value)})")

        with col[2]:
            if len(train_config["device"].split(",")) > 1:
                device = st.multiselect(
                    "device", get_available_devices(), default=train_config["device"].split(",")
                )
            else:
                device = st.multiselect(
                    "device", get_available_devices(), default=[train_config["device"]]
                )

            num_workers = st.number_input("num_workers", value=0)
            seed = st.number_input("seed", value=42)

        return {
            "batch_size": batch_size,
            "image_size": [image_width, image_height],
            "letter_box": letter_box,
            "half": half,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "device": "cpu" if "cpu" in device else ",".join(device),
            "advance_params": None,  # TODO: Advance Configs
            "workers": num_workers,
            "seed": seed,
        }

    def render_evaluate(self):
        hub_model_config = wh.get_model_config_dict(
            hub_name=st.session_state.select_waffle_hub,
            root_dir=st.session_state.waffle_hub_root_dir,
        )
        if hub_model_config is None:
            return
        dataset_list = wd.get_dataset_list(
            root_dir=st.session_state.waffle_dataset_root_dir,
            task=hub_model_config["task"],
        )
        col1, col2 = st.columns([0.7, 0.3], gap="medium")
        with col1:
            st.selectbox("Dataset", dataset_list, key="waffle_hub_eval_dataset")
        with col2:
            if st.session_state.waffle_hub_eval_dataset:
                set_list = wd.get_split_list(
                    st.session_state.waffle_hub_eval_dataset,
                    st.session_state.waffle_dataset_root_dir,
                )
                if "unlabeled" in set_list:
                    set_list.remove("unlabeled")
                st.selectbox(
                    "Set Name", set_list, index=len(set_list) - 1, key="waffle_hub_eval_set_name"
                )
        config = self.render_func_config()
        dataset_dict = {
            "dataset": st.session_state.waffle_hub_eval_dataset,
            "dataset_root_dir": st.session_state.waffle_dataset_root_dir,
        }
        config.update(dataset_dict)
        st.write(config)

        st.session_state.eval_button_disabled = False
        if not st.session_state.waffle_hub_eval_dataset:
            st.error("Please select a dataset")
            st.session_state.eval_button_disabled = True

        if config["device"] == "":
            st.error("Please select a device")
            st.session_state.eval_button_disabled = True

        if st.session_state.waffle_hub_eval_dataset:
            c = wh.get_category_names(
                st.session_state.select_waffle_hub, st.session_state.waffle_hub_root_dir
            )
            if c != []:
                if set(c) != set(
                    wd.get_category_names(
                        st.session_state.waffle_hub_eval_dataset,
                        st.session_state.waffle_dataset_root_dir,
                    )
                ):
                    st.error("Dataset and hub categories are not matched")
                    st.session_state.eval_button_disabled = True
        if "cpu" in config["device"]:
            st.info("CPU is selected. If you want to use GPU, please select only GPU number.")

        if st.button("Evaluate", disabled=st.session_state.eval_button_disabled):
            kwargs = config
            run_args = {
                "hub_name": st.session_state.select_waffle_hub,
                "args": kwargs,
                "root_dir": st.session_state.waffle_hub_root_dir,
            }
            run_name = st.session_state.select_waffle_hub + str(RunType.EVALUATE)
            run_service.add_run(run_name, RunType.EVALUATE, wh.evaluate, run_args)
            st.info("Evaluate Process is registered.")

    def render_evaluate_result(self):
        if wh.is_evaluated(
            st.session_state.select_waffle_hub, root_dir=st.session_state.waffle_hub_root_dir
        ):
            st.subheader("Evaluate Results")
            st.write(
                wh.get_evaluate_result(
                    st.session_state.select_waffle_hub,
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
                hub_name=st.session_state.select_waffle_hub,
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
                st.session_state.select_waffle_hub, root_dir=st.session_state.waffle_hub_root_dir
            ):
                self.render_evaluate()
                st.divider()
                self.render_evaluate_result()
            else:
                st.warning("This hub is not trained yet.")

        with infer_tab:
            st.subheader("Inference")
            if wh.is_trained(
                st.session_state.select_waffle_hub, root_dir=st.session_state.waffle_hub_root_dir
            ):
                pass
            else:
                st.warning("This hub is not trained yet.")

        with export_tab:
            st.subheader("Export Onnx")
            if wh.is_trained(
                st.session_state.select_waffle_hub, root_dir=st.session_state.waffle_hub_root_dir
            ):
                pass
            else:
                st.warning("This hub is not trained yet.")

        # if st.session_state.select_waffle_hub:
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
