import logging
from collections import defaultdict
from pathlib import Path

import streamlit as st
import streamlit_shadcn_ui as ui
from src.component.auto_component import generate_component
from src.schema.run import RunType
from src.service import waffle_dataset as wd
from src.service import waffle_hub as wh
from src.service.run_service import run_service
from src.utils.plot import plot_graphs
from src.utils.resource import get_available_devices
from streamlit_image_viewer import image_viewer
from streamlit_tags import st_tags
from waffle_utils.file import io, search

from .base_page import BasePage

logger = logging.getLogger(__name__)


class HubPage(BasePage):
    # render
    def render_new_hub(self):
        st.subheader("New Hub")
        create_type = st.radio(
            "Create Type", ["New", "From Waffle"], key="create_type", horizontal=True
        )
        if create_type == "New":
            hub_name = st.text_input("Hub Name", key="waffle_hub_name")
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
                if not hub_name:
                    st.error("Model Name is required")
                    return
                hub = wh.new(
                    name=hub_name,
                    backend=backend,
                    task=task_type,
                    model_type=model_type,
                    model_size=model_size,
                    categories=categories if categories else None,
                    hub_root_dir=st.session_state.waffle_hub_root_dir,
                )
                st.session_state.select_waffle_hub = hub
                st.success("Create done!")

        elif create_type == "From Waffle":
            hub_name = st.text_input("Hub Name", key="waffle_hub_name")
            st_waffle_file = st.file_uploader(
                "waffle_file",
                type=["waffle"],
                key="upload_waffle_file",
                accept_multiple_files=False,
            )
            if st.button("Create", disabled=not st_waffle_file):
                if not hub_name:
                    st.error("Model Name is required")
                    return
                hub = wh.from_waffle(
                    name=hub_name,
                    hub_root_dir=st.session_state.waffle_hub_root_dir,
                    waffle_file=st_waffle_file,
                )

                st.session_state.select_waffle_hub = hub
                st.success("Create done!")

    def render_select_hub(self):
        st.subheader("Select Hub")
        model_list = wh.get_hub_list(root_dir=st.session_state.waffle_hub_root_dir)

        filter_maps = defaultdict(list)
        filter_map_keys = ["backend", "task", "status"]
        model_infos = []
        model_captions = []
        for name in model_list:
            hub = wh.load(name, root_dir=st.session_state.waffle_hub_root_dir)
            model_info = wh.get_model_config_dict(hub)
            model_status = wh.get_train_status(hub)
            model_info["status"] = model_status.status_desc if model_status else "INIT"
            model_infos.append(model_info)
            for key in filter_map_keys:
                filter_maps[key].append(model_info[key])

            model_captions.append(
                f"Backend: {model_info['backend'].upper():>24}, Task: {model_info['task'].upper():>24}, Categories: {str([category['name'] for category in model_info['categories']]):>20}, Status: {model_status.status_desc if model_status else 'INIT'}"
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

        hub_name = st.radio(
            "Select Hub", model_list, 0, captions=model_captions, key="select_waffle_hub_name"
        )
        st.session_state.select_waffle_hub = wh.load(
            hub_name, root_dir=st.session_state.waffle_hub_root_dir
        )

    def render_hub_info(self):
        st.subheader("Hub Info")
        st.write(wh.get_model_config_dict(st.session_state.select_waffle_hub))

    def render_train_config(self):
        hub_model_config = wh.get_model_config_dict(st.session_state.select_waffle_hub)
        if hub_model_config is None:
            return
        dataset_list = wd.get_dataset_list(
            root_dir=st.session_state.waffle_dataset_root_dir,
            task=hub_model_config["task"],
        )
        st.selectbox("Dataset", dataset_list, key="waffle_hub_train_dataset")

        default_params = wh.get_default_train_params(st.session_state.select_waffle_hub)
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
                    st.session_state.select_waffle_hub
                )
                if default_advance_params:
                    for key, value in default_advance_params.items():
                        st.write(f"{key}: {value} ({type(value)})")

        with col[2]:
            st.multiselect("device", get_available_devices(), key="train_device")
            st.number_input("num_workers", value=0, key="train_num_workers")
            st.number_input("seed", value=42, key="train_seed")

    def render_train(self):
        if st.session_state.select_waffle_hub is None:
            return
        self.render_train_config()
        st.session_state.train_button_disabled = False
        if not st.session_state.waffle_hub_train_dataset:
            st.error("Please select a dataset")
            st.session_state.train_button_disabled = True

        if st.session_state.train_device == []:
            st.error("Please select a device")
            st.session_state.train_button_disabled = True

        if st.session_state.waffle_hub_train_dataset:
            c = wh.get_category_names(st.session_state.select_waffle_hub)
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
                "hub": st.session_state.select_waffle_hub,
                "args": kwargs,
            }

            wh.delete_status(st.session_state.select_waffle_hub, RunType.TRAIN)
            run_name = f"{st.session_state.select_waffle_hub.name}_{str(RunType.TRAIN)}"
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
        if wh.is_trained(st.session_state.select_waffle_hub):
            st.subheader("Train Results")
            train_loss, val_loss, metrics = wh.get_metrics(st.session_state.select_waffle_hub)
            col1, col2 = st.columns([0.5, 0.5], gap="medium")
            x = [i + 1 for i in range(len(list(train_loss.items())[0][1]))]
            with col1:
                if train_loss == {}:
                    st.warning("Train Loss is empty")
                else:
                    y = []
                    labels = []
                    for k, v in train_loss.items():
                        y.append(v)
                        labels.append(k)
                    st.pyplot(plot_graphs(x, y, labels, "Train Loss"))
            with col2:
                if val_loss == {}:
                    st.warning("Val Loss is empty")
                else:
                    y = []
                    labels = []
                    for k, v in val_loss.items():
                        y.append(v)
                        labels.append(k)
                    st.pyplot(plot_graphs(x, y, labels, "Val Loss"))

            col1, col2 = st.columns([0.5, 0.5], gap="medium")
            with col1:
                if metrics == {}:
                    st.warning("Metrics is empty")
                else:
                    y = []
                    labels = []
                    for k, v in metrics.items():
                        y.append(v)
                        labels.append(k)
                    st.pyplot(plot_graphs(x, y, labels, "Metrics"))

    def render_func_config(self, func: str) -> dict:
        train_config = wh.get_train_config(st.session_state.select_waffle_hub)
        col = st.columns(3, gap="medium")
        with col[0]:
            batch_size = st.number_input(
                "batch_size", value=int(train_config["batch_size"]), key=f"{func}_batch_size"
            )
            sub_col = st.columns(2, gap="medium")
            with sub_col[0]:
                image_width = st.number_input(
                    "image_width",
                    value=int(train_config["image_size"][0]),
                    key=f"{func}_image_width",
                )
            with sub_col[1]:
                image_height = st.number_input(
                    "image_height",
                    value=int(train_config["image_size"][1]),
                    key=f"{func}_image_height",
                )

            sub_col = st.columns(2, gap="medium")
            with sub_col[0]:
                container = st.container(border=True)
                letter_box = container.checkbox(
                    "letter_box", value=bool(train_config["letter_box"]), key=f"{func}_letter_box"
                )
            with sub_col[1]:
                container = st.container(border=True)
                half = container.checkbox("half", value=False, key=f"{func}_half")
        with col[1]:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                key=f"{func}_confidence_threshold",
            )
            iou_threshold = st.slider(
                "IoU Threshold", min_value=0.0, max_value=1.0, value=0.5, key=f"{func}_iou_threshold"
            )

        with col[2]:
            if len(train_config["device"].split(",")) > 1:
                device = st.multiselect(
                    "device",
                    get_available_devices(),
                    default=train_config["device"].split(","),
                    key=f"{func}_device",
                )
            else:
                device = st.multiselect(
                    "device",
                    get_available_devices(),
                    default=[train_config["device"]],
                    key=f"{func}_device",
                )

            num_workers = st.number_input("num_workers", value=0, key=f"{func}_num_workers")

        device = "cpu" if "cpu" in device else ",".join(device)
        if "cpu" in device:
            st.info("CPU is selected. If you want to use GPU, please select only GPU number.")

        return {
            "batch_size": batch_size,
            "image_size": [image_width, image_height],
            "letter_box": letter_box,
            "half": half,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "device": device,
            "workers": num_workers,
        }

    def render_evaluate(self):
        hub_model_config = wh.get_model_config_dict(st.session_state.select_waffle_hub)
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
        config = self.render_func_config("eval")
        dataset_dict = {
            "dataset": st.session_state.waffle_hub_eval_dataset,
            "dataset_root_dir": st.session_state.waffle_dataset_root_dir,
        }
        config.update(dataset_dict)

        st.session_state.eval_button_disabled = False
        if not st.session_state.waffle_hub_eval_dataset:
            st.error("Please select a dataset")
            st.session_state.eval_button_disabled = True

        if config["device"] == "":
            st.error("Please select a device")
            st.session_state.eval_button_disabled = True

        if st.session_state.waffle_hub_eval_dataset:
            c = wh.get_category_names(st.session_state.select_waffle_hub)
            if c != []:
                if set(c) != set(
                    wd.get_category_names(
                        st.session_state.waffle_hub_eval_dataset,
                        st.session_state.waffle_dataset_root_dir,
                    )
                ):
                    st.error("Dataset and hub categories are not matched")
                    st.session_state.eval_button_disabled = True

        if st.button("Evaluate", disabled=st.session_state.eval_button_disabled):
            kwargs = config
            run_args = {
                "hub": st.session_state.select_waffle_hub,
                "args": kwargs,
            }
            wh.delete_status(st.session_state.select_waffle_hub, RunType.EVALUATE)
            run_name = f"{st.session_state.select_waffle_hub.name}_{str(RunType.EVALUATE)}"
            run_service.add_run(run_name, RunType.EVALUATE, wh.evaluate, run_args)
            st.info("Evaluate Process is registered.")

    def render_evaluate_result(self):
        if wh.is_evaluated(st.session_state.select_waffle_hub):
            st.subheader("Evaluate Results")
            st.write(wh.get_evaluate_result(st.session_state.select_waffle_hub))
            self.render_delete_result(RunType.EVALUATE)

    def render_inference(self):
        st.session_state.infer_button_disabled = False
        data_type = st.radio(
            "Data Type",
            ["Image", "Video", "Webcam"],
            index=0,
            key="inference_data_type",
            horizontal=True,
        )
        if data_type == "Image":
            st_data = st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png"],
                key="inference_image",
                accept_multiple_files=True,
            )
        elif data_type == "Video":
            st_data = st.file_uploader(
                "Upload Video", type=["mp4", "avi", "mkv"], key="inference_video"
            )
        elif data_type == "Webcam":
            st.warning("To-do")  # TODO: Webcam
            st.session_state.infer_button_disabled = True
            st.write("Webcam")
            picture = st.camera_input("Take a picture")
            if picture:
                st.image(picture)
            st_data = None

        config = self.render_func_config("infer")

        if not st_data:
            st.error("Please upload a file")
            st.session_state.infer_button_disabled = True

        if config["device"] == "":
            st.error("Please select a device")
            st.session_state.infer_button_disabled = True

        if st.button("Inference", disabled=st.session_state.infer_button_disabled):
            # save image
            source_save_dir = (
                Path(st.session_state.waffle_hub_root_dir)
                / st.session_state.select_waffle_hub.name
                / "temp_source"
            )
            if source_save_dir.exists():
                io.remove_directory(source_save_dir, recursive=True)
            io.make_directory(source_save_dir)

            if data_type == "Image":
                for i, file in enumerate(st_data):
                    file_path = source_save_dir / f"image_{i}{Path(file.name).suffix}"
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                source = source_save_dir
            elif data_type == "Video":
                file_path = source_save_dir / f"video{Path(st_data.name).suffix}"
                with open(file_path, "wb") as f:
                    f.write(st_data.read())
                source = file_path

            infer_dict = {
                "source": str(source),
                "recursive": True,
                "draw": True,
            }
            config.update(infer_dict)
            run_args = {
                "hub": st.session_state.select_waffle_hub,
                "args": config,
            }

            wh.delete_status(st.session_state.select_waffle_hub, RunType.INFERENCE)
            run_name = f"{st.session_state.select_waffle_hub.name}_{str(RunType.INFERENCE)}"
            run_service.add_run(run_name, RunType.INFERENCE, wh.inference, run_args)
            st.info("Inference Process is registered.")

    def render_inference_result(self):
        if wh.is_inferenced(st.session_state.select_waffle_hub):
            st.subheader("Inference Results")
            infer_path = (
                Path(st.session_state.waffle_hub_root_dir)
                / st.session_state.select_waffle_hub.name
                / "inferences"
            )
            image_list = search.get_image_files(directory=infer_path / "draws")
            with st.spinner("Image Loading..."):
                if len(image_list) > 1000:
                    image_list = image_list[:1000]
                image_viewer(image_list, ncol=5, nrow=2, image_name_visible=True)
            video_path = search.get_video_files(directory=infer_path)
            if video_path != []:
                st.write(str(video_path[0].absolute()))
                st.video(str(video_path[0].absolute()), format="video/avi")
            self.render_delete_result(RunType.INFERENCE)

    def render_export_onnx(self):
        train_config = wh.get_train_config(st.session_state.select_waffle_hub)
        batch_size = st.number_input(
            "batch_size", value=int(train_config["batch_size"]), key="export_batch_size"
        )
        sub_col = st.columns(2, gap="medium")
        with sub_col[0]:
            image_width = st.number_input(
                "image_width", value=int(train_config["image_size"][0]), key=f"export_image_width"
            )
        with sub_col[1]:
            image_height = st.number_input(
                "image_height", value=int(train_config["image_size"][1]), key=f"export_image_height"
            )
        opset_version = st.number_input("opset_version", value=11, key="export_opset_version")
        container = st.container(border=True)
        half = container.checkbox("half", value=False, key=f"export_half")
        if len(train_config["device"].split(",")) > 1:
            device = st.multiselect(
                "device",
                get_available_devices(),
                default=train_config["device"].split(","),
                key=f"export_device",
            )
        else:
            device = st.multiselect(
                "device",
                get_available_devices(),
                default=[train_config["device"]],
                key=f"export_device",
            )
        device = "cpu" if "cpu" in device else ",".join(device)
        if "cpu" in device:
            st.info("CPU is selected. If you want to use GPU, please select only GPU number.")

        if st.button("Export Onnx"):
            kwargs = {
                "batch_size": batch_size,
                "image_size": [image_width, image_height],
                "half": half,
                "device": device,
                "opset_version": opset_version,
            }
            run_args = {
                "hub": st.session_state.select_waffle_hub,
                "args": kwargs,
            }
            wh.delete_status(st.session_state.select_waffle_hub, RunType.EXPORT_ONNX)
            run_name = f"{st.session_state.select_waffle_hub.name}_{str(RunType.EXPORT_ONNX)}"
            run_service.add_run(run_name, RunType.EXPORT_ONNX, wh.export_onnx, run_args)
            st.info("Export Onnx Process is registered.")

    def render_export_waffle(self):
        if st.button("Export Waffle"):
            run_args = {
                "hub": st.session_state.select_waffle_hub,
            }
            wh.delete_status(st.session_state.select_waffle_hub, RunType.EXPORT_WAFFLE)
            run_name = f"{st.session_state.select_waffle_hub.name}_{str(RunType.EXPORT_WAFFLE)}"
            run_service.add_run(run_name, RunType.EXPORT_WAFFLE, wh.export_waffle, run_args)
            st.info("Export Waffle Process is registered.")

    def render_export_onnx_result(self):
        if wh.is_exported_onnx(st.session_state.select_waffle_hub):
            st.subheader("Export Onnx Results")
            onnx_file_path = wh.get_export_onnx_path(st.session_state.select_waffle_hub)
            with open(onnx_file_path, "rb") as f:
                onnx_model_bytes = f.read()
            st.download_button(
                label="Download Onnx Model",
                data=onnx_model_bytes,
                file_name=Path(onnx_file_path).name,
                key="download_onnx_model",
            )
            self.render_delete_result(RunType.EXPORT_ONNX)

    def render_export_waffle_result(self):
        if wh.is_exported_waffle(st.session_state.select_waffle_hub):
            st.subheader("Export Waffle Results")
            waffle_file_path = wh.get_export_waffle_path(st.session_state.select_waffle_hub)
            with open(waffle_file_path, "rb") as f:
                waffle_model_bytes = f.read()
            st.download_button(
                label="Download Waffle Model",
                data=waffle_model_bytes,
                file_name=Path(waffle_file_path).name,
                key="download_waffle_model",
            )
            self.render_delete_result(RunType.EXPORT_WAFFLE)

    def render_delete_hub(self):
        agree = st.checkbox("I agree to delete this hub. This action cannot be undone.")
        if st.button("Delete", disabled=not agree):
            wh.delete_hub(st.session_state.select_waffle_hub)
            st.success("Delete done!")
            st.rerun()

    def render_delete_result(self, run_type: str):
        with st.expander(f"Delete {run_type} result"):
            agree = st.checkbox(
                "I agree to delete this result. This action cannot be undone.",
                key=f"delete_{run_type}_result_checkbox",
            )
            if st.button("Delete", disabled=not agree, key=f"delete_{run_type}_result_button"):
                wh.delete_result(st.session_state.select_waffle_hub, run_type)
                st.success(f"Delete {run_type} result done!")
                st.rerun()

    def render_content(self):
        with st.expander("Create new Hub"):
            self.render_new_hub()
        st.divider()
        col1, col2 = st.columns([0.6, 0.4], gap="medium")
        with col1:
            self.render_select_hub()
            with st.expander("Hub delete"):
                self.render_delete_hub()
        with col2:
            self.render_hub_info()

        st.divider()

        tab = ui.tabs(["Train", "Evaluate", "Inference", "Export"])
        # train_tab, eval_tab, infer_tab, export_tab = st.tabs(
        #     ["Train", "Evaluate", "Inference", "Export"]
        # )
        if tab == "Train":
            st.subheader("Train")
            self.render_train()
            st.divider()
            self.render_train_result()
        elif tab == "Evaluate":
            st.subheader("Evaluate")
            if wh.is_trained(st.session_state.select_waffle_hub):
                self.render_evaluate()
                st.divider()
                self.render_evaluate_result()
            else:
                st.warning("This hub is not trained yet.")
        elif tab == "Inference":
            st.subheader("Inference")
            if wh.is_trained(st.session_state.select_waffle_hub):
                self.render_inference()
                st.divider()
                self.render_inference_result()
            else:
                st.warning("This hub is not trained yet.")
        elif tab == "Export":
            if wh.is_trained(st.session_state.select_waffle_hub):
                container = st.container(border=True)
                col1, col2 = container.columns([0.5, 0.5], gap="medium")
                with col1:
                    st.subheader("Export Onnx")
                    with st.expander("Export Onnx"):
                        self.render_export_onnx()
                with col2:
                    st.subheader("Export Waffle")
                    with st.expander("Export Waffle"):
                        self.render_export_waffle()
                container = st.container(border=True)
                col1, col2 = container.columns([0.5, 0.5], gap="medium")
                with col1:
                    self.render_export_onnx_result()
                with col2:
                    self.render_export_waffle_result()
            else:
                st.warning("This hub is not trained yet.")
