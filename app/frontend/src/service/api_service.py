import logging
import threading
import time
from copy import copy
from typing import Union

from src.api import api_client
from src.schema.task import TaskInfo, TaskType
from waffle_hub.hub import Hub
from waffle_utils.logger.time import datetime_now

logger = logging.getLogger(__name__)


class APIService:
    def __init__(self):
        self.task_dict = {}
        self.running_task_list = []
        self.stop = False

        self.check_task_alive_thread = threading.Thread(target=self._check_task_alive_loop)
        self.check_task_alive_thread.start()

    def __del__(self):
        self.stop = True
        self.check_task_alive_thread.join()

    def _check_task_alive_loop(self):
        while not self.stop:
            while len(self.running_task_list) > 0:
                for running_task_name in self.running_task_list:
                    task_info = self.task_dict[running_task_name]
                    async_result = api_client.get(
                        target="task", endpoint="status", params={"task_id": task_info.task_id}
                    )
                    self._log_task_info(running_task_name)
                    if async_result["status"] in ["FAILURE", "SUCCESS", "REVOKED"]:
                        task_info.end_time = datetime_now()
                        self.running_task_list.remove(running_task_name)
                        self._log_task_info(running_task_name)
                time.sleep(0.5)
            time.sleep(1)

    def _log_task_info(self, name):
        task_info = self.task_dict[name]
        status = self._get_state(
            task_info.args["hub_name"], task_info.args["hub_root_dir"], task_info.task_type
        )
        if status is None:
            return
        task_info.status = status.status_desc
        task_info.current_step = status.step
        task_info.total_step = status.total_step
        task_info.error_type = status.error_type
        task_info.error_msg = status.error_msg

    def add_task(self, name: str, task_type: Union[str, TaskType], args: dict):
        if name in list(self.task_dict.keys()):
            # log 이미 존재하는 프로세스입니다.
            name = f"{name}_{time.time():.0f}"
        task = api_client.post(target="hub", endpoint=str(task_type), params=args)
        task_info = TaskInfo(
            name=name,
            task_type=str(task_type),
            task_id=task["task_id"],
            start_time=datetime_now(),
            status="INIT",
        )
        task_info.args = args
        self.task_dict[name] = task_info
        self.running_task_list.append(name)

    def kill(self, name):
        if (name in list(self.task_dict.keys())) and (name in self.running_task_list):
            api_client.get(
                target="task", endpoint="kill", params={"task_id": self.task_dict[name].task_id}
            )
            self.task_dict[name].end_time = datetime_now()
            self.running_task_list.remove(name)
            time.sleep(0.5)
            self._log_task_info(name)

    def get_task_info(self, name):
        return self.task_dict.get(name, None)

    def get_task_list(self, task_type: str = None):
        if task_type is None:
            return [task_name for task_name in self.task_dict.keys()]
        else:
            return [
                task_name
                for task_name, task in self.task_dict.items()
                if task.task_type == task_type
            ]

    def del_task_list(self, name: str):
        if (name in self.task_dict.keys()) and (not name in self.running_task_list()):
            del self.task_dict[name]

    def get_running_task_list(self, task_type: str = None):
        if task_type is None:
            return copy(self.running_task_list)
        else:
            return [
                task_name
                for task_name in self.running_task_list
                if self.task_dict[task_name].task_type == task_type
            ]

    # temporary function
    # TODO: api 제공 시 변경 필요
    def _get_state(self, hub_name, hub_root_dir, task_type):
        hub = Hub.load(hub_name, hub_root_dir)
        if hub is not None:
            if task_type == TaskType.TRAIN:
                return hub.get_training_status()
            elif task_type == TaskType.EVALUATE:
                return hub.get_evaluating_status()
            elif task_type == TaskType.INFERENCE:
                return hub.get_inferencing_status()
            elif task_type == TaskType.EXPORT_ONNX:
                return hub.get_exporting_onnx_status()
            elif task_type == TaskType.EXPORT_WAFFLE:
                return hub.get_exporting_onnx_status()
            else:
                return None
        else:
            return None


api_service = APIService()
