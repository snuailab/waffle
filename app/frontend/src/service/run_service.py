import queue
import threading
import time
from copy import copy
from multiprocessing import Process

import torch
from src.schema.run import RunInfo, RunType
from waffle_utils.logger.time import datetime_now

from .waffle_hub import get_status


class RunService:
    def __init__(self, max_run=2, max_queue=10):
        self.max_run = max_run
        self.max_queue = max_queue
        self.queue = queue.Queue(maxsize=max_queue)

        self.stop = False
        self.run_dict = {}
        self.run_loop_thread = threading.Thread(target=self.run_loop)
        self.run_loop_thread.start()

        self.running_process_dict = {}
        self.process_check_thread = threading.Thread(target=self.check_alive_loop)
        self.process_check_thread.start()

        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

    def __del__(self):
        self.stop = True
        self.run_loop_thread.join()
        self.process_check_thread.join()

    def add_run(self, name, run_type, func, args):
        if name in list(self.run_dict.keys()):
            # log 이미 존재하는 프로세스입니다.
            name = f"{name}_{time.time():.0f}"
        if self.queue.qsize() >= self.max_queue:
            # log 큐가 가득 찼습니다.
            return

        run_info = RunInfo(
            name=name,
            run_type=run_type,
            scheduled_time=datetime_now(),
            status="INIT",
        )
        run = {
            "run_info": run_info,
            "func": func,
            "args": args,
        }

        self.run_dict[name] = run
        self.queue.put(run)

    def get_run(self, name):
        return self.run_dict.get(name, None)

    def get_run_list(self, run_type: str = None):
        if run_type is None:
            return [run_name for run_name in self.run_dict.keys()]
        else:
            return [
                run_name
                for run_name, run in self.run_dict.items()
                if run["run_info"].run_type == run_type
            ]

    def del_run_list(self, name: str):
        if (name in self.run_dict.keys()) and (not name in self.get_running_process_name_list()):
            del self.run_dict[name]

    def run_loop(self):
        while not self.stop:
            while not self.queue.empty():
                if len(self.running_process_dict) < self.max_run:
                    run = self.queue.get()
                    self.run(run_info=run["run_info"], func=run["func"], args=run["args"])
                else:
                    time.sleep(0.5)
            time.sleep(1)

    def run(self, run_info, func, args):
        process = Process(target=func, kwargs=args, name=run_info.name)
        process.start()
        self._add_running_process_dict(run_info.name, process)

    def kill(self, name):
        process = self.running_process_dict.get(name, None)
        if process:
            self._del_running_process_dict(name)
            process.terminate()
            process.join(5)
            process.kill()
            process.join()
            process.close()
            self._log_run_info(name)

    def _log_run_info(self, name):
        run = self.run_dict[name]
        run_info = run["run_info"]
        status = get_status(run_info.run_type, run["args"]["hub"])
        if status is None:
            return
        run_info.status = status.status_desc
        run_info.current_step = status.step
        run_info.total_step = status.total_step
        run_info.error_type = status.error_type
        run_info.error_msg = status.error_msg

    def _add_running_process_dict(self, name, process):
        self.run_dict[name]["run_info"].start_time = datetime_now()
        self.running_process_dict[name] = process

    def _del_running_process_dict(self, name):
        self.run_dict[name]["run_info"].end_time = datetime_now()
        del self.running_process_dict[name]

    def get_running_process_name_list(self, run_type: str = None):
        if run_type is None:
            return [run_name for run_name in self.running_process_dict.keys()]
        else:
            return [
                run_name
                for run_name in self.running_process_dict.keys()
                if self.run_dict[run_name]["run_info"].run_type == run_type
            ]

    def check_alive_loop(self):
        while not self.stop:
            while len(self.running_process_dict) > 0:
                dict_keys = copy(list(self.running_process_dict.keys()))
                for key in dict_keys:
                    process = self.running_process_dict[key]
                    self._log_run_info(key)
                    if self.run_dict[key]["run_info"].status in ["SUCCESS", "FAILED", "STOPPED"]:
                        self._del_running_process_dict(key)
                        process.kill()
                        process.join()
                        process.close()
                time.sleep(0.5)
            time.sleep(1)


run_service = RunService()
