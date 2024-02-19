import queue
import threading
import time
from copy import copy
from multiprocessing import Process

import torch
from src.schema.run import RunInfo
from waffle_utils.logger.time import datetime_now


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
            torch.multiprocessing.set_start_method("forkserver")
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
        )
        run = (run_info, func, args)

        self.run_dict[name] = run
        self.queue.put(run)

    def get_run_info(self, name):
        return self.run_dict.get(name, None)[0]

    def get_run_list(self, run_type: str = None):
        if run_type is None:
            return [run_name for run_name in self.run_dict.keys()]
        else:
            return [
                run_name for run_name, run in self.run_dict.items() if run[0].run_type == run_type
            ]

    def del_run_list(self, name: str):
        if (name in self.run_dict.keys()) and (not name in self.get_running_process_name_list()):
            del self.run_dict[name]

    def run_loop(self):
        while not self.stop:
            while not self.queue.empty():
                if len(self.running_process_dict) < self.max_run:
                    run = self.queue.get()
                    self.run(run_info=run[0], func=run[1], args=run[2])
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
            time.sleep(0.5)

    def _add_running_process_dict(self, name, process):
        self.run_dict[name][0].start_time = datetime_now()
        self.running_process_dict[name] = process

    def _del_running_process_dict(self, name):
        self.run_dict[name][0].end_time = datetime_now()
        del self.running_process_dict[name]

    def get_running_process_name_list(self):
        return [run_name for run_name in self.running_process_dict.keys()]

    def check_alive_loop(self):
        # poll: error = 1, success = 0, kill and wait = None
        while not self.stop:
            while len(self.running_process_dict) > 0:
                dict_keys = copy(list(self.running_process_dict.keys()))
                for key in dict_keys:
                    name = key
                    process = self.running_process_dict[name]
                    if not process.is_alive():
                        self._del_running_process_dict(name)
                time.sleep(0.5)
            time.sleep(1)


run_service = RunService()
