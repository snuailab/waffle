import queue
import subprocess
import sys
import threading
import time
import uuid

from src.schema.run import Run
from waffle_utils.log.time import datetime_now


class RunService:
    def __init__(self, max_run=2, max_queue=10):
        self.max_run = max_run
        self.max_queue = max_queue
        self.queue = queue.Queue(maxsize=max_queue)
        self.run_list = []
        self.run_dict = {}

        self.process_dict = {}

        self.stop = False
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def __del__(self):
        self.stop = True
        self.thread.join()

    def add_run(self, name, run_type, run_file):
        run = Run(
            name=name,
            run_type=run_type,
            run_file=run_file,
            status="waiting",
            scheduled_time=datetime_now(),
        )
        self.run_list.append(run)
        self.run_dict[name] = run
        self.queue.put(run)

    def get_run(self, name):
        return self.run_dict.get(name, None)

    def get_run_list(self):
        return [run.name for run in self.run_list]

    def loop(self):
        while not self.stop:
            while not self.queue.empty():
                if len(self.process_dict) < self.max_run:
                    run = self.queue.get()
                    self.run(run.name, run.run_file)
                else:
                    break
            time.sleep(1)

    def run(self, name, run_file):
        process = subprocess.Popen(
            [sys.executable, run_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        self.process_dict[name] = process

    def kill(self, name):
        process = self.process_dict.get(name, None)
        if process:
            process.kill()
            del self.process_dict[name]


run_service = RunService()
