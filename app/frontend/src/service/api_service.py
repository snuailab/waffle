import queue
import threading
import time
from copy import copy
from multiprocessing import Process

import torch
from src.schema.run import RunInfo, RunType
from waffle_utils.logger.time import datetime_now

from .waffle_hub import get_status


class APIService:
    def __init__(self, max_run=2, max_queue=10):
        pass
