from dataclasses import asdict, dataclass


@dataclass
class Run:
    name: str = None
    run_type: str = None
    run_file: str = None
    status: str = None
    scheduled_time: str = None
    start_time: str = None
    end_time: str = None

    def dict(self):
        return asdict(self)


class RunType:
    TRAIN = "train"
    EVAL = "eval"
    PREDICT = "predict"
