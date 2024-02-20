from dataclasses import asdict, dataclass


@dataclass
class RunInfo:
    name: str = None
    run_type: str = None

    status: str = None

    scheduled_time: str = None
    start_time: str = None
    end_time: str = None

    current_step: int = None
    total_step: int = None

    error_type: str = None
    error_msg: str = None

    def to_dict(self):
        return asdict(self)


class RunType:
    TRAIN = "train"
    EVALUATE = "evaluate"
    INFERENCE = "inference"
    EXPORT_ONNX = "export_onnx"
    EXPORT_WAFFLE = "export_waffle"
