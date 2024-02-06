from dataclasses import asdict, dataclass


@dataclass
class RunInfo:
    name: str = None
    run_type: str = None

    scheduled_time: str = None
    start_time: str = None
    end_time: str = None

    def dict(self):
        return asdict(self)


class RunType:
    TRAIN = "train"
    EVALUATE = "evaluate"
    INFERENCE = "inference"
    EXPORT_ONNX = "export_onnx"
