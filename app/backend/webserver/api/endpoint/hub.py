from fastapi import APIRouter
from webserver.schema.waffle import (
    WaffleEvaluateParams,
    WaffleExportOnnxParams,
    WaffleExportWaffleParams,
    WaffleInferenceParams,
    WaffleTrainParams,
)
from worker.worker import (
    evaluate_task,
    export_onnx_task,
    export_waffle_task,
    inference_task,
    train_task,
)

router = APIRouter(prefix="/hub", tags=["Hub"])


@router.post("/train")
async def train(data_request: WaffleTrainParams):
    hub_name = data_request.hub_name
    hub_root_dir = data_request.hub_root_dir
    train_args = {
        "dataset": data_request.dataset,
        "dataset_root_dir": data_request.dataset_root_dir,
        "epochs": data_request.epochs,
        "learning_rate": data_request.learning_rate,
        "batch_size": data_request.batch_size,
        "image_size": data_request.image_size,
        "letter_box": data_request.letter_box,
        "device": data_request.device,
        "workers": data_request.workers,
        "seed": data_request.seed,
        "advance_params": data_request.advance_params,
    }
    task = train_task.delay(hub_name, hub_root_dir, train_args)
    return {"task_id": task.id}


@router.post("/evaluate")
async def evaluate(data_request: WaffleEvaluateParams):
    hub_name = data_request.hub_name
    hub_root_dir = data_request.hub_root_dir
    evaluate_args = {
        "dataset": data_request.dataset,
        "dataset_root_dir": data_request.dataset_root_dir,
        "batch_size": data_request.batch_size,
        "image_size": data_request.image_size,
        "letter_box": data_request.letter_box,
        "half": data_request.half,
        "device": data_request.device,
        "workers": data_request.workers,
    }
    task = evaluate_task.delay(hub_name, hub_root_dir, evaluate_args)
    return {"task_id": task.id}


@router.post("/inference")
async def inference(data_request: WaffleInferenceParams):
    hub_name = data_request.hub_name
    hub_root_dir = data_request.hub_root_dir
    inference_args = {
        "source": data_request.source,
        "batch_size": data_request.batch_size,
        "image_size": data_request.image_size,
        "letter_box": data_request.letter_box,
        "half": data_request.half,
        "device": data_request.device,
        "workers": data_request.workers,
        "recursive": data_request.recursive,
        "draw": data_request.draw,
    }
    task = inference_task.delay(hub_name, hub_root_dir, inference_args)
    return {"task_id": task.id}


@router.post("/export_onnx")
async def export_onnx(data_request: WaffleExportOnnxParams):
    hub_name = data_request.hub_name
    hub_root_dir = data_request.hub_root_dir
    export_onnx_args = {
        "batch_size": data_request.batch_size,
        "image_size": data_request.image_size,
        "half": data_request.half,
        "device": data_request.device,
        "opset_version": data_request.opset_version,
    }
    task = export_onnx_task.delay(hub_name, hub_root_dir, export_onnx_args)
    return {"task_id": task.id}


@router.post("/export_waffle")
async def export_waffle(data_request: WaffleExportWaffleParams):
    hub_name = data_request.hub_name
    hub_root_dir = data_request.hub_root_dir
    task = export_waffle_task.delay(hub_name, hub_root_dir)
    return {"task_id": task.id}
