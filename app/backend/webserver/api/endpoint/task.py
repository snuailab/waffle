from fastapi import APIRouter
from worker.worker import (
    celery_app,
    check_gpu,
    task_long,
)

router = APIRouter(prefix="/task", tags=["Task"])


@router.get("/status")
async def get_task_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task_result.state,
    }  # State: PENDING, FAILURE, SUCCESS, RETRY, REVOKED


@router.get("/kill")
async def kill_task(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    task_result.revoke(terminate=True)
    return {"task_id": task_id, "status": task_result.state}


@router.get("/long")
async def long(n: int):
    task = task_long.delay(n)
    return {"task_id": task.id}


@router.get("/check-gpu")
async def check_gpu_status():
    task = check_gpu.delay()
    return {"task_id": task.id}
