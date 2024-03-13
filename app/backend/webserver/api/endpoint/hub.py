from fastapi import APIRouter
from worker.worker import (
    celery_app,
    task_long,
    train,
)

router = APIRouter(prefix="/hub", tags=["Hub"])


@router.get("/long")
async def long(n: int):
    task = task_long.delay(n)
    return {"task_id": task.id}


@router.post("/train")
async def train_task(train_args: dict):
    task = train.delay(train_args)
    return {"task_id": task.id}


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": task_result.state}  # State: PENDING, FAILURE, SUCCESS
