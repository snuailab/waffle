from config import Config
from fastapi import FastAPI
from api.routers import (
    hub,
    task,
)

app = FastAPI(title=Config.TITLE, description=Config.DESCRIPTION)

app.include_router(hub.router, tags=["Hub"])
app.include_router(task.router, tags=["Task"])
