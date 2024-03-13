from config import Config
from fastapi import FastAPI
from webserver.api.endpoint import (
    hub,
)

app = FastAPI(title=Config.TITLE, description=Config.DESCRIPTION)

app.include_router(hub.router, tags=["Hub"])
