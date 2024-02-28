from pydantic import BaseModel


class PageInfo(BaseModel):
    title: str = None
    subtitle: str = None
    description: str = None
