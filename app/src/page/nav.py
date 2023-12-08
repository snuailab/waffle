from collections import OrderedDict, namedtuple

from src.schema import PageInfo

from . import DatasetPage, HubPage, PlayGround, RunPage

ComponentInfo = namedtuple("ComponentInfo", ["page_class", "page_info"])

pages = OrderedDict(
    {
        "Dataset": ComponentInfo(
            DatasetPage,
            PageInfo(title="Dataset", subtitle=None, description="This is Waffle Dataset!"),
        ),
        "Model": ComponentInfo(
            HubPage,
            PageInfo(title="Model", subtitle=None, description="This is Waffle Hub!"),
        ),
        "Playground": ComponentInfo(
            PlayGround,
            PageInfo(title="Playground", subtitle=None, description=None),
        ),
        "Run": ComponentInfo(
            RunPage,
            PageInfo(title="Run", subtitle=None, description=None),
        ),
    },
)


def get_page_list():
    return list(pages.keys())


def nav(name: str):
    component_info = pages[name]
    return component_info.page_class(**component_info.page_info.dict())()
