from collections import OrderedDict, namedtuple

from src.schema import PageInfo

from . import DatasetPage, HubPage, MenuPage, MenuTestPage, PlayGround, RunPage

ComponentInfo = namedtuple("ComponentInfo", ["page_class", "page_info"])

pages = OrderedDict(
    {
        "Dough": ComponentInfo(
            DatasetPage,
            PageInfo(title="Dough", subtitle="", description="This is Waffle Dataset!"),
        ),
        "Hub": ComponentInfo(
            HubPage,
            PageInfo(title="Hub", subtitle="", description="This is Waffle Hub!"),
        ),
        "Menu": ComponentInfo(
            MenuPage,
            PageInfo(title="Menu", subtitle="", description="This is Waffle Menu!"),
        ),
        "Menu-Test": ComponentInfo(
            MenuTestPage,
            PageInfo(title="Menu-Test", subtitle="", description="This is Waffle Menu Test Page!"),
        ),
        # "Playground": ComponentInfo(
        #     PlayGround,
        #     PageInfo(title="Playground", subtitle="", description=""),
        # ),
        "Run": ComponentInfo(
            RunPage,
            PageInfo(title="Run", subtitle="", description=""),
        ),
    },
)


def get_page_list():
    return list(pages.keys())


def nav(name: str):
    component_info = pages[name]
    return component_info.page_class(**component_info.page_info.dict())()
