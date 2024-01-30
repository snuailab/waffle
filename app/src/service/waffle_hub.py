import os

from waffle_hub.hub import Hub


def get_parse_root_dir():
    return Hub.parse_root_dir(os.getenv("WAFFLE_HUB_ROOT_DIR", None))
