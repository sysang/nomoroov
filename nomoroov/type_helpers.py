from typing import Callable


def get_dict_value(d: dict) -> Callable:
    return lambda x: d.get(x, -1)
