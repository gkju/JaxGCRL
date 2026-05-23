from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["agents", "envs", "utils"]

if TYPE_CHECKING:
    from . import agents, envs, utils


def __getattr__(name):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
