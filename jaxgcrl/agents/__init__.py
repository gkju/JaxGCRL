from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["CRL", "PPO", "SAC", "TD3"]

_AGENT_MODULES = {
    "CRL": "crl",
    "PPO": "ppo",
    "SAC": "sac",
    "TD3": "td3",
}

if TYPE_CHECKING:
    from .crl import CRL
    from .ppo import PPO
    from .sac import SAC
    from .td3 import TD3


def __getattr__(name):
    if name in _AGENT_MODULES:
        module = import_module(f"{__name__}.{_AGENT_MODULES[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
