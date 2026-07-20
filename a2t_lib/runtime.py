"""Системные гарантии на время длительной задачи."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _set_sleep_prevention(enabled: bool) -> None:
    if sys.platform != "win32":
        return
    flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED if enabled else _ES_CONTINUOUS
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except (AttributeError, OSError):
        pass


def prevent_system_sleep(func: Callable[P, R]) -> Callable[P, R]:
    """Не даёт Windows уснуть, пока функция выполняется в текущем потоке."""

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        _set_sleep_prevention(True)
        try:
            return func(*args, **kwargs)
        finally:
            _set_sleep_prevention(False)

    return wrapped
