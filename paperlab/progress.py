from __future__ import annotations

import os
import sys
from typing import Any

from tqdm.auto import tqdm as _tqdm

_TRUE = {"1", "true", "yes", "on", "always"}
_FALSE = {"0", "false", "no", "off", "never"}


def progress_enabled() -> bool:
    """Return whether dynamic progress bars should be rendered.

    PAPERLAB_PROGRESS=auto (default): only in an interactive terminal.
    PAPERLAB_PROGRESS=on: force progress bars.
    PAPERLAB_PROGRESS=off: disable progress bars.
    """
    mode = os.environ.get("PAPERLAB_PROGRESS", "auto").strip().lower()
    if mode in _TRUE:
        return True
    if mode in _FALSE:
        return False
    return bool(sys.stderr.isatty())


def paper_tqdm(*args: Any, **kwargs: Any):
    """tqdm wrapper that stays silent in nohup logs.

    Existing training code can keep calling ``set_postfix``. When disabled,
    tqdm consumes the iterator without emitting one line per step.
    """
    kwargs.setdefault("disable", not progress_enabled())
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    kwargs.setdefault(
        "mininterval",
        float(os.environ.get("PAPERLAB_PROGRESS_MININTERVAL", "2.0")),
    )
    return _tqdm(*args, **kwargs)
