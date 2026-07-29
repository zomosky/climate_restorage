"""Source adapter package for restorage.

Importing this module triggers registration of every built-in adapter via
the :func:`register` decorator. External packages can register their own
adapters by importing :func:`climate_restore.sources.registry.register`
and applying it to their class — no edits to this file are required.
"""

from __future__ import annotations

from climate_restore.sources.base import BaseAdapter, SourceAdapter
from climate_restore.sources.registry import (
    SOURCE_REGISTRY,
    get_source,
    list_sources,
    register,
)

# Side-effect imports: each module's @register call populates SOURCE_REGISTRY.
from climate_restore.sources import aifs as _aifs  # noqa: F401
from climate_restore.sources import gfs as _gfs  # noqa: F401
from climate_restore.sources import icon as _icon  # noqa: F401

__all__ = [
    "BaseAdapter",
    "SOURCE_REGISTRY",
    "SourceAdapter",
    "get_source",
    "list_sources",
    "register",
]
