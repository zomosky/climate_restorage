"""In-process registry mapping manifest source names to adapter classes.

Adapters self-register via the :func:`register` decorator at import time.
The orchestrator never imports adapters directly — it goes through
:func:`get_source` so adding a new file under
:mod:`climate_restore.sources` plus a side-effect import in
``__init__.py`` is enough to wire it in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from climate_restore.sources.base import SourceAdapter

__all__ = ["get_source", "list_sources", "register", "SOURCE_REGISTRY"]


SOURCE_REGISTRY: dict[str, type["SourceAdapter"]] = {}

_T = TypeVar("_T", bound=type)


def register(name: str) -> Callable[[_T], _T]:
    """Register an adapter class under ``name`` (the manifest's ``source.name``).

    Raises ``ValueError`` if ``name`` is already taken so silent overrides
    cannot mask a typo when two adapter files claim the same source.
    """

    def _decorate(cls: _T) -> _T:
        if name in SOURCE_REGISTRY:
            existing = SOURCE_REGISTRY[name]
            raise ValueError(
                f"source {name!r} already registered to {existing!r}; "
                f"refusing to overwrite with {cls!r}"
            )
        SOURCE_REGISTRY[name] = cls  # type: ignore[assignment]
        return cls

    return _decorate


def get_source(name: str) -> type["SourceAdapter"]:
    """Look up the adapter class registered for ``name``."""
    try:
        return SOURCE_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(SOURCE_REGISTRY)) or "<none>"
        raise KeyError(
            f"unknown source {name!r}; registered sources: {known}"
        ) from exc


def list_sources() -> list[str]:
    return sorted(SOURCE_REGISTRY)
