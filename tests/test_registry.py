"""Tests for the source-adapter registry.

Verifies the contract used by the CLI: registration, lookup, error on
unknown names and refusal to overwrite an existing alias.
"""

from __future__ import annotations

import pytest

from climate_restore.sources import (
    SOURCE_REGISTRY,
    BaseAdapter,
    get_source,
    list_sources,
    register,
)


@pytest.fixture
def clean_alias():
    """Ensure a test alias is absent before the test and removed after."""
    alias = "_test_alias_xyz"
    SOURCE_REGISTRY.pop(alias, None)
    yield alias
    SOURCE_REGISTRY.pop(alias, None)


def test_builtin_adapters_registered():
    names = list_sources()
    assert "gfs-0p25" in names
    assert "graphcast" in names
    assert "aifs-single" in names


def test_register_and_get_source(clean_alias):
    @register(clean_alias)
    class _Dummy(BaseAdapter):
        name = "dummy"

    assert get_source(clean_alias) is _Dummy
    assert clean_alias in list_sources()


def test_register_duplicate_raises(clean_alias):
    @register(clean_alias)
    class _First(BaseAdapter):
        name = "first"

    with pytest.raises(ValueError, match="already registered"):
        @register(clean_alias)
        class _Second(BaseAdapter):  # pragma: no cover - body never enters
            name = "second"

    # First registration must remain intact.
    assert get_source(clean_alias) is _First


def test_get_unknown_source_raises_with_listing():
    with pytest.raises(KeyError, match="unknown source"):
        get_source("_does_not_exist_")


def test_list_sources_sorted():
    names = list_sources()
    assert names == sorted(names)
