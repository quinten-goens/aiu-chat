"""Shared test fixtures.

The Data App client memoises entity lookups for a TTL. That cache is process-wide,
so without isolation one test's resolved entity leaks into the next one's mocked
response (which is exactly how test_resolve_entity_by_name started failing when
the cache was introduced). Clear it around every test.
"""
from __future__ import annotations

import pytest

from aiu_chat.sources import dataapp


@pytest.fixture(autouse=True)
def _clear_dataapp_caches():
    dataapp.clear_caches()
    yield
    dataapp.clear_caches()
