from __future__ import annotations

import pytest

from perceptimg._composition import ensure_default_wiring, reset_default_wiring_for_tests
from perceptimg.core.optimizer import (
    set_default_engine_provider,
    set_default_image_io_provider,
)


@pytest.fixture(autouse=True)
def _default_runtime_wiring() -> None:
    """Keep default wiring available for tests that exercise internal modules."""
    reset_default_wiring_for_tests()
    set_default_engine_provider(None)
    set_default_image_io_provider(None)
    ensure_default_wiring()
    yield
    reset_default_wiring_for_tests()
    set_default_engine_provider(None)
    set_default_image_io_provider(None)
