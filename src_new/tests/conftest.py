import pytest
from core.state import DatasetState


@pytest.fixture
def state():
    return DatasetState()
