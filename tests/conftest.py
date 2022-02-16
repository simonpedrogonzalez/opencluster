import numpy as np
import pytest

@pytest.fixture(scope='session', autouse=True)
def random():
    np.random.seed(0)