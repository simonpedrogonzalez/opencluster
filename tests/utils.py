import pytest

def test_if_raises_exception(exception, fun):
    if issubclass(exception, Exception):
        with pytest.raises(exception):
            fun()
    else:
        return fun()