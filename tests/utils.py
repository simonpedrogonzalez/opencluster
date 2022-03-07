import pytest

def raises_exception(exception, fun):
    if exception is not None and issubclass(exception, Exception):
        with pytest.raises(exception):
            fun()
    else:
        return fun()