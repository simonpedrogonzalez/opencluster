import warnings
from opencluster.exceptions import CatalogNotFoundException


def unsilence_warnings():
    def decorator(func):
        def wrapper(*args, **kwargs):
            warnings.filterwarnings("error")
            try:
                return func(*args, **kwargs)
            except Exception as exception:
                if "No known catalog could be found" in str(exception):
                    raise CatalogNotFoundException from None
                warnings.warn(str(exception))
        return wrapper

    return decorator
