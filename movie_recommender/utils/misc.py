from typing import Any

import numpy as np

__all__ = ['is_valid_sequence']


def is_valid_sequence(obj: Any):
    return np.iterable(obj) and not isinstance(obj, str)
