"""
PATE utility subset for baseline6.

Source:
- /home/tyt21/PATE/pate/PATE_utils.py
- Upstream project: https://github.com/Raminghorbanii/PATE
- License: MIT

This file keeps only lightweight helpers needed by baseline6 detection.
"""

from __future__ import annotations

from itertools import groupby
from operator import itemgetter

import numpy as np


def convert_vector_to_events_PATE(vector_array: np.ndarray) -> list[tuple[int, int]]:
    """
    Convert a binary 1D array to contiguous anomaly ranges.
    """
    positive_indexes = np.where(vector_array > 0)[0]
    events: list[tuple[int, int]] = []
    for _, grp in groupby(enumerate(positive_indexes), lambda ix: ix[0] - ix[1]):
        cur = list(map(itemgetter(1), grp))
        events.append((int(cur[0]), int(cur[-1])))
    return events


def generate_buffer_points(
    max_buffer_size: int, num_splits: int, include_zero: bool = True
) -> np.ndarray:
    """
    Generate evenly spaced buffer points in [0, max_buffer_size].
    """
    if max_buffer_size <= 0:
        return np.array([0], dtype=int)
    if num_splits <= 0:
        num_splits = 1

    if include_zero:
        start_point = 0
        num_points = num_splits + 1
    else:
        start_point = max_buffer_size / num_splits
        num_points = num_splits

    return np.linspace(start_point, max_buffer_size, num=num_points, dtype=int)
