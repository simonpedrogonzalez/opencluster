import os
import sys
from abc import abstractmethod
from typing import Union

from attr import attrs

import numpy as np

sys.path.append(os.path.join(os.path.dirname("opencluster"), "."))
from opencluster.synthetic import is_inside_circle, is_inside_sphere


class DataMasker:
    @abstractmethod
    def mask(self, data) -> np.ndarray:
        pass


@attrs(auto_attribs=True)
class RangeMasker(DataMasker):
    limits: Union[list, np.ndarray]

    def mask(self, data: np.ndarray):
        # mask data outside a hypercube according to limits
        # data and limits must be in order
        obs, dims = data.shape
        limits = np.array(self.limits)
        ldims, lrange = limits.shape
        if lrange != 2:
            raise ValueError("limits must be of shape (d, 2)")

        mask = np.ones(obs, dtype=bool)

        for i in range(ldims):
            if i >= dims:
                break
            mask[
                (data[:, i] < limits[i][0]) | (data[:, i] > limits[i][1])
            ] = False
        return mask


@attrs(auto_attribs=True)
class CenterMasker(DataMasker):
    center: Union[list, np.ndarray]
    radius: Union[int, float]

    def mask(self, data: np.ndarray):
        # Crop data in a circle or sphere according to limits
        # takes into account first 2 or 3 dims
        obs, dims = data.shape
        center = np.array(self.center)
        radius = self.radius
        cdims = center.shape[0]
        if len(center.shape) > 1 or cdims not in [2, 3] or cdims > dims:
            raise ValueError(
                "Center must be shape (2,) or (3,) and <= data dimensions"
            )

        obs, dims = data.shape

        if cdims == 2:
            return is_inside_circle(center, radius, data[:, 0:2])
        else:
            return is_inside_sphere(center, radius, data[:, 0:3])
