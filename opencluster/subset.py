import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))

from bayes_opt import BayesianOptimization
from opencluster.fetcher import load_file
from opencluster.synthetic import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy import ndimage
from scipy.stats import gaussian_kde
from typing_extensions import TypedDict
from typing import Optional, Tuple, List, Union, Callable
from attr import attrib, attrs, validators
import copy
import math
from astropy.coordinates import Distance, SkyCoord
import astropy.units as u
from skimage.feature import peak_local_max
from statsmodels.robust.scale import huber, hubers_scale
from astropy.stats import biweight_location, biweight_scale, mad_std
from astropy.stats.sigma_clipping import sigma_clipped_stats
from hdbscan import HDBSCAN, all_points_membership_vectors
from hdbscan.validity import validity_index
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from KDEpy import FFTKDE
from opencluster.kdeh import HKDE
from sklearn.metrics import pairwise_distances

from abc import abstractmethod

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
            raise ValueError('limits must be of shape (d, 2)')

        mask = np.ones(obs, dtype=bool)

        for i in range(ldims):
            if i >= dims:
                break
            mask[(data[:,i] < limits[i][0]) | (data[:,i] > limits[i][1])] = False
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
            raise ValueError('Center must be shape (2,) or (3,) and <= data dimensions')

        obs, dims = data.shape

        if cdims == 2:
            return is_inside_circle(center, radius, data[:,0:2])
        else:
            return is_inside_sphere(center, radius, data[:,0:3])

class Cropper:
    @abstractmethod
    def crop(self, data: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass


@attrs(auto_attribs=True)
class RangeCropper(Cropper):
    limits: Union[list, np.ndarray]

    def crop(self, data: np.ndarray):
        # Crop data in a hypercube according to limits
        # data and limits must be in order
        obs, dims = data.shape
        limits = np.ndarray(self.limits)
        ldims, lrange = limits.shape
        if lrange != 2:
            raise ValueError('limits must be of shape (d, 2)')

        for i in ldims:
            if i >= dims:
                break
            data = data[(data[:,i] > limits[i][0]) & (data[:,i] < limits[i][1])]
        return data


@attrs(auto_attribs=True)
class CenterCropper(Cropper):
    center: Union[list, np.ndarray]
    radius: Union[int, float]

    def crop(self, data: np.ndarray):
        # Crop data in a circle or sphere according to limits
        # takes into account first 2 or 3 dims
        obs, dims = data.shape
        center = np.ndarray(center)
        cdims, ccols = center.shape
        if ccols > 0 or cdims not in [2, 3] or cdims > dims:
            raise ValueError('Center must be of shape [(2,), (3,)] and <= data dimensions')
        obs, dims = data.shape
        limits = np.ndarray(self.limits)
        ldims, lrange = limits.shape
        if lrange != 2:
            raise ValueError('limits must be of shape (d, 2)')

        if ldims == 2:
            return data[is_inside_circle(center, radius, data[:,0:2])]
        else:
            return data[is_inside_sphere(center, radius, data[:,0:3])]