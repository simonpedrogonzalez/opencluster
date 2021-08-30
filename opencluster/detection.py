
import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))


from opencluster.fetcher import load_file
from opencluster.synthetic import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from typing_extensions import TypedDict
from typing import Optional, Tuple, List, Union, Callable
from attr import attrib, attrs, validators
import copy
import math
from astropy.coordinates import Distance, SkyCoord
import astropy.units as u


def find_cluster(data, mask: np.ndarray, bin_size: list):
    offset = [(bin_size[i] - (data[:,i].max() - data[:,i].min()) % bin_size[i])/2 for i in range(data.shape[1])]
    ranges = [[data[:,i].min() - offset[i], data[:,i].max() + offset[i]] for i in range(data.shape[1])]
    bins = [int((ranges[i][1]-ranges[i][0])//bin_size[i]) for i in range(data.shape[1])]

    hist, edges = np.histogramdd(data, bins=bins, range=ranges, density=False)
    return hist, edges

def window3D(w):
    # Convert a 1D filtering kernel to 3D
    # eg, window3D(numpy.hanning(5))
    L=w.shape[0]
    m1=np.outer(np.ravel(w), np.ravel(w))
    win1=np.tile(m1,np.hstack([L,1,1]))
    m2=np.outer(np.ravel(w),np.ones([1,L]))
    win2=np.tile(m2,np.hstack([L,1,1]))
    win2=np.transpose(win2,np.hstack([1,2,0]))
    win=np.multiply(win1,win2)
    return win

""" mask = mask*1
mask = mask/np.count_nonzero(mask)
snH4 = signal.fftconvolve(H, mask) """

""" h = np.hanning(5)
w = window3D(h)
print(w) """

rt = 'spherical'
field = Field(
    pm=stats.multivariate_normal(mean=(0., 0.), cov=3),
    # space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=700),
    space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=3),
    representation_type=rt,
    star_count=int(60)
)
cluster = Cluster(
    space=stats.multivariate_normal(
        mean=polar_to_cartesian([120.7, -28.5, 5]),
        cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ),
    pm=stats.multivariate_normal(mean=(.5, .5), cov=.5),
    representation_type=rt,
    star_count=40
)
s = Synthetic(field=field, clusters=[cluster]).rvs()
find_cluster(s[['pmra', 'pmdec', 'log_parallax']].to_numpy(), mask=np.array([0,0,0]), bin_size=[.5, .5, .05])