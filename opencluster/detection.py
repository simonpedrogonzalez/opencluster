
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
from scipy import ndimage
from typing_extensions import TypedDict
from typing import Optional, Tuple, List, Union, Callable
from attr import attrib, attrs, validators
import copy
import math
from astropy.coordinates import Distance, SkyCoord
import astropy.units as u

def convolve(data, mask: np.ndarray=None, c_filter: callable=None, *args, **kwargs):
    if c_filter:
        return c_filter(data, *args, **kwargs)
    if mask is not None:
        return ndimage.convolve(data, mask, *args, **kwargs)

def histogram(data, bin_size: list):
    offset = [(bin_size[i] - (data[:,i].max() - data[:,i].min()) % bin_size[i])/2 for i in range(data.shape[1])]
    ranges = [[data[:,i].min() - offset[i], data[:,i].max() + offset[i]] for i in range(data.shape[1])]
    bins = [round((ranges[i][1]-ranges[i][0])/bin_size[i]) for i in range(data.shape[1])]
    hist, edges = np.histogramdd(data, bins=bins, range=ranges, density=False)
    return hist, edges

@attrs(auto_attribs=True)
class FindClusterResult:
    hist: np.ndarray
    edges: list
    max_bin: np.ndarray
    max_center: np.ndarray        

def location_estimator(data):
    return np.median(data, axis=0)

def find_peaks(image, mask, treshold):
    local_max = ndimage.maximum_filter(image, footprint=mask)
    peaks = ((local_max==image) & (image >= treshold))
    return peaks


def find_cluster(data, bin_size, treshold=50, *args, **kwargs):
    dim = data.shape[1]
    # TODO: check bin_size and data shapes
    hist, edges = histogram(data, bin_size)
    smoothed = convolve(hist, *args, **kwargs)
    sharp = hist - smoothed
    peaks = find_peaks(sharp, np.ones([3]*dim), treshold)
    max_bin = np.ravel(np.where(sharp == sharp.max()))
    max_edges = [[edges[i][max_bin[i]], edges[i][min(max_bin[i]+1, edges[i].shape[0])]] for i in range(dim)]
    initial_max_params = [edges[i][max_bin[i]] + bin_size[i]/2 for i in range(dim)]
    subset = data
    # taking 1 more bin in each direction, for each dimension
    subset_limits = [[max_edges[i][0] - bin_size[i], max_edges[i][1] + bin_size[i]] for i in range(dim)]
    # taking no extra bins
    # subset_limits = max_edges
    for i in range(dim):
        subset = subset[(subset[:,i] >= subset_limits[i][0]) & (subset[:,i] <= subset_limits[i][1])]
    max_center = location_estimator(subset)
    
    return FindClusterResult(
        hist=hist,
        edges=edges,
        max_bin=max_bin,
        max_center=max_center,
    )
    
# TODO: should use mask, e.g.: not include center pixel. idea: use generic_filter footprint?
def var_filter(data, mask=None, *args, **kwargs):
    if mask is not None:
        kwargs['footprint'] = mask!=0
    return convolve(
        data,
        c_filter=ndimage.generic_filter,
        function=np.var, *args, **kwargs,
    )

# TODO: use mask, should not include center pixel faster
def std_filter(data, mask=None, *args, **kwargs):
    if mask is not None:
        kwargs['footprint'] = mask!=0
    return convolve(
        data,
        c_filter=ndimage.generic_filter,
        function=np.std, *args, **kwargs,
    )

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

field = Field(
    pm=stats.multivariate_normal(mean=(0., 0.), cov=5),
    # space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=700),
    space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=5),
    star_count=int(5e4)
)
cluster = Cluster(
    space=stats.multivariate_normal(
        mean=polar_to_cartesian([120.7, -28.5, 5]),
        cov=.01
    ),
    pm=stats.multivariate_normal(mean=(2, 2), cov=.05),
    star_count=200
)
s = Synthetic(field=field, clusters=[cluster]).rvs()
# histogram(s[['pmra', 'pmdec', 'log_parallax']].to_numpy(), bin_size=[.5, .5, .05])
# res = find_cluster(s[['pmra', 'pmdec', 'log_parallax']].to_numpy(), [.5, .5, .05], c_filter=ndimage.gaussian_filter, sigma=1)
mask=np.array(
    [[[0,0,0],
    [0,1,0],
    [0,0,0]],
    [[0,1,0],
    [1,0,1],
    [0,1,0]],
    [[0,0,0],
    [0,1,0],
    [0,0,0]]]
    )
mask2=np.array(
    [[[0,0,1,0,0],
      [0,1,1,1,0],
      [1,1,0,1,1],
      [0,1,1,1,0],
      [0,0,1,0,0]],
      [[0,0,1,0,0],
      [0,1,1,1,0],
      [1,1,0,1,1],
      [0,1,1,1,0],
      [0,0,1,0,0]],
      [[0,0,1,0,0],
      [0,1,1,1,0],
      [1,1,0,1,1],
      [0,1,1,1,0],
      [0,0,1,0,0]]]
)
mask = mask/np.count_nonzero(mask)
mask2 = mask2/np.count_nonzero(mask2)
res = find_cluster(s[['pmra', 'pmdec', 'log_parallax']].to_numpy(), [.5, .5, .01], mask=mask2)
# res = find_cluster(s[['pmra', 'pmdec', 'log_parallax']].to_numpy(), [.5, .5, .01], c_filter=ndimage.gaussian_filter, sigma=1)

""" data = np.genfromtxt('data/detection_example.csv', delimiter=',', dtype="f8").reshape((20, 20, 20))
filtered = np.genfromtxt('data/detection_example_filtered.csv', delimiter=',', dtype="f8").reshape((20, 20, 20))
values = np.genfromtxt('data/detection_example_values.csv', delimiter=',', dtype="f8")

print('reading data...')
original_data = np.genfromtxt('scripts/data/clusters/ngc2527dr3.csv', skip_header=True, dtype='f8', delimiter=',')
# Ra  Dec  Plx  ePlx pmRa epmRa pmDec epmDec  G  BP-RP
#  0   1    2    3    4     5     6     7     8    9
original_data = original_data[:,[4,6,2]]
original_data[:,2] = np.log10(original_data[:,2])
original_data = original_data[~np.isnan(original_data).any(axis=1), :]

mask = [[[True for k in range(5)] for j in range(5)] for i in range(5)]
for i in range(5):
    for j in range(5):
            for k in range(5):
                mask[i][j][k]=(0<(i-2)**2+(j-2)**2+(k-2)**2<5)
mask=np.array(mask)
mask = mask*1
mask = mask/np.count_nonzero(mask)

find_cluster(original_data, bin_size=[.5, .5, .05], mask=mask) """

# test convolve
""" filtered2 = convolve(data, mask=mask)
variance2 = ndimage.generic_filter(data, function=np.var, size=(5,5,5))
variance = convolve(data, c_filter=ndimage.generic_filter, function=np.var, size = (5,5,5)) """
print('coso')