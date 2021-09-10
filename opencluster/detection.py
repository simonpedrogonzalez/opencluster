
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
from skimage.feature import peak_local_max
from statsmodels.robust.scale import huber, hubers_scale
from astropy.stats import biweight_location, biweight_scale, mad_std
from astropy.stats.sigma_clipping import sigma_clipped_stats

def convolve(data, mask: np.ndarray=None, c_filter: callable=None, *args, **kwargs):
    if c_filter:
        return c_filter(data, *args, **kwargs)
    if mask is not None:
        return ndimage.convolve(data, mask, *args, **kwargs)

def histogram(data, bin_shape: list):
    offset = [(bin_shape[i] - (data[:,i].max() - data[:,i].min()) % bin_shape[i])/2 for i in range(data.shape[1])]
    ranges = [[data[:,i].min() - offset[i], data[:,i].max() + offset[i]] for i in range(data.shape[1])]
    bins = [round((ranges[i][1]-ranges[i][0])/bin_shape[i]) for i in range(data.shape[1])]
    hist, edges = np.histogramdd(data, bins=bins, range=ranges, density=False)
    return hist, edges

@attrs(auto_attribs=True)
class FindClusterResult:
    locs: np.ndarray
    stds: np.ndarray
    star_counts: np.ndarray
    heatmaps=None
    kdeplots=None

def location_estimator(data):
    return np.median(data, axis=0)

def find_peaks(image, mask, threshold):
    local_max = ndimage.maximum_filter(image, footprint=mask)
    peaks = ((local_max==image) & (image >= threshold))
    return peaks

def create_heatmaps(hist, edges, bin_shape, clusters_idx):
    dim = len(hist.shape)
    labels = [(np.round(edges[i]+bin_shape[i]/2, 2))[:-1] for i in range(dim)]
    if dim == 2:
        data = hist
        annot_idx = clusters_idx
        annot = np.ndarray(shape=data.shape, dtype=str).tolist()
        for i in range(annot_idx.shape[1]):
            annot[annot_idx[0,i]][annot_idx[1,i]] = str(round(data[annot_idx[0][i]][annot_idx[1][i]]))
        ax = sns.heatmap(data, annot=annot, fmt='s', yticklabels=labels[0], xticklabels=labels[1])
        hlines = np.concatenate((annot_idx[0], annot_idx[0]+1))
        vlines = np.concatenate((annot_idx[1], annot_idx[1]+1))
        ax.hlines(hlines, *ax.get_xlim(), color='w')
        ax.vlines(vlines, *ax.get_ylim(), color='w')
    else:
        cuts = np.unique(clusters_idx[2])
        ncuts = cuts.size
        ncols = min(2, ncuts)
        nrows = math.ceil(ncuts/ncols)
        delete_last = nrows>ncuts/ncols
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows)
        for row in range(nrows):
            for col in range(ncols):
                idx = col*nrows+row
                if idx < cuts.size:
                    cut_idx = cuts[idx]
                    data = hist[:,:,cut_idx]
                    annot_idx = clusters_idx.T[(clusters_idx.T[:,2] == cut_idx)].T[:2]
                    annot = np.ndarray(shape=data.shape, dtype=str).tolist()
                    for i in range(annot_idx.shape[1]):
                        annot[annot_idx[0,i]][annot_idx[1,i]] = str(round(data[annot_idx[0][i]][annot_idx[1][i]]))
                    if ncuts <= 1:
                        subplot = ax
                    else:
                        if nrows == 1:
                            subplot = ax[col]
                        else:
                            subplot = ax[row, col]
                    current_ax = sns.heatmap(data, annot=annot, fmt='s', yticklabels=labels[0], xticklabels=labels[1], ax=subplot)
                    current_ax.axes.set_xlabel("x")
                    current_ax.axes.set_ylabel("y")
                    current_ax.title.set_text(f'z slice at value {round(edges[2][cut_idx]+bin_shape[2]/2, 4)}') 
                    hlines = np.concatenate((annot_idx[0], annot_idx[0]+1))
                    vlines = np.concatenate((annot_idx[1], annot_idx[1]+1))
                    current_ax.hlines(hlines, *current_ax.get_xlim(), color='w')
                    current_ax.vlines(vlines, *current_ax.get_ylim(), color='w')
        if delete_last:
            ax.flat[-1].set_visible(False)
        fig.subplots_adjust(wspace=.05,  hspace=.3)
    return ax



        

def find_cluster(
    data, bin_shape, threshold=50,
    estimate_loc=True, max_cluster_count=np.inf,
    heatmaps=True, kdeplots=True,
    *args, **kwargs):
    dim = data.shape[1]
    # TODO: check bin_shape and data shapes
    hist, edges = histogram(data, bin_shape)
    smoothed = convolve(hist, *args, **kwargs)
    sharp = hist - smoothed
    clusters_idx = peak_local_max(sharp, min_distance=1,
        threshold_abs=threshold, exclude_border=True,
        num_peaks=max_cluster_count).T
    locs=[]
    stds=[]
    for i in range(dim):
        if estimate_loc:
            i_edges = (edges[i][clusters_idx[i]]-bin_shape[i], edges[i][clusters_idx[i]]+bin_shape[i])
            c_loc = []
            c_std = []
            for c in range(clusters_idx.shape[1]):
                subset = data[:,i]
                subset = subset[((subset>i_edges[0][c])&(subset<i_edges[1][c]))]
                _, median, std = sigma_clipped_stats(subset, cenfunc='median', stdfunc='mad_std', maxiters=None, sigma=1)
                c_loc.append(median)
                c_std.append(std)
            locs.append(c_loc)
            stds.append(c_std)
        else:
            locs.append(edges[i][clusters_idx[i]]+bin_shape[i]/2)
            stds.append(bin_shape[i])
    star_counts = sharp[tuple(clusters_idx)]

    if heatmaps and clusters_idx.size!=0:
        heatmaps = create_heatmaps(sharp, edges, bin_shape, clusters_idx)
    
    return FindClusterResult(
        locs=np.array(locs).T,
        stds=np.array(stds).T,
        star_counts=star_counts
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
    star_count=int(1e4)
)
cluster = Cluster(
    space=stats.multivariate_normal(
        mean=polar_to_cartesian([120.7, -28.5, 5]),
        cov=.01
    ),
    pm=stats.multivariate_normal(mean=(2.1, 2.1), cov=.05),
    star_count=200
)
cluster2 = Cluster(
    space=stats.multivariate_normal(
        mean=polar_to_cartesian([119.7, -27.5, 4.93]),
        cov=.01
    ),
    pm=stats.multivariate_normal(mean=(4.1, 4.1), cov=.05),
    star_count=200
)
cluster3 = Cluster(
    space=stats.multivariate_normal(
        mean=polar_to_cartesian([119.7, -27.5, 5.05]),
        cov=.01
    ),
    pm=stats.multivariate_normal(mean=(3.9, 3.9), cov=.05),
    star_count=200
)
s = Synthetic(field=field, clusters=[cluster, cluster2, cluster3]).rvs()
# histogram(s[['pmra', 'pmdec', 'log_parallax']].to_numpy(), bin_shape=[.5, .5, .05])
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
mask3 = np.array(
    [[0,0,1,0,0],
      [0,1,1,1,0],
      [1,1,0,1,1],
      [0,1,1,1,0]]
)
mask = mask/np.count_nonzero(mask)
mask2 = mask2/np.count_nonzero(mask2)
mask3 = mask3/np.count_nonzero(mask3)
res = find_cluster(s[['pmdec', 'pmra', 'log_parallax']].to_numpy(), [.5, .5, .005], mask=mask2)
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

find_cluster(original_data, bin_shape=[.5, .5, .05], mask=mask) """

# test convolve
""" filtered2 = convolve(data, mask=mask)
variance2 = ndimage.generic_filter(data, function=np.var, size=(5,5,5))
variance = convolve(data, c_filter=ndimage.generic_filter, function=np.var, size = (5,5,5)) """
print('coso')