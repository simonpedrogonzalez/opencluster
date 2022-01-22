
import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))

from opencluster.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy import ndimage
from typing_extensions import TypedDict
from typing import Optional, Tuple, List, Union, Callable, Type, Optional
from attr import attrib, attrs, validators
import copy
import math
from astropy.coordinates import Distance, SkyCoord
import astropy.units as u
from skimage.feature import peak_local_max
from statsmodels.robust.scale import huber, hubers_scale
from astropy.stats import biweight_location, biweight_scale, mad_std
from astropy.stats.sigma_clipping import sigma_clipped_stats
from warnings import warn

def default_mask(dim: int):
    indexes = np.array(np.meshgrid(*np.tile(np.arange(5), (dim, 1)))).T.reshape((-1, dim))
    mask = np.zeros([5]*dim)
    cond = np.sum((indexes-2)**2, axis=1)
    mask[tuple(indexes[np.argwhere((cond>0)&(cond<5))].reshape((-1, dim)).T)] = 1
    return mask/np.count_nonzero(mask)

def convolve(data, mask: np.ndarray=None, c_filter: callable=None, *args, **kwargs):
    if c_filter:
        return c_filter(data, *args, **kwargs)
    if mask is not None:
        return ndimage.convolve(data, mask, *args, **kwargs)

def histogram(data, bin_shape: Union[list, np.ndarray], nyquist_offset: list=None):
    dim = data.shape[1]
    offset = [(bin_shape[i] - (data[:,i].max() - data[:,i].min()) % bin_shape[i])/2 for i in range(dim)]
    ranges = [[data[:,i].min() - offset[i], data[:,i].max() + offset[i]] for i in range(dim)]
    if nyquist_offset is not None:
        ranges = [[ranges[i][0]+nyquist_offset[i], ranges[i][1]+nyquist_offset[i]] for i in range(dim)]
    bins = [round((ranges[i][1]-ranges[i][0])/bin_shape[i]) for i in range(dim)]
    hist, edges = np.histogramdd(data, bins=bins, range=ranges, density=False)
    return hist, edges

def nyquist_offsets(bin_shape: list):
    dim = len(bin_shape)
    values = np.vstack((np.array(bin_shape)/2, np.zeros(dim))).T
    combinations = np.array(np.meshgrid(*values)).T.reshape((-1,dim))
    return np.flip(combinations, axis=0)

@attrs(auto_attribs=True)
class Peak():
    index: np.ndarray
    significance: Union[float, int]
    count: Union[float, int]
    center: np.ndarray
    sigma: Optional[np.ndarray]

    def is_in_neighbourhood(self, b):
        return not np.any(np.abs(self.index-b.index) > 1)

@attrs(auto_attribs=True)
class FindClustersResult:
    peaks: list=[]
    heatmaps=None

def best_peaks(peaks: list):
    bests = [peaks[0]]
    peaks = peaks[1:]
    while len(peaks) > 0:
        peak = peaks.pop()
        i = 0
        while(i < len(bests)):
            if bests[i].is_in_neighbourhood(peak):
                if bests[i].significance < peak.significance:
                    bests[i] = peak
                break
            i+=1
        if i == len(bests):
            bests.append(peak)
    return bests

            
    

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
        ncols = min(3, ncuts)
        nrows = math.ceil(ncuts/ncols)
        delete_last = nrows>ncuts/ncols
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*8,nrows*5))
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
        fig.subplots_adjust(wspace=.1,  hspace=.3)
        plt.tight_layout()
    return ax


# TODO: improve memory usage
def count_based_outlier_removal(data, bin_shape, mask, min_count):
    
    dim = np.atleast_2d(data).shape[1]
    mask_shape = np.array(mask.shape)
    
    hist, edges = histogram(data, bin_shape)
    idcs = np.argwhere(hist>=min_count)
    idx_lim = np.vstack((idcs.min(axis=0), idcs.max(axis=0))).T

    idx_lim[:,0] = np.clip(idx_lim[:,0]-mask.shape, 0, None)
    idx_lim[:,1] = np.clip(idx_lim[:,1]+mask.shape, None, hist.shape)

    value_lim = [(edges[i][idx_lim[i][0]], edges[i][min(idx_lim[i][1]+1, hist.shape[i])]) for i in range(dim)]
    data = subset(data, value_lim)
    
    if data.shape[0] == 0:
        raise Exception('No bin passed minimum density check. Check min_count parameter.')
    return data, hist, edges

def find_clusters(
    data: np.ndarray,
    bin_shape: Union[list, np.ndarray],
    mask: Union[list, np.ndarray] = None,
    nyquist_offset=True,
    min_count:Union[int, float]=5,
    min_dif:Union[int, float]=10,
    min_sigma_dif:Union[int, float]=None,
    min_significance:Union[int, float]=1,
    max_num_peaks:Union[int, float]=np.inf,
    min_interpeak_dist:Union[int, float]=1,
    heatmaps=False,
    *args, **kwargs):

    dim = np.atleast_2d(data).shape[1]

    if mask is None:
        mask = default_mask(dim)

    mask = np.array(mask)
    bin_shape = np.array(bin_shape)
    if len(mask.shape) != dim:
        raise ValueError('mask does not match data dimensions')
    if len(bin_shape) != dim:
        raise ValueError('bin_shape does not match data dimensions')

    # outlier removal
    if min_count:
        data, hist, edges = count_based_outlier_removal(data, bin_shape, mask, min_count)
    
    # calculate possible bin offsets
    if(nyquist_offset):
        offsets = nyquist_offsets(bin_shape)
    else:
        offsets = np.atleast_2d(np.zeros(dim))
    
    # set detection parameters for all runs
    peak_detection_params = { 'exclude_border' : True }
    if min_interpeak_dist:
        peak_detection_params['min_distance'] = min_interpeak_dist
    if min_significance:
        peak_detection_params['threshold_abs'] = min_significance
    if max_num_peaks:
        peak_detection_params['num_peaks'] = max_num_peaks
    if np.any(np.array(hist.shape) < 3):
        warn(f'histogram has too few bins in some dimensions: hist shape is {hist.shape}')
        peak_detection_params['exclude_border'] = False

    # detect
    peaks = []
    for offset in offsets:

        hist, edges = histogram(data, bin_shape, offset)
        smoothed = convolve(hist, mask=mask)
        sharp = hist - smoothed
        std = fast_std_filter(hist, mask=mask)
        normalized = sharp/(std+1)

        if min_dif is not None:
            # check for other way to implement
            normalized[sharp < min_dif] = 0
        if min_sigma_dif is not None:
            normalized[sharp < min_sigma_dif*std] = 0

        clusters_idx = peak_local_max(
            normalized,
            **peak_detection_params
        ).T
        
        peak_count = clusters_idx.shape[1]
        
        if peak_count != 0:
        
            counts = sharp[tuple(clusters_idx)]
            significance = normalized[tuple(clusters_idx)]
            limits = [
                    [(edges[i][clusters_idx[i][j]]-bin_shape[i], edges[i][clusters_idx[i][j]]+bin_shape[i])
                    for i in range(dim)] for j in range(peak_count)
                ]
            subsets = [subset(data, limits[j]) for j in range(peak_count)]
            statitstics = np.array([
                [sigma_clipped_stats(subsets[j][:,i], cenfunc='median', 
                    stdfunc='mad_std', maxiters=None, sigma=1)
                for i in range(dim)] for j in range(peak_count)
                ])
            current_peaks = [
                Peak(
                    index=clusters_idx[:,i].T,
                    significance=significance[i],
                    count=counts[i],
                    center=statitstics[i,:,1],
                    sigma=np.array(bin_shape),
                ) for i in range(peak_count) ]
            peaks += current_peaks

    if len(peaks) == 0:
        return FindClustersResult()

    global_peaks = best_peaks(peaks)
    global_peaks.sort(key=lambda x: x.significance, reverse=True)
    
    if max_num_peaks != np.inf:
        global_peaks = global_peaks[0:max_num_peaks]
    
    res = FindClustersResult(
        peaks=global_peaks
    )
    if heatmaps:
        res.heatmaps = create_heatmaps(sharp, edges, bin_shape, clusters_idx)
    return res
    
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

def fast_std_filter(data, mask):
    u_x2 = convolve(data, mask=mask)
    ux_2 = convolve(data*data, mask=mask)
    return ((ux_2 - u_x2*u_x2)**.5)

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

""" field = Field(
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
# res = find_clusters(s[['pmra', 'pmdec', 'log_parallax']].to_numpy(), [.5, .5, .05], c_filter=ndimage.gaussian_filter, sigma=1)
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
res = find_clusters(s[['pmdec', 'pmra', 'log_parallax']].to_numpy(), [.5, .5, .005], mask=mask2, heatmaps=True)
plt.show() """
# res = find_clusters(s[['pmra', 'pmdec', 'log_parallax']].to_numpy(), [.5, .5, .01], c_filter=ndimage.gaussian_filter, sigma=1)

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

find_clusters(original_data, bin_shape=[.5, .5, .05], mask=mask) """

# test convolve
""" filtered2 = convolve(data, mask=mask)
variance2 = ndimage.generic_filter(data, function=np.var, size=(5,5,5))
variance = convolve(data, c_filter=ndimage.generic_filter, function=np.var, size = (5,5,5)) """

def test_detection():
    n_field = int(1e3)
    min_f, max_f = (0, 10)
    min_c, max_c = (4.5, 5.5)
    dens_f = n_field/(max_f - min_f)**3
    n_cluster = int(1e2)
    dens_c = n_cluster/(max_c - min_c)**3

    field = np.random.uniform(min_f, max_f, (n_field, 3))
    cluster = np.random.uniform(min_c, max_c, (n_cluster, 3))
    data = np.vstack((field, cluster))
    sns.scatterplot(data[:,0], data[:,1])
    res = find_clusters(data, [1,1,1])
    print(res.peaks[0])
    print('coso')

# test_detection()

    