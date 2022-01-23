
import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))

from opencluster.utils import *
from opencluster.synthetic import *
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
from abc import abstractmethod
from opencluster.subset import *
from opencluster.detection import find_clusters

def get_default_mask(dim: int):
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

# unused
# TODO: should use mask, e.g.: not include center pixel. idea: use generic_filter footprint?
def var_filter(data, mask=None, *args, **kwargs):
    if mask is not None:
        kwargs['footprint'] = mask!=0
    return convolve(
        data,
        c_filter=ndimage.generic_filter,
        function=np.var, *args, **kwargs,
    )

# unused
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
    # TODO: make n dimensional (ndkernel(1dkernel, ndim))
    # TODO: make n dimensional from function (ndkernel(1dfun, bin_shape))
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

def get_histogram_bins(data: np.ndarray, bin_shape: np.ndarray, offsets: list=None):
    _, dim = data.shape
    # calculate the margins which are added to the range in order to fit a number of bins that is integer
    margins = [(bin_shape[i] - (data[:,i].max() - data[:,i].min()) % bin_shape[i])/2 for i in range(dim)]
    # add base margins
    ranges = [[data[:,i].min() - margins[i], data[:,i].max() + margins[i]] for i in range(dim)]
    if offsets is not None:
        ranges = [[ranges[i][0]+offsets[i], ranges[i][1]+offsets[i]] for i in range(dim)]
    bins = [round((ranges[i][1]-ranges[i][0])/bin_shape[i]) for i in range(dim)]
    return np.array(bins), np.array(ranges)

def histogram(data: np.ndarray, bin_shape: Union[list, np.ndarray], offsets: list=None):
    _, dim = data.shape
    bins, ranges = get_histogram_bins(data, bin_shape, offsets)
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

def get_most_significant_peaks(peaks: list):
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

@attrs(auto_attribs=True)
class FindClustersResult:
    peaks: list=[]
    heatmaps=None    

class PeakDetector:
    @abstractmethod
    def detect(data):
        pass

@attrs(auto_attribs=True)
class CountPeakDetector(PeakDetector):
    bin_shape: Union[list, np.ndarray]
    mask: Union[list, np.ndarray] = None
    nyquist_offset=True
    min_count:Union[int, float]=5
    min_dif:Union[int, float]=10
    min_sigma_dif:Union[int, float]=None
    min_significance:Union[int, float]=1
    max_num_peaks:Union[int, float]=np.inf
    min_interpeak_dist:Union[int, float]=1
    offsets=None
    
    def trim_low_density_regions(self, data:np.ndarray):
        obs, dim = data.shape

        # calculate data ranges in each dimension, taking into account that bin number must be integer
        _, ranges = get_histogram_bins(data, self.bin_shape)

        for i in range(dim):
            shifts = [self.bin_shape[i], - self.bin_shape[i]]
            for j in range(2):
                while True:
                    if ranges[i][0] >= ranges[i][1]:
                        raise Exception('No bin passed minimum density check. Check min_count parameter.')
                    slice_ranges = np.copy(ranges)
                    # if j = 0 -> upper limit = lower limit + bin shape
                    # if j = 1 -> lower limit = upper limit - bin shape
                    slice_ranges[i][int(not(j))] = slice_ranges[i][j] + shifts[j]
                    data_slice = data[RangeMasker(limits=slice_ranges).mask(data)]
                    if data_slice.shape[0] != 0:
                        slice_histogram, _ = histogram(data_slice, self.bin_shape)
                        # if any bin has min required count, stop trimming
                        if np.any(slice_histogram >= self.min_count):
                            break
                        # else, move limit towards the center and continue
                    ranges[i][j] = slice_ranges[i][int(not(j))]

        # extend ranges half mask shape in each direction so data that belongs to 
        # an invalid bin can contribute in a border valid bin when the mask is applied
        mask_shape = np.array(self.mask.shape)
        half_mask_shape = np.ceil(mask_shape/2)
        ranges[:,0] = ranges[:,0] - half_mask_shape * self.bin_shape
        ranges[:,1] = ranges[:,1] + half_mask_shape * self.bin_shape

        # trim data and return
        trimmed_data = data[RangeMasker(limits=ranges).mask(data)]
        return trimmed_data

    def set_nyquist_offsets(self):
        if not self.nyquist_offset:
            self.offsets = np.atleast_2d(np.zeros(dim))
        else:
            dim = len(self.bin_shape)
            values = np.vstack((np.array(self.bin_shape)/2, np.zeros(dim))).T
            combinations = np.array(np.meshgrid(*values)).T.reshape((-1,dim))
            self.offsets = np.flip(combinations, axis=0)

    def detect(self, data: np.ndarray):
        if len(data.shape) != 2:
            raise ValueError('data array must have 2 dimensions')
        obs, dim = data.shape

        if self.mask is None:
            mask = get_default_mask(dim)
        self.mask = np.array(mask)

        # TODO: do in init
        self.bin_shape = np.array(self.bin_shape)
        if len(mask.shape) != dim:
            raise ValueError('mask does not match data dimensions')
        if len(self.bin_shape) != dim:
            raise ValueError('bin_shape does not match data dimensions')
        
        if self.min_count:
            data = self.trim_low_density_regions(data)

        # TODO: do it in init
        self.set_nyquist_offsets()
        
        bins, ranges = get_histogram_bins(data, self.bin_shape)

        # set detection parameters for all runs
        peak_detection_params = { }
        if np.any(np.array(bins) < np.array(mask.shape)):
            warn(f'Histogram has too few bins in some dimensions: bin numbers are {bins}')
            peak_detection_params['exclude_border'] = False
        else:
            peak_detection_params['exclude_border'] = True
        if self.min_interpeak_dist:
            peak_detection_params['min_distance'] = self.min_interpeak_dist
        if self.min_significance:
            peak_detection_params['threshold_abs'] = self.min_significance
        if self.max_num_peaks:
            peak_detection_params['num_peaks'] = self.max_num_peaks
        
        # detection
        peaks = []
        for offset in self.offsets:
            hist, edges = histogram(data, self.bin_shape, offset)
            smoothed = convolve(hist, mask=mask)
            sharp = hist - smoothed
            std = fast_std_filter(hist, mask=mask)
            # +1 is added to avoid zero division
            normalized = sharp/(std+1)

            if self.min_dif is not None:
                # check for other way to implement
                normalized[sharp < self.min_dif] = 0
            if self.min_sigma_dif is not None:
                normalized[sharp < self.min_sigma_dif*std] = 0

            clusters_idx = peak_local_max(normalized, **peak_detection_params).T
            
            _, peak_count = clusters_idx.shape
            
            if peak_count != 0:
            
                counts = sharp[tuple(clusters_idx)]
                significance = normalized[tuple(clusters_idx)]
                limits = [
                        [
                            (
                                edges[i][clusters_idx[i][j]]-self.bin_shape[i],
                                edges[i][clusters_idx[i][j]]+self.bin_shape[i]
                            ) for i in range(dim)
                        ] for j in range(peak_count)
                    ]
                subsets = [subset(data, limits[j]) for j in range(peak_count)]
                
                # stats may be useless if other center and sigma are calculated afterwards
                # e.g. meanshift and profile analysis
                statitstics = np.array([
                    [
                        sigma_clipped_stats(
                            subsets[j][:,i], cenfunc='median', stdfunc='mad_std', maxiters=None, sigma=1,
                        ) for i in range(dim)
                    ] for j in range(peak_count)
                    ])

                current_peaks = [
                    Peak(
                        index=clusters_idx[:,i].T,
                        significance=significance[i],
                        count=counts[i],
                        center=statitstics[i,:,1],
                        sigma=np.array(self.bin_shape),
                    ) for i in range(peak_count) ]
                peaks += current_peaks

        if len(peaks) == 0:
            return FindClustersResult()

        # compare same peaks in different histogram offsets
        # and return most sifnificant peak for all offsets
        global_peaks = get_most_significant_peaks(peaks)
        global_peaks.sort(key=lambda x: x.significance, reverse=True)
        
        if self.max_num_peaks != np.inf:
            global_peaks = global_peaks[0:self.max_num_peaks]
        
        res = FindClustersResult(peaks=global_peaks)
        return res

def test_detection():
   
    df = three_clusters_sample()
    data = df[['pmra', 'pmdec', 'log10_parallax']].to_numpy()

    res = CountPeakDetector(bin_shape=[ .5, .5, .05], max_num_peaks=3).detect(data)
    res2 = find_clusters(data, [ .5, .5, .05], max_num_peaks=3)
    print(res.peaks[0])
    print('coso')

test_detection()

    