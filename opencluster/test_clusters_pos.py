import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))

from opencluster.synthetic import *
from opencluster.detection import *
from opencluster.membership import *
from opencluster.fetcher import *

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
from pprint import pprint

test_clusters = [
    {
        'name': 'Stock 8',
        'description': 'rafael',
        'filters': { 'parallax': '> 0'},
        'area_radius': 1,
    },
    { 
        'name': 'ngc2527',
        'description': 'OK' ,
        'filters': { 'parallax': '> 0.2'},
        'area_radius': 2.5,
    },
    { 
        'name': 'ic2395',
        'description': 'interesting structures',
        'filters': {
            'parallax': '> 0',
            'phot_g_mean_mag': '< 17'
            },
        'area_radius': 4
    },
    { 
        'name': 'ngc2323',
        'description': 'OK',
        'filters': {
            'parallax': '> 0',
            'phot_g_mean_mag': '< 18'
            },
        'area_radius': 0.7
    },
    # { 'name': 'ic4665', 'description': 'Ok with 2.8 plx' },
    { 
        'name': 'ic2602',
        'description': 'small star count',
        'filters': {
            'parallax': '> 0',
            'phot_g_mean_mag': '< 18'
        },
        'area_radius': 0.7
    },
    { 
        'name': 'ngc2168',
        'description': 'OK',
        'filters': {
            'parallax': '> 0',
            'phot_g_mean_mag': '< 18'
        },
        'area_radius': 0.5
    }
]

i = 2
cluster = test_clusters[i]

print('reading file')
data = load_file(f'/home/simon/repos/opencluster/scripts/data/{cluster.get("name")}.vot').to_pandas()

detection_data = data[['pmra', 'pmdec', 'log10_parallax']].to_numpy()
increase = [.01, .01, 0]
max_bin_shape = np.array([1, 1, .05])
bin_shape = np.array([ .3, .3, .05])
best_shape = None
best_significance = 0
cont = 0
while not np.any(bin_shape > max_bin_shape):
    bin_shape = bin_shape+increase
    print(f'iter: {str(cont)}')
    cont+=1
    res = find_clusters(
        data=detection_data,
        bin_shape=bin_shape,
        mask=default_mask(3)
    )
    if len(res.peaks) > 0 and res.peaks[0].significance > best_significance:
        best_significance = res.peaks[0].significance
        best_shape = bin_shape

print(best_shape)
print(best_significance)
