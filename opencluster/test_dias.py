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

data = load_file('scripts/data/DIAS_REGION/DIAS_REGION_data_2021-10-10_18-34-40')
s = data[['ra', 'dec', 'pmra', 'pmdec', 'parallax']].to_pandas()
mask2=np.array(
    [[[0,0,0,0,0],
      [0,0,0,0,0],
      [0,0,1,0,0],
      [0,0,0,0,0],
      [0,0,0,0,0]],
      [[0,0,0,0,0],
      [0,1,1,1,0],
      [0,1,1,1,0],
      [0,1,1,1,0],
      [0,0,0,0,0]],
      [[0,0,1,0,0],
      [0,1,1,1,0],
      [1,1,0,1,1],
      [0,1,1,1,0],
      [0,0,1,0,0]],
      [[0,0,0,0,0],
      [0,1,1,1,0],
      [0,1,1,1,0],
      [0,1,1,1,0],
      [0,0,0,0,0]],
      [[0,0,0,0,0],
      [0,0,0,0,0],
      [0,0,1,0,0],
      [0,0,0,0,0],
      [0,0,0,0,0]]]
)

print('detecting')

mask2 = mask2/np.count_nonzero(mask2)
det_data = s[['pmdec', 'pmra', 'parallax', 'ra', 'dec']].to_numpy()
det_data = det_data[det_data[:,2] > 0]
det_data[:,2] = np.log10(det_data[:,2])
det_pos = det_data[:,[3,4,2]]
det_pm = det_data[:,[0,1,2]]
pm_bin = [.5, .5, .05]
pos_bin = [0.05, 0.05, .05]
res = find_clusters(
    data=det_pm,
    bin_shape=pm_bin,
    mask=mask2,
    heatmaps=False,
    #min_significance=1,
    #min_sigma_dif=2,
    #min_star_dif=5,
    #max_cluster_count=4
)
n_sigmas = 1
limits = np.dstack((res.locs - res.stds*n_sigmas, res.locs+res.stds*n_sigmas))

print('subseting')

subsets = [ subset(det_data, limits[i]) for i in range(len(limits)) ]
coords = [ (np.median(subsets[i][:,3]), np.median(subsets[i][:,4])) for i in range(len(subsets)) ]
pm = [ (res.locs[i][0], res.locs[i][1]) for i in range(len(subsets)) ]
# sub_c3 = subset(s[['pmdec', 'pmra', 'log_parallax']].to_numpy(), limits[2])
# sns.scatterplot(x=sub_c1[:,0], y=sub_c1[:,1])
# plt.show()
real = [('FSR 0775', 81.39583333333333, 34.9575, 0.10500000317891439), ('Majaess 58', 81.46666666666665, 34.875, 0.23333333333333334), ('FSR 0777', 81.87916666666665, 34.73361111111111, 0.11166666348775228), ('Stock 8', 82.02916666666667, 34.42333333333333, 0.2), ('Kronberger 1', 82.08749999999999, 34.775, 0.026666667064030966)]

ax = sns.scatterplot(x=det_data[:,3], y=det_data[:,4])

for r in real:
    ax.plot([r[1]], [r[2]], 'o', ms=60, mec='b', mfc='none')

for d in coords:
    ax.plot([d[0]], [d[1]], 'o', ms=60, mec='r', mfc='none')

plt.show()

axpm = sns.scatterplot(x=det_data[:,0], y=det_data[:,1])
for d in pm:
    axpm.plot([d[0]], [d[1]], 'o', ms=60, mec='r', mfc='none')
plt.show()
print('membership')
res2 = fuzzy_dbscan(subsets[0])
print('coso')
