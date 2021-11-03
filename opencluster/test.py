import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))

from opencluster.synthetic import *
from opencluster.detection import *
from opencluster.membership import *
from opencluster.utils import *

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

field = Field(
    pm=stats.multivariate_normal(mean=(0., 0.), cov=20),
    # space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=700),
    space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=10),
    star_count=int(1e5)
)
clusters = [
    Cluster(
        space=stats.multivariate_normal(
            mean=polar_to_cartesian([120.7, -28.5, 5]),
            cov=.5
        ),
        pm=stats.multivariate_normal(mean=(.5, 0), cov=1./35),
        star_count=200
    ),
    Cluster(
        space=stats.multivariate_normal(
            mean=polar_to_cartesian([120.7, -28.5, 5]),
            cov=.5
        ),
        pm=stats.multivariate_normal(mean=(4.5, 4), cov=1./35),
        star_count=200
    ),
    Cluster(
        space=stats.multivariate_normal(
            mean=polar_to_cartesian([120.7, -28.5, 5]),
            cov=.5
        ),
        pm=stats.multivariate_normal(mean=(7.5, 7), cov=1./35),
        star_count=200
    )
]
s = Synthetic(field=field, clusters=clusters).rvs()

print('detecting')

data = s[['pmra', 'pmdec', 'parallax']].to_numpy()

bin_shape = [1, 1, .1]

res = find_clusters(
    data=data,
    bin_shape=bin_shape,
    mask=default_mask(3),
    heatmaps=False
    )

print('subseting')
subsets = []
for p in res.peaks:
    lim = np.vstack((p.center-bin_shape, p.center+bin_shape)).T
    subsets.append(subset(data, lim))

print('membership')
res2 = membership(subsets[0], res.peaks[0].star_count)
print('coso')
