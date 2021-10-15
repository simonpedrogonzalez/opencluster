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

res = find_clusters(
    data=data,
    bin_shape=[1, 1, .1],
    mask=default_mask(3),
    heatmaps=False
    )

ax = sns.scatterplot(x=data[:,0], y=data[:,1])
for p in res.peaks:
    ax.plot([p.center[0]], [p.center[1]], 'o', ms=60, mec='r', mfc='none')
plt.show()


n_sigmas = 1
limits = np.dstack((res.locs - res.stds*n_sigmas, res.locs+res.stds*n_sigmas))

print('subseting')

sub_c1 = subset(s[['pmdec', 'pmra', 'log_parallax']].to_numpy(), limits[0])
sub_c2 = subset(s[['pmdec', 'pmra', 'log_parallax']].to_numpy(), limits[1])
# sub_c3 = subset(s[['pmdec', 'pmra', 'log_parallax']].to_numpy(), limits[2])
# sns.scatterplot(x=sub_c1[:,0], y=sub_c1[:,1])
# plt.show()
print('membership')
res2 = fuzzy_dbscan(sub_c2)
print('coso')
