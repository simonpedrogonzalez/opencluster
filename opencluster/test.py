import sys
import os

sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))
from astropy.stats.sigma_clipping import sigma_clipped_stats
from astropy.stats import biweight_location, biweight_scale, mad_std
from statsmodels.robust.scale import huber, hubers_scale
from skimage.feature import peak_local_max
import astropy.units as u
from astropy.coordinates import Distance, SkyCoord
import math
import copy
from attr import attrib, attrs, validators
from typing import Optional, Tuple, List, Union, Callable
from typing_extensions import TypedDict
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from opencluster.utils import *
from opencluster.membership import *
from opencluster.detection import *
from opencluster.synthetic import *
from sklearn.metrics import confusion_matrix, accuracy_score, normalized_mutual_info_score



field = Field(
    pm=stats.multivariate_normal(mean=(0., 0.), cov=20),
    # space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=700),
    space=UniformSphere(center=polar_to_cartesian(
        (120.5, -27.5, 5)), radius=10),
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

detection_data = s[['pmra', 'pmdec', 'log10_parallax']].to_numpy()
data = s[[
    'pmra', 'pmdec', 'log10_parallax', 'ra', 'dec', 'parallax',
    'p_cluster1', 'p_cluster2', 'p_cluster3', 'p_field']].to_numpy()

bin_shape = [1, 1, .05]

res = find_clusters(
    data=detection_data,
    bin_shape=bin_shape,
    min_sigma_dif=None,
)

sigma_multi = 1.5

print('subseting')
subsets = []
for p in res.peaks:
    lim = np.vstack((p.center-p.sigma*sigma_multi,
                    p.center+p.sigma*sigma_multi)).T
    subsets.append(subset(data, lim))

print('membership')
# res2 = membership(subsets[0], res.peaks[0].star_count)
# memership_result = membership(subset(data, arbitrary_limits), 50)

for i in range(len(res.peaks)):
    m = membership4(
        data=subsets[i][:,0:3],
        min_cluster_size=res.peaks[i].count,
        n_iters=100,
    ).probabilities
    p1 = subsets[i][:,6]
    p2 = subsets[i][:,7]
    p3 = subsets[i][:,8]
    pf = subsets[i][:,9]
    real_labels = np.zeros_like(pf)
    real_labels[pf<.5] = 1
    calculated_labels = np.zeros_like(pf)
    calculated_labels[m[:,0] < .5] = 1
    acc = accuracy_score(real_labels, calculated_labels)
    conf = confusion_matrix(real_labels, calculated_labels)
    minfo = normalized_mutual_info_score(real_labels, calculated_labels)
    error = np.sum((pf-m[:,0])**2)
    pair(subsets[i][:,0:3], m[:,1], np.argmax(m, axis=1)-1)
    print('coso')
