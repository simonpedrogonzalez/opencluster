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
from hdbscan import HDBSCAN, all_points_membership_vectors
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE

def subset(data: np.ndarray, limits: list):
    for i in range(len(limits)):
        data = data[(data[:,i] > limits[i][0]) & (data[:,i] < limits[i][1])]
    return data

def kde(data):
    return data

def custom_method(data):
    return data

def cano_alfaro_method(data):
    return data

def fuzzy_dbscan(data, *args, **kwargs):
    data = RobustScaler().fit(data).transform(data)
    res = HDBSCAN(
        min_cluster_size=50,
        min_samples=2*data.shape[1],
        allow_single_cluster=True,
        metric='euclidean',
        prediction_data=True,
        ).fit(data)

    # ojo: los outlier scores y las membresías por cúmulo nunca suman 1! son un score nada más
    # habría que ajustarlos!
    cluster_memberships = all_points_membership_vectors(res)
    outlier_scores = res.outlier_scores_
    memberships = cluster_memberships
    # memberships = np.insert(cluster_memberships, 1, outlier_scores, 1)
    
    projection = TSNE().fit_transform(data)
    color_palette = sns.color_palette('Paired', 12)
    alphas = memberships
    max_membership_colors = np.array([color_palette[np.argmax(x)] for x in memberships])
    # max_membership_colors = np.insert(max_membership_colors, 1, alphas, 1)
    
    # sns.scatterplot(x=data[:,0], y=data[:,1], hue=res.labels_)
    sns.scatterplot(x=projection[:,0], y=projection[:,1], c=max_membership_colors)
    plt.show()
    print(res)
    
    # another way: calculate 1-1 distances of all sample.
    # order them from min to max
    # plot it, find the point of most curvature
    return res 