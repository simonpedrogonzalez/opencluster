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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

""" class VariableBandiwthKDE:
    def fit(data):
        dims = data.shape[1]
        bandwidths = []
        for i in range(dim):
            estimate_bandwidth """

def estimate_bandwidth(data, n_iters=20):
    #seperate into validation and training
    #random_sample_size = data.shape[0]*.9
    #train, test = train_test_split(data, test_size=random_sample_size, train_size=random_sample_size)
    #1. now do the KDE with Gaussian kernel with cross validation
    grid = GridSearchCV(
        KernelDensity(),
        { 'bandwidth': np.logspace(-1, 1, 20)}) # 20-fold cross-validation
    #grid.fit(train_data)
    grid.fit(data)
    print("grid.best_params_: " + str(grid.best_params_))
    #get the best estimator
    #kde_gauss=grid.best_estimator_
    #what is the likelihood of validation data
    #kde_gauss.score(val_data)
    return grid

def pair(data, mem, labels):
    df = pd.DataFrame(data)
    df.columns = ['pmra', 'pmdec', 'parallax']
    return sns.pairplot(
        df, plot_kws={'hue':np.round(mem, 2)},
        diag_kind='kde', diag_kws={'hue':labels},
        corner=True,
        ).map_lower(sns.kdeplot, levels=4, color=".1")

def membership(data, star_count):
    scaled = RobustScaler().fit(data).transform(data)
    cl_results = fuzzy_dbscan(scaled, star_count)
    labels = np.unique(cl_results.labels_)
    mem = np.zeros((data.shape[0], labels.shape[0]))
    for label in labels:
        grid = estimate_bandwidth(scaled[cl_results.labels_ == label])
        mem[:,label+1] = np.exp(grid.best_estimator_.score_samples(scaled))

    mem = mem/np.atleast_2d(mem.sum(axis=1)).repeat(labels.shape[0], axis=0).T

    p1 = pair(data, mem[:,1], cl_results.labels_)
    
    p2 = pair(scaled, mem[:,1], cl_results.labels_)
    plt.show()
    
    return mem
    
def fuzzy_dbscan(data, star_count):
    res = HDBSCAN(
        min_cluster_size=int(star_count),
        # min_samples=data.shape[1]*2,
        allow_single_cluster=True,
        metric='mahalanobis',
        V=np.cov(data, rowvar=False),
        prediction_data=True,
        ).fit(data)

    """ df = pd.DataFrame(data)
    df.columns = ['pmra', 'pmdec', 'parallax']
    df['label'] = res.labels_
    sns.pairplot(data, hue='label')
    plt.show() """
    # ojo: los outlier scores y las membresías por cúmulo nunca suman 1! son un score nada más
    # habría que ajustarlos!
    #outlier_scores = res.outlier_scores_
    #memberships = cluster_memberships
    return res 
    # memberships = np.insert(cluster_memberships, 1, outlier_scores, 1)
    
"""  projection = TSNE().fit_transform(data)
    color_palette = sns.color_palette('Paired', 12)
    alphas = memberships
    max_membership_colors = np.array([color_palette[np.argmax(x)] for x in memberships]) """
    # max_membership_colors = np.insert(max_membership_colors, 1, alphas, 1)
    
    # sns.scatterplot(x=data[:,0], y=data[:,1], hue=res.labels_)
"""  sns.scatterplot(x=projection[:,0], y=projection[:,1], c=max_membership_colors)
    plt.show()
    print(res) """
    
    # another way: calculate 1-1 distances of all sample.
    # order them from min to max
    # plot it, find the point of most curvature