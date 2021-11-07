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
from hdbscan.validity import validity_index
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
# from pyclustertend import hopkins, ivat, vat
from clustering_tendency import hopkins, dip
from statsmodels.nonparametric.kernel_density import KDEMultivariate

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
    if (data.shape[1] == 3):
        df.columns = ['pmra', 'pmdec', 'log10_parallax']
    elif (data.shape[1] == 5):
        df.columns = ['pmra', 'pmdec', 'log10_parallax', 'ra', 'dec']
    else:
        raise Exception('wrong col number')
    return sns.pairplot(
        df, plot_kws={'hue':np.round(mem, 2)},
        diag_kind='kde', diag_kws={'hue':labels},
        corner=True,
        ).map_lower(sns.kdeplot, levels=4, color=".1")


@attrs(auto_attribs=True)
class ClusteringResult:
    hdbscan: HDBSCAN
    diptest_pval: float


@attrs(auto_attribs=True)
class MembershipResult:
    probabilities: np.ndarray
    hopkins_metric: float
    clustering_result: ClusteringResult
    success: bool

def membership(data: np.ndarray, star_count: int, hopkins_threshold:float=.75, scaler=RobustScaler()):

    dim = np.atleast_2d(data).shape[1]

    if scaler is not None:
        scaled = scaler.fit(data).transform(data)
    else:
        scaled = data
    
    hopkins_metric = hopkins(scaled)

    if hopkins_threshold is not None and hopkins_metric < hopkins_threshold:
        return MembershipResult(
            probabilities=None,
            hopkins_metric=hopkins_metric,
            clustering_result=None,
            success=False
        )
    
    cl_result = hdbscan(scaled, star_count)
    labels = np.unique(cl_result.hdbscan.labels_)
    mem = np.zeros((data.shape[0], labels.shape[0]))
    # use stats model implementation instead
    
    for label in labels:
        mem[:,label+1] = KDEMultivariate(
            scaled[cl_result.hdbscan.labels_],
            var_type='c'*dim, bw='cl_ml'
        ).pdf(scaled)

    mem = mem/np.atleast_2d(mem.sum(axis=1)).repeat(labels.shape[0], axis=0).T

    pair(scaled, mem, cl_result.hdbscan.labels_)
    plt.show()

    return MembershipResult(
        probabilities=mem,
        hopkins_metric=hopkins_metric,
        clustering_result=cl_result,
        success=True
    )



def hdbscan(data, star_count):

    diptest_pval = dip(data)

    if (diptest_pval <= .1):
        # pval < .05 => strong multimodality tendency
        # pval < .1 => marginally significant multimodality tendency
        multiple_clusters_result = HDBSCAN(
            min_cluster_size=int(star_count),
            metric='mahalanobis',
            V=np.cov(data, rowvar=False),
            prediction_data=True,
        ).fit(data)
        return ClusteringResult(multiple_clusters_result, diptest_pval)

    single_cluster_result = HDBSCAN(
        min_cluster_size=int(star_count),
        allow_single_cluster=True,
        metric='mahalanobis',
        V=np.cov(data, rowvar=False),
        prediction_data=True,
    ).fit(data)
    return ClusteringResult(single_cluster_result, diptest_pval)
    
    """ df = pd.DataFrame(data)
    df.columns = ['pmra', 'pmdec', 'parallax']
    df['label'] = res.labels_
    sns.pairplot(data, hue='label')
    plt.show() """
    # ojo: los outlier scores y las membresías por cúmulo nunca suman 1! son un score nada más
    # habría que ajustarlos!
    #outlier_scores = res.outlier_scores_
    #memberships = cluster_memberships
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