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
from scipy.stats import gaussian_kde
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
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from KDEpy import FFTKDE
from opencluster.kdeh import HKDE
from sklearn.metrics import pairwise_distances


""" class VariableBandiwthKDE:
    def fit(data):
        dims = data.shape[1]
        bandwidths = []
        for i in range(dim):
            estimate_bandwidth """

def best_kde(data, n_iters=20):
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

def pair(data, mem=None, labels=None):
    df = pd.DataFrame(data)
    if (data.shape[1] == 3):
        df.columns = ['pmra', 'pmdec', 'parallax']
    elif (data.shape[1] == 5):
        df.columns = ['pmra', 'pmdec', 'parallax', 'ra', 'dec']
    else:
        raise Exception('wrong col number')
    if mem is None and labels is None:
        return sns.pairplot(df)
    if mem is not None:
        hue = np.round(mem, 2)
    else:
        hue = labels
    return sns.pairplot(
        df, plot_kws={'hue': hue },
        diag_kind='kde', diag_kws={'hue':labels},
        corner=True,
        ).map_lower(sns.kdeplot, levels=4, color=".1")


@attrs(auto_attribs=True)
class ClusteringResult:
    hdbscan: HDBSCAN
    diptest_pval: float


@attrs(auto_attribs=True)
class Membership:
    probabilities: np.ndarray
    clustering: HDBSCAN

@attrs(auto_attribs=True)
class MembershipResult:
    probabilities: np.ndarray
    hopkins_metric: float
    clustering_result: ClusteringResult
    success: bool

def membership(
    data: np.ndarray,
    star_count: int, 
    errors: np.ndarray = None,
    corr: np.ndarray = None,
    hopkins_threshold:float=.6,
    scaler=RobustScaler(),
    ):

    dim = np.atleast_2d(data).shape[1]

    if scaler is not None:
        scaled = scaler.fit(data).transform(data)
    else:
        scaled = data
    
    print('hopkins')
    #hopkins_metric = hopkins(scaled)

    if hopkins_threshold is not None and hopkins_metric < hopkins_threshold:
        return MembershipResult(
            probabilities=None,
            hopkins_metric=hopkins_metric,
            clustering_result=None,
            success=False
        )
    
    print('clustering')
    cl_result = hdbscan(scaled, star_count)
    labels = np.unique(cl_result.hdbscan.labels_)
    mem = np.zeros((data.shape[0], labels.shape[0]))
    mem2 = np.zeros((data.shape[0], labels.shape[0]))
    
    for label in labels:
        population = scaled[cl_result.hdbscan.labels_==label]
        # TODO: add bw estimation
        print(f'kde for label {label}')
        # mem[:,label+1] = np.exp(KernelDensity().fit(population).score_samples(scaled))
        mem2[:,label+1] = gaussian_kde(population.T).pdf(scaled.T)
        mem[:,label+1] = HKDE().fit(
            data=data[cl_result.hdbscan.labels_ == label],
            errors=None if errors is None else errors[cl_result.hdbscan.labels_==label],
            corr=None if corr is None else corr[cl_result.hdbscan.labels_==label],
        ).pdf(data)

    mem = mem/np.atleast_2d(mem.sum(axis=1)).repeat(labels.shape[0], axis=0).T
    mem2 = mem2/np.atleast_2d(mem2.sum(axis=1)).repeat(labels.shape[0], axis=0).T

    pair(data, mem[:,1], cl_result.hdbscan.labels_)
    plt.show()

    return MembershipResult(
        probabilities=mem,
        hopkins_metric=hopkins_metric,
        clustering_result=cl_result,
        success=True
    )



def hdbscan(data, star_count, n_clusters: str='dip'):

    print('dip')
    #diptest_pval = dip(data)

    print('hdbscan')
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

def calculate_membership(
    data: np.ndarray,
    labels: np.ndarray,
    err: np.ndarray = None,
    corr: np.ndarray = None,
    ):
    obs, dims = data.shape

    unique_labels = np.unique(labels)

    if len(unique_labels) == 1:
        return np.atleast_2d(np.ones(obs)).T

    d = np.zeros((obs, len(unique_labels)))

    for label in unique_labels:
        population = data[labels==label]
        d[:,label+1] = HKDE().fit(
            data=population,
            errors=None if err is None else err[labels==label],
            corr=None if corr is None else corr[labels==label],
        ).pdf(data)
    p = d/np.atleast_2d(d.sum(axis=1)).repeat(len(unique_labels), axis=0).T
    return p

def membership2(
    data: np.ndarray,
    star_count: int, 
    errors: np.ndarray = None,
    corr: np.ndarray = None,
    hopkins_threshold:float=.6,
    scaler=RobustScaler(),
    e: Union[float, int] = 10,
    ):
    r = membership(data, star_count, errors, corr, hopkins_threshold, scaler)
    ns = r.probabilities.sum(axis=0)
    old_ns = np.zeros_like(ns) + np.array([data.shape[0]-star_count, star_count])
    while(np.any(old_ns - ns > 1)):
        r = membership(data, ns[1], errors, corr, hopkins_threshold, scaler)
        old_ns = ns
        ns = r.probabilities.sum(axis=0)
        print(old_ns)
        print(ns)
    return r

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

def cluster(
    data: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    single_cluster: bool=None,
    *args,
    **kwargs,
    ):
    
    if single_cluster != True:
        multiple_clusters_result = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                *args, **kwargs,
            ).fit(data)
        labels = np.unique(multiple_clusters_result.labels_)
        if single_cluster == False or len(labels) != 1:
            return multiple_clusters_result

    single_cluster_result = HDBSCAN(
        min_cluster_size=min_cluster_size,
        allow_single_cluster=True,
        min_samples=min_samples,
        *args, **kwargs,
    ).fit(data)

    return single_cluster_result

def cluster2(
    data: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    single_cluster: bool=None,
    *args,
    **kwargs,
    ):
    
    if single_cluster != True:
        multiple_clusters_result = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                *args, **kwargs,
            ).fit(data)
        labels = np.unique(multiple_clusters_result.labels_)
        if single_cluster == False or len(labels) != 1:
            return multiple_clusters_result

    single_cluster_result = HDBSCAN(
        min_cluster_size=min_cluster_size,
        allow_single_cluster=True,
        min_samples=min_samples,
        *args, **kwargs,
    ).fit(data)

    return single_cluster_result

# TODO: include distance metric as parameter and the precomputed option
def membership3(
    data: np.ndarray,
    min_cluster_size: int, 
    single_cluster: bool = None,
    err: np.ndarray = None,
    corr: np.ndarray = None,
    scaler=RobustScaler(),
    n_iters: int = 1,
    mix_err: float = .01,
    dist: Union[str, np.ndarray] = 'mahalanobis',
    ):

    obs, dims = data.shape
    if scaler is not None:
        scaled = scaler.fit(data).transform(data)
    else:
        scaled = data
    
    if isinstance(dist, str):
        dist_matrix = pairwise_distances(scaled, metric=dist)
    else:
        assert dist.shape == (obs, obs)
        dist_matrix = dist

    min_samples = min_cluster_size

    c = cluster(
        dist_matrix,
        int(min_cluster_size),
        int(min_samples),
        single_cluster,
        metric='precomputed')
    p = calculate_membership(data, c.labels_, err, corr)
    
    i = 1

    unique_labels = np.unique(c.labels_)

    previous_mix = np.zeros_like(unique_labels) + \
        np.array([obs-min_cluster_size, min_cluster_size]) / obs
    
    p_sum_per_label = p.sum(axis=0)
    current_mix = p_sum_per_label / obs
    min_cluster_size = p_sum_per_label[1:].min()
    
    while((i < n_iters) and (np.any(np.abs(previous_mix - current_mix) > mix_err))):
        c = cluster(
            dist_matrix,
            int(min_cluster_size),
            int(min_samples),
            single_cluster,
            metric='precomputed')
        p = calculate_membership(data, c.labels_, err, corr)
        i += 1
        previous_mix = np.copy(current_mix)
        p_sum_per_label = p.sum(axis=0)
        current_mix = p_sum_per_label / obs
        min_cluster_size = p_sum_per_label[1:].min()

    return Membership(clustering=c, probabilities=p)


def membership4(
    data: np.ndarray,
    min_cluster_size: int, 
    single_cluster: bool = None,
    err: np.ndarray = None,
    corr: np.ndarray = None,
    scaler=RobustScaler(),
    n_iters: int = 100,
    min_iter_diff: float = .01,
    dist: Union[str, np.ndarray] = 'mahalanobis',
    ):

    obs, dims = data.shape
    if scaler is not None:
        scaled = scaler.fit(data).transform(data)
    else:
        scaled = data
    
    if isinstance(dist, str):
        dist_matrix = pairwise_distances(scaled, metric=dist)
    else:
        assert dist.shape == (obs, obs)
        dist_matrix = dist

    min_samples = min_cluster_size

    c = cluster(
        dist_matrix,
        int(min_cluster_size),
        int(min_samples),
        single_cluster,
        metric='precomputed')
    
    previous_labels = c.labels_

    for i in range(n_iters):
        p = calculate_membership(data, previous_labels, err, corr)
        maxs = np.max(p[:,1:], axis=1)
        labels = np.argmax(p[:,1:], axis=1)
        labels[maxs < .5] = -1
        if np.alltrue(np.equal(previous_labels, labels)):
            break
        previous_labels = np.copy(labels)

    return Membership(clustering=c, probabilities=p)

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
