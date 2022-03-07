import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))
from sklearn.metrics import confusion_matrix, accuracy_score, normalized_mutual_info_score

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
from opencluster.hkde import HKDE, PluginSelector
from sklearn.metrics import pairwise_distances

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
        df, plot_kws={'hue': hue , 'hue_norm': (0,1)},
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
        corr_param = corr if not isinstance(corr, np.ndarray) else corr[cl_result.hdbscan.labels_==label]
        err = None if errors is None else errors[cl_result.hdbscan.labels_==label]
        mem[:,label+1] = HKDE().fit(
            data=data[cl_result.hdbscan.labels_ == label],
            err=err,
            corr=corr_param,
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
    corr: Union[np.ndarray, str]='auto',
    *args,
    **kwargs,
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
            err=None if err is None else err[labels==label],
            corr=corr if isinstance(corr, str) else corr[labels==label],
            *args,**kwargs,
        ).pdf(data)

    # safe divide??
    #p = np.e**(np.log(d) - np.log(np.atleast_2d(d.sum(axis=1)).repeat(len(unique_labels), axis=0).T))
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
    single_cluster: bool = True,
    err: np.ndarray = None,
    corr: Union[np.ndarray, str] = 'auto',
    scaler=RobustScaler(),
    n_iters: int = 100,
    min_iter_diff: float = .01,
    dist: Union[str, np.ndarray] = 'mahalanobis',
    *args,
    **kwargs,
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
        p = calculate_membership(data, previous_labels, err, corr, *args, **kwargs)
        maxs = np.max(p[:,1:], axis=1)
        labels = np.argmax(p[:,1:], axis=1)
        labels[maxs < .5] = -1
        if np.alltrue(np.equal(previous_labels, labels)):
            break
        previous_labels = np.copy(labels)

    return Membership(clustering=c, probabilities=p)

def membership5(
    data: np.ndarray,
    min_cluster_size: int, 
    single_cluster: bool = True,
    err: np.ndarray = None,
    corr: Union[np.ndarray, str] = 'auto',
    scaler=RobustScaler(),
    n_iters: int = 100,
    min_iter_diff: float = .01,
    dist: Union[str, np.ndarray] = 'mahalanobis',
    *args,
    **kwargs,
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

    hkde = HKDE(bw=PluginBandwidth(binned=True, pilot='unconstr')).fit(
        data=data,
        err=None if err is None else err,
        corr=corr,
    )

    for i in range(n_iters):

        # calculate membership
        unique_labels = np.unique(previous_labels)
        if len(unique_labels) == 1:
            p = np.atleast_2d(np.ones(obs)).T
        else:
            d = np.zeros((obs, len(unique_labels)))
            for label in unique_labels:
                # create kde per class but not recalculate every kernel and cov matrix
                class_hkde = HKDE()
                class_hkde.kernels = hkde.kernels[previous_labels==label]
                class_hkde.covariances = hkde.covariances[previous_labels==label]
                class_hkde.d = 3
                class_hkde.n = hkde.kernels.shape[0]
                d[:,label+1] = class_hkde.pdf(data, leave1out=True)
                
            p = d/np.atleast_2d(d.sum(axis=1)).repeat(len(unique_labels), axis=0).T
        
            # check if labels change between 2 iterations
            maxs = np.max(p[:,1:], axis=1)
            labels = np.argmax(p[:,1:], axis=1)
            labels[maxs < .5] = -1
            if np.alltrue(np.equal(previous_labels, labels)):
                break
            previous_labels = np.copy(labels)

    return Membership(clustering=c, probabilities=p)


def test_membership():
    df = one_cluster_sample_small()
    s = df.to_numpy()[:,0:3]
    calculated_p = membership5(s, 200, n_iters=1).probabilities[:,1]
    # compare
    real_p = df['p_cluster1'].to_numpy()
    real_labels = np.zeros_like(real_p)
    real_labels[real_p >.5] = 1
    calculated_labels = np.zeros_like(calculated_p)
    calculated_labels[calculated_p > .5] = 1
    acc = accuracy_score(real_labels, calculated_labels)
    conf = confusion_matrix(real_labels, calculated_labels)
    minfo = normalized_mutual_info_score(real_labels, calculated_labels)
    print('coso')

# test_membership()
