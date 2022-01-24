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
from hdbscan import HDBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from opencluster.hkde import HKDE, PluginBandwidth
from sklearn.metrics import pairwise_distances
from abc import abstractmethod
from sklearn.base import ClassifierMixin, ClusterMixin

@attrs(auto_attribs=True)
class Membership:
    p: np.ndarray
    clustering_result: HDBSCAN

@attrs(auto_attribs=True)
class DensityBasedMembershipEstimator(ClassifierMixin):
    
    min_cluster_size: int
    n_iters: int=100
    iteration_atol: float = .01
    metric: str = 'mahalanobis'
    clustering_scaler: TransformerMixin = RobustScaler()
    clusterer: ClusterMixin = None
    pdf_estimator: HKDE = HKDE(bw=PluginBandwidth(binned=True, pilot='unconstr'))
    allow_single_cluster: bool = True
    min_samples: int = None

    def calculate_p_with_labeled_data(self, data: np.ndarray, labels: np.ndarray):
        obs, dims = data.shape
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            p = np.atleast_2d(np.ones(obs)).T
        else:
            d = np.zeros((obs, len(unique_labels)))
            for label in unique_labels:
                # create kde per class but not recalculate every kernel and cov matrix
                class_estimator = HKDE(
                    bw=self.pdf_estimator.bw,
                    kernels=self.pdf_estimator.kernels[labels==label],
                    covariances=self.pdf_estimator.covariances[labels==label],
                    d=self.pdf_estimator.d, n=labels[labels==label].shape[0],
                )
                d[:,label+1] = class_estimator.pdf(data, leave1out=True)                
            p = d/np.atleast_2d(d.sum(axis=1)).repeat(len(unique_labels), axis=0).T
        return p
    
    def calculate_p_with_weighted_data(self, data: np.ndarray, weigths: np.ndarray):
        obs, dims = data.shape
        _, n_classes = weigths.shape
        if n_classes == 1:
            p = np.atleast_2d(np.ones(obs)).T
        else:
            d = np.zeros((obs, n_classes))
            for i in range(n_classes):
                # create kde per class but not recalculate every kernel and cov matrix
                class_estimator = HKDE(
                    bw=self.pdf_estimator.bw,
                    kernels=self.pdf_estimator.kernels,
                    covariances=self.pdf_estimator.covariances,
                    d=self.pdf_estimator.d,
                )
                class_estimator.set_weigths(weights)
                d[:,label+1] = class_estimator.pdf(data, leave1out=True)
            p = d/np.atleast_2d(d.sum(axis=1)).repeat(len(unique_labels), axis=0).T
        return p

    def fit_predict(self, data: np.ndarray, err: np.ndarray=None, corr:Union[np.ndarray, str]='auto'):
        
        obs, dims = data.shape
        if self.clustering_scaler is not None:
            clustering_data = self.clustering_scaler.fit(data).transform(data)
        else:
            clustering_data = data

        if self.clusterer is None:
            distance_matrix = pairwise_distances(clustering_data, metric=self.metric)
            self.clusterer = HDBSCAN(
                min_samples=self.min_samples,
                min_cluster_size=self.min_cluster_size,
                allow_single_cluster=self.allow_single_cluster,
                metric='precomputed',
            ).fit(clustering_data)
        
        clustering_result = self.clusterer.fit(clustering_data)
        self.pdf_estimator = self.pdf_estimator.fit(data=data, err=err, corr=corr)
        first_estimation = self.calculate_p_with_labeled_data(data=data, labels=clustering_result.labels_)
        
        if n_iters < 2:
            return first_estimation

        previous_estimation = first_estimation
        for i in range(n_iters):
            current_estimation = self.calculate_p_with_weighted_data(data=data, weigths=previous_estimation)
            if np.allclose(current_estimation, previous_estimation, atol=self.iteration_atol):
                break
            # is copy actually needed?
            previous_estimation = np.copy(current_estimation)

        return Membership(clustering_result=clustering_result, p=current_estimation)

def test_membership():
    df = one_cluster_sample_small()
    data = df.to_numpy()[:,0:3]
    calculated_p = DensityBasedMembershipEstimator(min_cluster_size=200, n_iters=1).fit_predict(data)
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

test_membership()