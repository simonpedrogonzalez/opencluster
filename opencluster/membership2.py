import os
import sys
from copy import deepcopy
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from attr import attrs
from hdbscan import HDBSCAN
from sklearn.base import ClassifierMixin, ClusterMixin, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    normalized_mutual_info_score,
    pairwise_distances,
)
from sklearn.preprocessing import RobustScaler

sys.path.append(os.path.join(os.path.dirname("opencluster"), "."))

from opencluster.hkde import HKDE, PluginSelector
from opencluster.synthetic import one_cluster_sample_small
from opencluster.utils import combinations
from opencluster.fetcher import load_file
from opencluster.masker import RangeMasker


@attrs(auto_attribs=True)
class Membership:
    p: np.ndarray
    labels: np.ndarray
    clustering_result: HDBSCAN



@attrs(auto_attribs=True)
class DensityBasedMembershipEstimator(ClassifierMixin):

    min_cluster_size: int
    n_iters: int = 3
    iteration_atol: float = 0.01
    metric: str = "mahalanobis"
    clustering_scaler: TransformerMixin = RobustScaler()
    clusterer: ClusterMixin = None
    pdf_estimator: HKDE = HKDE()
    allow_single_cluster: bool = True
    min_samples: int = None
    iter_pdf_update: bool = False
    
    # internal attrs
    estimators: list = []

    n_classes: int = None
    n_obs: float = None

    labels: np.ndarray = None
    priors: np.ndarray = None
    class_counts: np.ndarray = None

    posteriors: np.ndarray = None
    data: np.ndarray = None

    """ def calculate_likelihood(
        self,
        data: np.ndarray,
        weights: np.ndarray,
    ):

        densities = np.zeros((self.n_obs, self.n_classes))

        if not len(self.estimators) or self.iter_pdf_update:
            # create pdf estimators for the first time or uptade them
            # (change kernels, covariances and weights)
            estimators = []
            for i in range(self.n_classes):
                class_estimator = deepcopy(self.pdf_estimator).fit(
                    data=data, weights=weights[:, i]
                )
                estimators.append(class_estimator)
                densities[:, i] = class_estimator.pdf(data, leave1out=True)
            self.estimators = estimators
        else:
            # use same estimator (same kernel and covariances) but change weights
            for i in range(self.n_classes):
                densities[:, i] = (
                    self.estimators[i]
                    .set_weights(weights[:, i])
                    .pdf(data, leave1out=True)
                )

        total_density = (densities * self.class_priors).sum(axis=1, keepdims=True).repeat(self.n_classes, axis=1)
        likelihood = densities / total_density
        return likelihood """

    """     def calculate_posterior(self, data: np.ndarray, weights: np.ndarray):
        likelihoods = self.calculate_likelihood(data, weights)
        weighted_likelihoods = likelihoods * self.class_priors
        obs_priors = weighted_likelihoods.sum(axis=1)
        posterior = weighted_likelihoods / np.repeat(
            obs_priors[:, np.newaxis], self.n_classes, axis=1
        )
        return posterior, likelihoods """

    def calculate_posterior2(self, data: np.ndarray, err, corr, weights: np.ndarray):
        densities = np.zeros((self.n_obs, self.n_classes))

        if not len(self.estimators) or self.iter_pdf_update:
            # create pdf estimators for the first time or uptade them
            # (change kernels, covariances and weights)
            estimators = []
            for i in range(self.n_classes):
                class_estimator = deepcopy(self.pdf_estimator).fit(
                    data=data, err=err, corr=corr, weights=weights[:, i],
                )
                estimators.append(class_estimator)
                densities[:, i] = class_estimator.pdf(data, leave1out=True)
            self.estimators = estimators
        else:
            # use same estimator (same kernel and covariances) but change weights
            for i in range(self.n_classes):
                densities[:, i] = (
                    self.estimators[i]
                    .set_weights(weights[:, i])
                    .pdf(data, leave1out=True)
                )

        total_density = (densities * self.class_counts).sum(axis=1, keepdims=True).repeat(self.n_classes, axis=1)
        posteriors = densities*self.class_counts / total_density
        return posteriors

    def get_labels(self, posteriors):
        labels = np.argmax(posteriors, axis=1)-1
        return labels

    def fit_predict(
        self,
        data: np.ndarray,
        err: np.ndarray = None,
        corr: Union[np.ndarray, str] = "zero",
    ):

        self.n_obs, dims = data.shape
        self.data = data

        if self.clustering_scaler is not None:
            clustering_data = self.clustering_scaler.fit(data).transform(data)
        else:
            clustering_data = data

        distance_matrix = pairwise_distances(
            clustering_data, metric=self.metric
        )

        if self.clusterer is None:
            if self.min_samples is None:
                self.min_samples = self.min_cluster_size
            self.clusterer = HDBSCAN(
                min_samples=self.min_samples,
                min_cluster_size=self.min_cluster_size,
                allow_single_cluster=self.allow_single_cluster,
                metric="precomputed",
            )

        clustering_result = self.clusterer.fit(distance_matrix)
        labels = clustering_result.labels_

        self.labels, self.class_counts = np.unique(
            labels, return_counts=True
        )
        self.n_classes = len(self.labels)
        self.priors = self.class_counts / self.class_counts.sum()

        if self.n_classes == 1:
            self.posteriors = np.atleast_2d(np.ones(self.n_obs)).T
            self.labels = self.get_labels(self.posteriors)
            return Membership(
                p=self.posteriors,
                clustering_result=clustering_result,
                labels=self.labels
            )

        weights = (self.labels == labels[:, None]).astype(np.int)

        posteriors = self.calculate_posterior2(data, err=err, corr=corr, weights=weights)
        self.posteriors = posteriors
        self.labels = self.get_labels(posteriors)

        if self.n_iters < 2:
            self.posteriors = posteriors
            self.labels = self.get_labels(self.posteriors)
            return Membership(
                clustering_result=clustering_result, p=posteriors, labels=self.labels
            )

        previous_posteriors = posteriors

        for i in range(self.n_iters):
            # self.membership_plot(0)
            # plt.show()
            # print([e.covariances[0] for e in self.estimators])
            # update priors
            print(self.priors)
            print(self.posteriors.sum(axis=0))
            weights = previous_posteriors
            self.class_counts = posteriors.sum(axis=0)
            self.priors = self.class_counts / self.class_counts.sum()
            posteriors = self.calculate_posterior2(
                data=data, weights=weights, err=err, corr=corr,
            )
            self.posteriors = posteriors

            if np.allclose(
                posteriors,
                previous_posteriors,
                atol=self.iteration_atol,
            ):
                break
            # is copy actually needed?
            previous_posteriors = np.copy(posteriors)

        self.posteriors = posteriors
        self.labels = self.get_labels(self.posteriors)
        return Membership(clustering_result=clustering_result, p=posteriors, labels=self.labels)

    def membership_plot(self, label=-1, **kwargs):
        return membership_plot(self.data, self.posteriors[:,label+1], self.labels, **kwargs)


def membership_plot(
    data:Union[np.ndarray, pd.DataFrame],
    posteriors,
    labels,
    colnames:list=None,
    palette='flare',
    corner=False,
    marker='o',
    ):

    if isinstance(data, np.ndarray):
        obs, dims = data.shape
        data = pd.DataFrame(data)
        if colnames is not None:
            data.columns = colnames
        else:
            data.columns = [f"var {i+1}" for i in range(dims)]
    
    # norm = plt.Normalize(0, 1)
    cmap = ListedColormap(sns.color_palette(palette).as_hex())
    sm = plt.cm.ScalarMappable(cmap=sns.color_palette(palette, as_cmap=True))#, norm=norm)
    sm.set_array([])
    
    ax = sns.pairplot(
        data,
        plot_kws={"hue": posteriors, "hue_norm": (0, 1), "marker": marker},
        diag_kind="kde",
        diag_kws={"hue": labels},
        palette=palette,
        corner=corner,
    )
    plt.colorbar(sm, ax=ax.axes)
    return ax

def test_membership():
    np.random.seed(0)
    df = one_cluster_sample_small(cluster_size=50)
    data = df[["pmra", "pmdec"]].to_numpy()

    real_pmp = df["p_pm_cluster1"].to_numpy()
    real_pmlabels = np.zeros_like(real_pmp)
    real_pmlabels[real_pmp > 0.5] = 1

    estimator = DensityBasedMembershipEstimator(
        min_cluster_size=50,
        n_iters=30,
        pdf_estimator=HKDE(bw=PluginSelector(diag=True)),
        iter_pdf_update=False,
    )
    result = estimator.fit_predict(data)
    calculated_pmp = result.p[:, 1]
    calculated_labels = np.zeros_like(calculated_pmp)
    calculated_labels[calculated_pmp > 0.5] = 1

    acc = accuracy_score(real_pmlabels, calculated_labels)
    conf = confusion_matrix(real_pmlabels, calculated_labels)
    minfo = normalized_mutual_info_score(real_pmlabels, calculated_labels)

    print("minfo")
    print(minfo)
    print("acc")
    print(acc)
    print("conf")
    print(conf)
    print("end")


def test_membership_real():
    file = 'scripts/data/clusters/ngc2527.vot'
    data = load_file(file).to_pandas()
    data_to_filter = data[['pmra', 'pmdec', 'log10_parallax']].to_numpy()
    data_membership = data[['pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr']].to_numpy()

    limits = np.array([
        [-6.31562475, -4.81562475],
        [ 6.58107034,  8.08107034],
        [ 0.11824112,  0.26824112]])

    mask = RangeMasker(limits).mask(data_to_filter)
    data_membership = data_membership[mask]
    data=data_membership[:,:2]
    err=data_membership[:,2:4]
    corr = np.atleast_2d(data_membership[:,4]).T

    estimator = DensityBasedMembershipEstimator(
        min_cluster_size=213,
        n_iters=200,
        pdf_estimator=HKDE(bw=PluginSelector(diag=True)),
        iter_pdf_update=False,
        allow_single_cluster=True,
    )
    result = estimator.fit_predict(data=data, err=err, corr=corr)

    p = result.p

    sns.scatterplot(data[:,0], data[:,1], hue=p[:,1])
    print('coso')

#test_membership()
# test_membership_real()
