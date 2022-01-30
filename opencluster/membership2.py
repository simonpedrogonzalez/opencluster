import os
import sys
from copy import deepcopy
from typing import Union

import matplotlib.pyplot as plt
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

from opencluster.hkde import HKDE, PluginBandwidth
from opencluster.synthetic import one_cluster_sample_small
from opencluster.utils import combinations


@attrs(auto_attribs=True)
class Membership:
    p: np.ndarray
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
    class_estimators: list = []
    n_classes: int = None
    class_labels: np.ndarray = None
    class_priors: np.ndarray = None
    class_counts: np.ndarray = None
    n_obs: float = None
    iter_pdf_update: bool = False

    def calculate_likelihood(
        self,
        data: np.ndarray,
        weights: np.ndarray,
    ):

        densities = np.zeros((self.n_obs, self.n_classes))

        if not len(self.class_estimators) or self.iter_pdf_update:
            # create pdf estimators for the first time or uptade them
            # (change kernels, covariances and weights)
            estimators = []
            for i in range(self.n_classes):
                class_estimator = deepcopy(self.pdf_estimator).fit(
                    data=data, weights=weights[:, i]
                )
                estimators.append(class_estimator)
                densities[:, i] = class_estimator.pdf(data, leave1out=True)
            self.class_estimators = estimators
        else:
            # use same estimator (same kernel and covariances) but change weights
            for i in range(self.n_classes):
                densities[:, i] = (
                    self.class_estimators[i]
                    .set_weights(weights[:, i])
                    .pdf(data, leave1out=True)
                )

        total_density = densities.sum(axis=1, keepdims=True).repeat(
            self.n_classes, axis=1
        )
        likelihood = densities / total_density
        return likelihood

    def calculate_posterior(self, data: np.ndarray, weights: np.ndarray):
        likelihoods = self.calculate_likelihood(data, weights)
        weighted_likelihoods = (
            likelihoods
            * np.repeat(self.class_priors[:, np.newaxis], self.n_obs, axis=1).T
        )
        obs_priors = weighted_likelihoods.sum(axis=1)
        posterior = weighted_likelihoods / np.repeat(
            obs_priors[:, np.newaxis], self.n_classes, axis=1
        )
        return posterior, likelihoods

    def fit_predict(
        self,
        data: np.ndarray,
        err: np.ndarray = None,
        corr: Union[np.ndarray, str] = "auto",
    ):

        self.n_obs, dims = data.shape
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

        self.class_labels, self.class_counts = np.unique(
            labels, return_counts=True
        )
        self.n_classes = len(self.class_labels)
        self.class_priors = self.class_counts / self.class_counts.sum()

        if self.n_classes == 1:
            return Membership(
                p=np.atleast_2d(np.ones(self.n_obs)).T,
                clustering_result=clustering_result,
            )

        weights = (self.class_labels == labels[:, None]).astype(np.int)

        posteriors, likelihoods = self.calculate_posterior(data, weights)

        if self.n_iters < 2:
            return Membership(
                clustering_result=clustering_result, p=posteriors
            )

        previous_posteriors = posteriors

        for i in range(self.n_iters):
            print(self.class_priors)
            print(self.class_counts)
            print([e.covariances[0] for e in self.class_estimators])
            # update priors
            weights = previous_posteriors
            self.class_counts = likelihoods.sum(axis=0)
            self.class_priors = self.class_counts / self.class_counts.sum()
            posteriors, likelihoods = self.calculate_posterior(
                data=data, weights=weights
            )

            if np.allclose(
                posteriors,
                previous_posteriors,
                atol=self.iteration_atol,
            ):
                break
            # is copy actually needed?
            previous_posteriors = np.copy(posteriors)

        return Membership(clustering_result=clustering_result, p=posteriors)


# TODO: remove
def pair(data, mem=None, labels=None):
    df = pd.DataFrame(data)
    if data.shape[1] == 2:
        df.columns = ["pmra", "pmdec"]
    elif data.shape[1] == 3:
        df.columns = ["pmra", "pmdec", "parallax"]
    elif data.shape[1] == 5:
        df.columns = ["pmra", "pmdec", "parallax", "ra", "dec"]
    else:
        raise Exception("wrong col number")
    if mem is None and labels is None:
        return sns.pairplot(df)
    if mem is not None:
        hue = np.round(mem, 2)
    else:
        hue = labels
    return sns.pairplot(
        df,
        plot_kws={"hue": hue, "hue_norm": (0, 1)},
        diag_kind="kde",
        diag_kws={"hue": labels},
        corner=True,
    ).map_lower(sns.kdeplot, levels=4, color=".1")


# TODO: remove
def grid(data, resolution: list = None):
    obs, dims = data.shape
    dim_points = []
    if resolution is None:
        resolution = [50] * dims
    for i in range(dims):
        dim_points.append(
            np.linspace(data[:, i].min(), data[:, i].max(), num=resolution[i])
        )
    c = combinations(dim_points)
    return np.array(c)


def test_membership():
    np.random.seed(0)
    df = one_cluster_sample_small(cluster_size=50)
    data = df[["pmra", "pmdec"]].to_numpy()

    real_pmp = df["p_pm_cluster1"].to_numpy()
    real_pmlabels = np.zeros_like(real_pmp)
    real_pmlabels[real_pmp > 0.5] = 1

    estimator = DensityBasedMembershipEstimator(
        min_cluster_size=50,
        n_iters=2,
        pdf_estimator=HKDE(bw=PluginBandwidth(diag=True)),
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


# test_membership()
