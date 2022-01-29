import os
import sys
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

from opencluster.hkde import HKDE
from opencluster.synthetic import (
    one_cluster_sample_small,
)
from opencluster.utils import combinations


@attrs(auto_attribs=True)
class Membership:
    p: np.ndarray
    clustering_result: HDBSCAN


@attrs(auto_attribs=True)
class DensityBasedMembershipEstimator(ClassifierMixin):

    min_cluster_size: int
    n_iters: int = 100
    iteration_atol: float = 0.01
    metric: str = "mahalanobis"
    clustering_scaler: TransformerMixin = RobustScaler()
    clusterer: ClusterMixin = None
    pdf_estimator: HKDE = HKDE()
    allow_single_cluster: bool = True
    min_samples: int = None
    class_estimators: list = None

    def calculate_p_with_labeled_data(
        self, data: np.ndarray, labels: np.ndarray
    ):
        obs, dims = data.shape
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
            p = np.atleast_2d(np.ones(obs)).T
        else:
            d = np.zeros((obs, len(unique_labels)))
            estimators = []
            for label in unique_labels:
                # create kde per class but not recalculate every kernel and cov matrix
                class_estimator = HKDE(bw=self.pdf_estimator.bw).fit(
                    data[labels == label]
                )
                estimators.append(class_estimator)
                d[:, label + 1] = class_estimator.pdf(data, leave1out=True)
            p = (
                d
                / np.atleast_2d(d.sum(axis=1))
                .repeat(len(unique_labels), axis=0)
                .T
            )
        return p

    def calculate_p_with_weighted_data(
        self, data: np.ndarray, weigths: np.ndarray
    ):
        obs, dims = data.shape
        _, n_classes = weigths.shape
        if n_classes == 1:
            p = np.atleast_2d(np.ones(obs)).T
        else:
            d = np.zeros((obs, n_classes))
            for i in range(n_classes):
                # create kde per class but not recalculate every kernel and cov matrix
                class_estimator = HKDE(bw=self.pdf_estimator.bw).fit(
                    data=data, weigths=weigths
                )
                d[:, i] = class_estimator.pdf(data, leave1out=True)
            p = d / np.atleast_2d(d.sum(axis=1)).repeat(n_classes, axis=0).T
        return p

    def fit_predict(
        self,
        data: np.ndarray,
        err: np.ndarray = None,
        corr: Union[np.ndarray, str] = "auto",
    ):

        obs, dims = data.shape
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

        first_estimation = self.calculate_p_with_labeled_data(
            data=data, labels=clustering_result.labels_
        )

        if self.n_iters < 2:
            return Membership(
                clustering_result=clustering_result, p=first_estimation
            )

        previous_estimation = first_estimation
        for i in range(self.n_iters):
            current_estimation = self.calculate_p_with_weighted_data(
                data=data, weigths=previous_estimation
            )
            if np.allclose(
                current_estimation,
                previous_estimation,
                atol=self.iteration_atol,
            ):
                break
            # is copy actually needed?
            previous_estimation = np.copy(current_estimation)

        return Membership(
            clustering_result=clustering_result, p=current_estimation
        )


# TODO: remove
def pair(data, mem=None, labels=None):
    df = pd.DataFrame(data)
    if data.shape[1] == 3:
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
    df = one_cluster_sample_small(cluster_size=50)
    real_pmp = df["p_pm_cluster1"].to_numpy()
    real_pmlabels = np.zeros_like(real_pmp)
    real_pmlabels[real_pmp > 0.5] = 1
    data = df[["pmra", "pmdec"]].to_numpy()
    calculated_pmp = (
        DensityBasedMembershipEstimator(min_cluster_size=50, n_iters=100)
        .fit_predict(data)
        .p[:, 1]
    )
    calculated_p2 = (
        DensityBasedMembershipEstimator(min_cluster_size=50, n_iters=1)
        .fit_predict(data)
        .p[:, 1]
    )
    # compare

    calculated_labels = np.zeros_like(calculated_pmp)
    calculated_labels[calculated_pmp > 0.5] = 1
    calculated_labels2 = np.zeros_like(calculated_p2)
    calculated_labels2[calculated_p2 > 0.5] = 1
    acc = accuracy_score(real_pmlabels, calculated_labels)
    conf = confusion_matrix(real_pmlabels, calculated_labels)
    minfo = normalized_mutual_info_score(real_pmlabels, calculated_labels)
    print("minfo")
    print(minfo)
    print("acc")
    print(acc)
    print("conf")
    print(conf)


# test_membership()
