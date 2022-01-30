import os
import sys


import copy
import math
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Type, Union

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.coordinates import Distance, SkyCoord
from astropy.stats import biweight_location, biweight_scale, mad_std
from astropy.stats.sigma_clipping import sigma_clipped_stats
from attr import attrib, attrs, validators
from bayes_opt import BayesianOptimization
from hdbscan import HDBSCAN, all_points_membership_vectors
from hdbscan.validity import validity_index
from KDEpy import FFTKDE
from scipy import ndimage, stats
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from skimage.feature import peak_local_max
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import RobustScaler
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.robust.scale import huber, hubers_scale
from typing_extensions import TypedDict
sys.path.append(os.path.join(os.path.dirname("opencluster"), "."))
from opencluster.detection2 import CountPeakDetector
from opencluster.hkde import HKDE
from opencluster.masker import RangeMasker
from opencluster.membership2 import DensityBasedMembershipEstimator, pair
from opencluster.synthetic import three_clusters_sample
from opencluster.utils import Colnames2


class Pipeline:
    @abstractmethod
    def process(self, df: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        pass


@attrs(auto_attribs=True)
class PMPlxPipeline(Pipeline):
    def process(self, df, cluster_count=3):
        # only usable columns are going to be used
        # other data is going to be ignored

        obs, dims = df.shape

        df["idx"] = np.arange(obs)

        colnames = Colnames2(df.columns)
        detection_cols = ["pmra", "pmdec", "log10_parallax"]

        bin_shape = [0.5, 0.5, 0.05]
        detection_data = df[detection_cols].to_numpy()

        detector = CountPeakDetector(
            bin_shape=bin_shape,
            min_sigma_dif=None,
            max_num_peaks=cluster_count
        )
        res = detector.detect(data=detection_data)

        sigma_multiplier = 1.5

        membership_cols = ["pmra", "pmdec", "parallax"]
        n_vars = len(membership_cols)
        err_cols, missing_err = colnames.get_error_names(membership_cols)
        corr_cols, missing_corr = colnames.get_corr_names(membership_cols)

        if not missing_err:
            membership_cols += err_cols
            n_errs = len(err_cols)
        else:
            err = None
        if not missing_corr:
            membership_cols += corr_cols
            n_corrs = len(corr_cols)
        else:
            corr = "auto"

        membership_cols += ["idx"]

        membership_data = df[membership_cols].to_numpy()

        global_p = []
        for i, peak in enumerate(res.peaks):
            limits = np.vstack(
                (
                    peak.center - peak.sigma * sigma_multiplier,
                    peak.center + peak.sigma * sigma_multiplier,
                )
            ).T

            detection_mask = RangeMasker(limits).mask(detection_data)
            detection_subset = detection_data[detection_mask]

            # remove next 2 lines if want to use center recalculation
            membership_mask = detection_mask
            membership_subset = membership_data[membership_mask]

            # recalculate center with kde
            # scaler = RobustScaler().fit(np.vstack((detection_subset, limits.T)))
            # scaled_data = scaler.transform(detection_subset).T
            # scaled_limits = scaler.transform(limits.T).T
            # kde = gaussian_kde(scaled_data)
            # could also use mean shift
            # optimizer = BayesianOptimization(
            #    lambda x,y,z: kde.pdf((x, y, z))[0],
            #    { 'x': scaled_limits[0], 'y': scaled_limits[1], 'z': scaled_limits[2] },
            #    verbose=0,
            # )
            # optimizer.maximize()
            # max_params = optimizer.max.get('params')
            # center = scaler.inverse_transform(
            #    np.array(
            #        (max_params.get('x'), max_params.get('y'), max_params.get('z'))
            #    ).reshape(1,-1)
            # ).ravel()
            # inside_pm_limits = CenterMasker((center[0], center[1]), bin_shape[0]).mask(detection_data)
            # inside_log10_plx_limits = RangeMasker(
            #    [[center[2]-bin_shape[2], center[2]+bin_shape[2]]]
            # ).mask(np.atleast_2d(detection_data[:,2]).T)

            # membership_mask = inside_pm_limits & inside_log10_plx_limits

            # membership_subset = membership_data[membership_mask]

            data = membership_subset[:, :n_vars]
            if not missing_err:
                err = subset[:, n_vars : n_vars + n_errs]
            if not missing_corr:
                corr = subset[:, n_vars + n_errs : n_vars + n_errs + n_corrs]

            mestimator = DensityBasedMembershipEstimator(
                min_cluster_size=int(peak.count),
                iter_pdf_update=True,
                n_iters=5,
            )
            mres = mestimator.fit_predict(data)
            p = mres.p
            region_obs, n_classes = p.shape
            n_clusters = n_classes - 1

            # add each found cluster probs
            for n_c in range(n_clusters):
                cluster_p = np.zeros(obs)
                cluster_p[membership_mask] = p[:, n_c + 1]
                global_p.append(cluster_p)

        # add row for field prob
        global_p = np.array(global_p).T
        _, total_clusters = global_p.shape
        result = np.empty((obs, total_clusters + 1))
        result[:, 1:] = global_p
        result[:, 0] = 1 - global_p.sum(axis=1)

        return result


def test_PMPlxPipeline():
    df = three_clusters_sample(cluster_size=50)

    p = PMPlxPipeline().process(df)
    _, n_clus = p.shape

    data_to_plot = df[['pmra', 'pmdec', 'log10_parallax']].to_numpy()

    for n in range(n_clus):
        pair(data_to_plot, p[:,n], (p[:,n]>.5).astype(np.int))
        """ sns.scatterplot(df.ra, df.dec, hue=p[:, n], hue_norm=(0, 1)).set(
            title=f"p(x∈C{n}) ra-dec"
        )
        plt.figure()
        sns.scatterplot(df.pmra, df.pmdec, hue=p[:, n], hue_norm=(0, 1)).set(
            title=f"p(x∈C{n}) pm"
        )
        plt.figure()
        sns.scatterplot(
            df.pmra, df.parallax, hue=p[:, n], hue_norm=(0, 1)
        ).set(title=f"p(x∈C{n}) pmra-plx") """

    plt.show()
    print("coso")


test_PMPlxPipeline()
