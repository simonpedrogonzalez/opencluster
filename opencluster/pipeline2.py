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
from attrs import define, field, validators, Factory
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
from opencluster.detection2 import CountPeakDetector, DetectionResult
from opencluster.hkde import HKDE
from opencluster.masker import RangeMasker
from opencluster.membership2 import DensityBasedMembershipEstimator
from opencluster.synthetic import three_clusters_sample
from opencluster.utils import Colnames
from astropy.table.table import Table
from opencluster.stat_tests import StatTest, RipleysKTest, DipDistTest, HopkinsTest
from opencluster.membership3 import DBME
from opencluster.shdbscan import SHDBSCAN


@define
class PipelineResult:
    p: np.ndarray
    detection_result: DetectionResult
    membership_estimators: list

@define
class Pipeline:

    detector: CountPeakDetector = CountPeakDetector(bin_shape=(.5, .5, .05), min_dif=30)
    detection_kwargs: dict = {}
    detection_cols: List[str] = ['pmra', 'pmdec', 'log10_parallax']
    
    tests: List[StatTest] = field(default=[ RipleyKTest(mode='ks'), HopkinsTest()])
    test_cols: List[List[str]] = field(default=[['ra', 'dec'], ['ra', 'dec', 'pmra', 'pmdec']])
    
    
    membership_cols: List[str] = ['pmra', 'pmdec', 'parallax']
    clusterer: SHDBSCAN = SHDBSCAN(auto_allow_single_cluster=True)
    estimator: DBME = DBME()

    sample_sigma_factor: int = 5

    colnames: Colnames = None

    @test_cols.validator
    def test_cols_validator(self, attr, value):
        if len(value) != len(self.tests):
            raise ValueError("test_cols must have the same length as tests")

    def check_cols(self, cols):
        if len(self.colnames.get_data_names(cols)) != len(cols):
            raise ValueError("Columns must be a subset of {}".format(self.colnames.get_data_names()))
    
    def process(self, df):
        
        obs, dims = df.shape
        df["idx"] = np.arange(obs)
        df = df.dropna()

        self.colnames = Colnames(df.columns)

        if 'log10_parallax' in self.detection_cols and not len(colnames.get_data_names('log10_parallax')):
            df['log10_parallax'] = np.log10(df['parallax'].to_numpy())

        check_cols(self.detection_cols)

        detection_data = df[detection_cols].to_numpy()

        detection_result = detector.detect(detection_data, **self.detection_kwargs)
        
        n_vars = len(self.membership_cols)
        err_cols, missing_err = colnames.get_error_names(self.membership_cols)
        corr_cols, missing_corr = colnames.get_corr_names(self.membership_cols)

        if not missing_err:
            self.membership_cols += err_cols
            n_errs = len(err_cols)
        else:
            err = None
        if not missing_corr:
            self.membership_cols += corr_cols
            n_corrs = len(corr_cols)
        else:
            corr = None

        membership_data = df[self.membership_cols]

        global_proba = []
        clusterers = []
        estimators = []

        for i, peak in enumerate(detection_res.peaks):
            limits = np.vstack(
                (
                    peak.center - peak.sigma * self.sample_sigma_factor,
                    peak.center + peak.sigma * self.sample_sigma_factor,
                )
            ).T

            region_mask = RangeMasker(limits).mask(detection_data)

            region_df = membership_data[region_mask]

            self.test_results = []
            for i, stat_test in enumerate(self.tests):
                test_data = region_df[self.test_cols[i]]
                self.test_results.append(stat_test.test(test_data))

            should_cluster = np.all(np.asarray([tr.passed for tr in self.test_results]))

            if should_cluster:
                region_data = region_df.to_numpy()
                
                if self.clusterer.min_cluster_size and self.clusterer.clusterer is None:
                    self.clusterer.min_cluster_size = peak.count

                self.clusterer.fit(data=region_data)
                init_proba = self.clusterer.proba
                n_classes = self.clusterer.n_classes
                n_clusters = self.clusterer.n_clusters
                
                self.estimator.fit(data=region_data, init_proba=init_proba, err=err, corr=corr)
                proba = self.estimator.posteriors

            # add each found cluster probs
            for n_c in range(n_clusters):
                cluster_proba = np.zeros(obs)
                cluster_proba[region_mask] = proba[:, n_c + 1]
                global_proba.append(cluster_proba)

        # add row for field prob
        global_proba = np.array(global_proba).T
        _, total_clusters = global_proba.shape
        result = np.empty((obs, total_clusters + 1))
        result[:, 1:] = global_proba
        result[:, 0] = 1 - global_proba.sum(axis=1)

        return global_proba


def test_PMPlxPipeline():
    df = three_clusters_sample(cluster_size=50)

    p = PMPlxPipeline().process(df)
    _, n_clus = p.shape

    data_to_plot = df[['pmra', 'pmdec', 'log10_parallax']].to_numpy()

    for n in range(n_clus):
        sns.scatterplot(df.ra, df.dec, hue=p[:, n], hue_norm=(0, 1)).set(
            title=f"p(x∈C{n}) ra-dec"
        )
        plt.figure()
        sns.scatterplot(df.pmra, df.pmdec, hue=p[:, n], hue_norm=(0, 1)).set(
            title=f"p(x∈C{n}) pm"
        )
        plt.figure()
        sns.scatterplot(
            df.pmra, df.parallax, hue=p[:, n], hue_norm=(0, 1)
        ).set(title=f"p(x∈C{n}) pmra-plx")

    plt.show()
    print("coso")

def test_PMPlxPipeline_real_data():
    print('reading')
    ngc2527 = "scripts/data/clusters_phot/ngc2527.xml"
    s = "scripts/data/clusters_phot/ngc2527.xml"
    df = Table.read(ngc2527).to_pandas()
    print('processing')
    PMPlxPipeline().process(df)


#test_PMPlxPipeline()
test_PMPlxPipeline_real_data()

