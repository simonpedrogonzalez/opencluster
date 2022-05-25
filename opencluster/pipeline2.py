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
from opencluster.membership3 import DBME
from opencluster.synthetic import three_clusters_sample, sample3c
from opencluster.utils import Colnames
from astropy.table.table import Table
from opencluster.stat_tests import StatTest, RipleysKTest, DipDistTest, HopkinsTest, TestResult
from opencluster.membership3 import DBME
from opencluster.shdbscan import SHDBSCAN


@define
class PipelineResult:
    p: np.ndarray
    detection_result: DetectionResult
    membership_estimators: list

@define
class Pipeline:

    detector: CountPeakDetector
    detection_cols: List[str]
    
    tests: List[StatTest]
    test_cols: List[List[str]] = field()
    
    
    membership_cols: List[str]
    clusterer: SHDBSCAN = SHDBSCAN(auto_allow_single_cluster=True, min_cluster_size=50)
    estimator: DBME = DBME()

    sample_sigma_factor: int = 5
    detection_kwargs: dict = {}

    colnames: Colnames = None
    
    test_results: List[List[TestResult]] = Factory(list)

    @test_cols.validator
    def test_cols_validator(self, attr, value):
        if len(value) != len(self.tests):
            raise ValueError("test_cols must have the same length as tests")

    def check_cols(self, cols):
        if len(self.colnames.get_data_names(cols)) != len(cols):
            raise ValueError("Columns must be a subset of {}".format(self.colnames.get_data_names()))
    
    def process(self, df):
        
        df = df.dropna()
        obs, dims = df.shape
        df["idx"] = np.arange(obs)

        self.colnames = Colnames(df.columns)

        if 'log10_parallax' in self.detection_cols and not len(self.colnames.get_data_names('log10_parallax')):
            df['log10_parallax'] = np.log10(df['parallax'].to_numpy())

        self.check_cols(self.detection_cols)

        detection_data = df[self.detection_cols].to_numpy()

        detection_result = self.detector.detect(detection_data, **self.detection_kwargs)
        
        n_vars = len(self.membership_cols)
        err_cols, missing_err = self.colnames.get_error_names(self.membership_cols)
        corr_cols, missing_corr = self.colnames.get_corr_names(self.membership_cols)

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

        test_cols = list(set([item for sublist in self.test_cols for item in sublist]))
        test_data = df[test_cols]

        membership_data = df[self.membership_cols]

        global_proba = []
        clusterers = []
        estimators = []

        if not len(detection_result.peaks):
            return np.ones(obs).reshape(-1, 1)

        for i, peak in enumerate(detection_result.peaks):
            limits = np.vstack(
                (
                    peak.center - peak.sigma * self.sample_sigma_factor,
                    peak.center + peak.sigma * self.sample_sigma_factor,
                )
            ).T

            region_mask = RangeMasker(limits).mask(detection_data)

            region_df = membership_data[region_mask]

            test_results = []
            region_test_df = test_data[region_mask]
            for i, stat_test in enumerate(self.tests):
                t_data = region_test_df[self.test_cols[i]].to_numpy()
                test_results.append(stat_test.test(t_data))

            self.test_results.append(test_results)

            should_cluster = np.all(np.asarray([tr.passed for tr in test_results]))

            print(f'should cluster { should_cluster }')

            if should_cluster:
                region_data = region_df.to_numpy()
                
                clusterer = copy.deepcopy(self.clusterer)
                if clusterer.min_cluster_size and clusterer.clusterer is None:
                    clusterer.min_cluster_size = int(peak.count)
                center = np.array([
                    sigma_clipped_stats(region_data[:,i], cenfunc="median", stdfunc="mad_std", maxiters=None, sigma=1)
                    for i in range(region_data.shape[1])
                ])[:,1]
                clusterer.fit(data=region_data, centers=[center])
                init_proba = clusterer.proba
                n_classes = clusterer.n_classes
                n_clusters = clusterer.n_classes - 1
                
                estimator = copy.deepcopy(self.estimator)
                estimator.fit(data=region_data, init_proba=init_proba, err=err, corr=corr)
                proba = estimator.posteriors

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

        return result


def test_PMPlxPipeline():
    df = sample3c()

    p = Pipeline(
        detector=CountPeakDetector(min_dif=50, bin_shape=[1, 1, .1]),
        detection_kwargs={ 'heatmaps': False },
        detection_cols=['pmra', 'pmdec', 'log10_parallax'],
        tests=[RipleysKTest(mode='chiu', pvalue_threshold=.05), RipleysKTest(mode='chiu', pvalue_threshold=.05)],
        test_cols=[['pmra', 'pmdec'], ['ra', 'dec']],
        membership_cols=['pmra', 'pmdec', 'parallax', 'ra', 'dec'],
        sample_sigma_factor=3,
        ).process(df)
    _, n_clus = p.shape

    for n in range(1):
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
    ngc2527 = "scripts/data/clusters/ngc2527.vot"
    s = "scripts/data/clusters_phot/ngc2527.xml"
    df = Table.read(ngc2527).to_pandas()

    p = Pipeline(
        detector=CountPeakDetector(min_dif=30, bin_shape=[.5, .5, .05]),
        detection_kwargs={ 'heatmaps': True },
        detection_cols=['pmra', 'pmdec', 'parallax'],
        sample_sigma_factor=5,
        ).process(df)
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

test_PMPlxPipeline()
#test_PMPlxPipeline_real_data()

