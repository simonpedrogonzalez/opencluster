from opencluster.synthetic import case2_sample0c, case2_sample1c, case2_sample2c, BivariateUnifom
from scipy.stats import kstest, multivariate_normal
from opencluster.stat_tests import HopkinsTest, DipTest, RipleysKTest
import pandas as pd
import math
import numpy as np
import pytest
import seaborn as sns
from matplotlib import pyplot as plt

@pytest.fixture
def uniform_sample():
    return BivariateUnifom(locs=(0,0), scales=(2.5, 2.5)).rvs(1000)

@pytest.fixture
def cluster_structure_sample():
    sample = BivariateUnifom(locs=(0,0), scales=(2.5, 2.5)).rvs(600)
    sample2 = multivariate_normal(mean=(.5, .5), cov=1./34).rvs(400)
    return np.concatenate((sample, sample2))


def test_hopkins_uniform(uniform_sample):
    assert not HopkinsTest(metric='mahalanobis', n_iters=100).test(data=uniform_sample).passed

def test_hopkins_cluster_structure(cluster_structure_sample):
    assert HopkinsTest(metric='mahalanobis', n_iters=100).test(data=cluster_structure_sample).passed

def test_hopkins_iris():
    """ Compare Hopkins implementation with R hopkins https://kwstat.github.io/hopkins/.
    hopkins(X, m=150, U=U)
    X is sklearn iris passed though rpy2
    m is number of samples, taken as all iris, so random sampling does not affect the result
    U is uniform 4-variate distribution created with numpy uniform from seed 0, with locs and scales given by sklearn iris
    value: 0.9978868058086875
    pvalue: 0.0
    """
    from sklearn.datasets import load_iris
    ht = HopkinsTest(n_iters=1,n_samples=150,metric='euclidean').test(load_iris().data)
    assert ht.passed
    assert np.isclose(0.9978868058086875, ht.value, atol=1e-3)
    assert np.isclose(0.0, ht.pvalue, atol=1e-3)

def test_dip_uniform(uniform_sample):
    assert not HopkinsTest().test(data=uniform_sample).passed

def test_dip_cluster_structure(cluster_structure_sample):
    assert DipTest().test(data=cluster_structure_sample).passed

def test_ripleysk_uniform(uniform_sample):
    assert not RipleysKTest().test(data=uniform_sample).passed

def test_ripleysk_cluster_structure(cluster_structure_sample):
    assert RipleysKTest().test(data=cluster_structure_sample).passed