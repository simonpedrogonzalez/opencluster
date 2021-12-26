from sklearn.neighbors import BallTree
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
from sklearn.utils import resample
import seaborn as sns
from matplotlib import pyplot as plt
from unidip.dip import diptst
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Callable
from warnings import warn
from astropy.stats import RipleysKEstimator
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull


import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))

from opencluster.synthetic import *
from opencluster.utils import *
from abc import abstractmethod

@attrs(auto_attribs=True)
class TestResult:
    value: float
    passed: bool

class StatTest:
    @abstractmethod
    def test(self, data: np.ndarray, *args, **kwargs) -> TestResult:
        pass


@attrs(auto_attribs=True)
class HopkinsTest(StatTest):
    n_samples: int = None
    metric: str = 'mahalanobis'
    n_iters: int = 100
    # reduction: Callable = np.median
    # interpretation:
    # H0: data comes from uniform distribution
    # H1: data does not come from uniform distribution
    # if h = u/(u+w) ~ 1 => w = 0 luego hay estructura
    # if h = u/(u+w) ~ .5 => w ~ u luego no hay estructura
    # if h > .75 => reject H0, and in general  indicates a clustering tendency at the 90% confidence level.
    threshold: float = .75

    def test(self, data: np.ndarray, *args, **kwargs):
        """Assess the clusterability of a dataset. A score between 0 and 1, a score around 0.5 express
        no clusterability and a score tending to 1 express a high cluster tendency.

        Parameters
        ----------
        data : numpy array
            The input dataset
        n_samples : int
            The sampling size which is used to evaluate the number of DataFrame.

        Returns
        ---------------------
        score : float
            The hopkins score of the dataset (between 0 and 1)

        Examples
        --------
        >>> from sklearn import datasets
        >>> from pyclustertend import hopkins
        >>> X = datasets.load_iris().data
        >>> hopkins(X,150)
        0.16
        """
        assert len(data.shape) == 2
        
        obs, dims = data.shape

        if self.n_samples is None:
            n_samples = int(obs*.1)
        else:
            n_samples = min(obs, self.n_samples)

        results = []
        for i in range(self.n_iters):
            sample = resample(data, n_samples=n_samples, replace=False)
            if self.metric == 'mahalanobis':
                kwargs['V'] = np.cov(sample, rowvar=False)
            tree = BallTree(sample, leaf_size=2, metric=self.metric, *args, **kwargs)
            dist, _ = tree.query(sample, k=2)
            sample_nn_distance = dist[:, 1]

            max_data = data.max(axis=0)
            min_data = data.min(axis=0)
            uniform_sample = np.random.uniform(
                low=min_data, high=max_data,
                size=(n_samples, dims)
            )

            dist, _ = tree.query(uniform_sample, k=1)
            uniform_nn_distance = dist[:, 0]

            sample_sum = np.sum(sample_nn_distance**dims)
            uniform_sum = np.sum(uniform_nn_distance**dims)
            # sample_sum = self.reduction(sample_nn_distance)
            # uniform_sum = self.reduction(uniform_nn_distance)
            if sample_sum + uniform_sum == 0:
                raise Exception('The denominator of the hopkins statistics is null')
            results.append(uniform_sum / (uniform_sum + sample_sum))

        value = np.median(np.array(results))
        passed = value > self.threshold
        return TestResult(value=value, passed=passed)

@attrs(auto_attribs=True)
class DipTest(StatTest):
    n_samples: int =None
    metric: str = 'mahalanobis'
    threshold: float = .1

    def test(self, data: np.ndarray, *args, **kwargs):
        ''' dip test of unimodality over multidimensional data based on distance metric'''
        assert len(data.shape) == 2
        
        obs, dims = data.shape

        if self.n_samples is None:
            n_samples = min(obs, 100)
        else:
            n_samples = min(obs, self.n_samples)
    
        sample = resample(data, n_samples=n_samples, replace=False)
        dist = np.ravel(np.tril(pairwise_distances(sample, metric=self.metric)))
        dist = np.msort(dist[dist > 0])
        _, pval, _ = diptst(dist, *args, **kwargs)
        #sns.histplot(dist).set(title=str(pval))
        #plt.show()
        #print(pval)
        passed = pval < self.threshold
        return TestResult(value=pval, passed=passed)


@attrs(auto_attribs=True)
class RipleysKTest(StatTest):
    rk_estimator: RipleysKEstimator = None
    
    def test(self, data: np.ndarray, radii: np.ndarray=None, *args, **kwargs):

        obs, dims = data.shape
        if dims != 2:
            raise ValueError('Data must be bidimensional.')

        if radii is None:
            data =  MinMaxScaler().fit(data).transform(data)
            radii = np.linspace(.01, .25, 50)
        
        if self.rk_estimator is None:
            area = ConvexHull(data).volume
            self.rk_estimator = RipleysKEstimator(
                area=area,
                x_min=data[:,0].min(),
                x_max=data[:,0].max(),
                y_min=data[:,1].min(),
                y_max=data[:,1].max(),
            )
        else:
            area = self.rk_estimator.area
        
        l_function = self.rk_estimator.Lfunction(data, radii, *args, **kwargs) 

        if np.isnan(l_function).all():
            value = -np.inf
        else:
            value = np.nanmax(abs(l_function - radii))

        passed = True if value >= 1.68 * np.sqrt(area) / obs else False
        
        return TestResult(value=value, passed=passed)


""" 
@attrs(auto_attribs=True)
class RipleysKTest(StatTest):
    RipleysKEstimator(area=1, x_max=1, y_max=1, x_min=0, y_min=0) """

""" def cox_lewis(
    data: np.ndarray,
    metric='mahalanobis',
    reduction=np.median,
    *args,
    **kwargs
    ):
    dim = np.atleast_2d(data).shape[1]

    if metric == 'mahalanobis':
        kwargs['V'] = np.cov(sample, rowvar=False)
    tree = BallTree(sample, leaf_size=2, metric=metric, *args, **kwargs)
    
    max_data = data.max(axis=0)
    min_data = data.min(axis=0)
    uniform_sample = np.random.uniform(
        low=min_data, high=max_data,
        size=data.shape
    )

    dist, _ = tree.query(uniform_sample, k=2)
    distance_to_nn = dist[:, 0]
    distance_to_2nd_nn = dist[:, 1]

    b = 2*distance_to_nn/distance_to_2nd_nn
    b = fd[fd > 1]

    if fd.shape[0] == 0:
        raise Exception('No usable f(x) = 2*dist(yj, xj)/dist(xj, xi) found')

    g = 2**dim * a * np.sin(b/2)**dim - ap/dim ((dim - 1) * 2**dim * I )

    sample_sum = reduction(sample_nn_distance)
    uniform_sum = reduction(uniform_nn_distance)
    if sample_sum + uniform_sum == 0:
        raise Exception('The denominator of the hopkins statistics is null')
    results.append(uniform_sum / (uniform_sum + sample_sum))

    print(np.median(np.array(results)))
    return np.median(np.array(results)) """

def test_dip():
    ns = [100, 1000, int(1e4)]
    # case uniform
    uniforms = [
        UniformSphere(
            center=polar_to_cartesian((120.5, -27.5, 5)),
            radius=10
        ).rvs(n) for n in ns
    ]
    
    results_euclidean = np.array([ dip(u, metric='euclidean') for u in uniforms ])
    results_mahalanobis = np.array([ dip(u) for u in uniforms ])

    print(results_euclidean)
    print(results_mahalanobis)
    assert np.all(results_euclidean > .05)
    assert np.all(results_mahalanobis > .05)
    
    # case 1 k
    any_pm = stats.multivariate_normal(mean=(7.5, 7), cov=1./35)
    c_f_mixes = [.1, .5, .9]
    cov_diag = .5
    random_matrix = np.random.rand(3,3)
    cov_full = [np.dot(rm, rm.T) for rm in random_matrix]
    covs = [cov_diag, cov_full]
    metrics = ['euclidean', 'mahalanobis']
    parameters = combinations([ns, c_f_mixes, covs, metrics])
    oneclusters = [
        Synthetic(
            field=Field(
                space=UniformSphere(
                    center=polar_to_cartesian((120.5, -27.5, 5)),
                    radius=10
                ),
                pm=any_pm,
                star_count=int(p[0]*(1-p[1])),
            ),
            clusters = [
                Cluster(
                    space=stats.multivariate_normal(
                        mean=polar_to_cartesian([120.7, -28.5, 5]),
                        cov=p[2],
                    ),
                    pm=any_pm,
                    star_count=int(p[0]*p[1]),
                ),
            ], representation_type='cartesian',
        ).rvs()[['x', 'y', 'z']].to_numpy()
        for p in parameters
    ]
    results = [dip(oc, metric=p[3]) for oc, p in zip(oneclusters, parameters)]
    a = [(p[0], p[1], r) for p,r in zip(parameters, results)]
    print(results)
    print('coso')
    assert np.all(results < .2)
   
    # NOTE: "fails" when mix is too imbalanced, e.g. .1 to .9, or viceversa
    # Meaning: 
    # if dip > .1 
    # there is unimodal tendency, there are no clusters or there is only one cluster and no noise
    # if dip < .1
    # if there is multimodal tendency
    #   there are several clusters
    #   or one cluster + noise


    # case 2 k
    any_pm = stats.multivariate_normal(mean=(7.5, 7), cov=1./35)
    c_f_mixes = [.1, .5, .9]
    cov_diag = .5
    random_matrix = np.random.rand(3,3)
    cov_full = [np.dot(rm, rm.T) for rm in random_matrix]
    covs = [cov_diag, cov_full]
    metrics = ['euclidean', 'mahalanobis']
    parameters = combinations([ns, c_f_mixes, covs, metrics])

    twoclusters = [
        Synthetic(
            field=Field(
                space=UniformSphere(
                    center=polar_to_cartesian((120.5, -27.5, 5)),
                    radius=10
                ),
                pm=any_pm,
                star_count=int(p[0]*(1-p[1])),
            ),
            clusters = [
                Cluster(
                    space=stats.multivariate_normal(
                        mean=polar_to_cartesian([119.5, -28.5, 4.8]),
                        cov=p[2],
                    ),
                    pm=any_pm,
                    star_count=int(p[0]*p[1]),
                ),
                Cluster(
                    space=stats.multivariate_normal(
                        mean=polar_to_cartesian([121.5, -26.5, 5.2]),
                        cov=p[2],
                    ),
                    pm=any_pm,
                    star_count=int(p[0]*p[1]),
                ),
            ], representation_type='cartesian',
            
        ).rvs()[['x', 'y', 'z']].to_numpy()
        for p in parameters
    ]
    results = [dip(oc, metric=p[3]) for oc, p in zip(twoclusters, parameters)]
    a = [(p[0], p[1], r) for p,r in zip(parameters, results)]
    print(results)
    print('coso')


def test_hopkins():
    ns = [100, 1000, int(1e4)]
    # case uniform
    uniforms = [
        UniformSphere(
            center=polar_to_cartesian((120.5, -27.5, 5)),
            radius=10
        ).rvs(n) for n in ns
    ]
    metrics = ['euclidean', 'mahalanobis']
    reductions = [np.median, np.mean, np.sum, lambda x: np.sum(x**3)]
    cases = combinations([metrics, reductions, uniforms])
    results = [ (len(d), m, f, HopkinsTest(metric=m, reduction=f).test(d)) for m, f, d in cases]

    false_positives = [r for r in results if r[3].passed]
    df = pd.DataFrame([(r[3].value, r[2].__name__) for r in false_positives])
    df.columns = ['v', 'f']
    sns.kdeplot(df.v, hue=df.f)
    plt.show()

    print(results)

    # assert np.all(np.array([r.passed for _,_,r in results]) == False)

    # all good except for np.sum(x**3), which makes the test pass in some cases while it should not

    any_pm = stats.multivariate_normal(mean=(7.5, 7), cov=1./35)
    c_f_mixes = [.1, .5, .9]
    cov_diag = .5
    random_matrix = np.random.rand(3,3)
    cov_full = [np.dot(rm, rm.T) for rm in random_matrix]
    covs = [cov_diag, cov_full]
    cases = combinations([metrics, reductions, ns, c_f_mixes, covs])

    results = [(
        n, p, m, f, c,
        HopkinsTest(metric=m, reduction=f).test(
            Synthetic(field=Field(
                space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)),radius=10),
                pm=any_pm, star_count=int(n*(1-p)),
            ), clusters = [
                Cluster(space=stats.multivariate_normal(
                        mean=polar_to_cartesian([120.7, -28.5, 5]),
                        cov=c,),
                        pm=any_pm, star_count=int(n*p),),],
            representation_type='cartesian',
            ).rvs()[['x', 'y', 'z']].to_numpy()
            ),
            Synthetic(field=Field(
                space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)),radius=10),
                pm=any_pm, star_count=int(n*(1-p)),
            ), clusters = [
                Cluster(space=stats.multivariate_normal(
                        mean=polar_to_cartesian([120.7, -28.5, 5]),
                        cov=c,),
                        pm=any_pm, star_count=int(n*p),),],
            representation_type='cartesian',
            ).rvs()[['x', 'y', 'z']].to_numpy()
            ) for m, f, n, p, c in cases]
    
    # 1/3 of results are false negatives !!!
    # whats wrong here
    false_negatives = [r for r in results if r[5].passed == False]
    df = pd.DataFrame([(r[5].value, r[3].__name__) for r in false_positives])
    df.columns = ['v', 'f']
    sns.kdeplot(df.v, hue=df.f)
    plt.show()
    assert np.all(np.array([r.passed for _,_,r in results]) == True)

# test_hopkins()