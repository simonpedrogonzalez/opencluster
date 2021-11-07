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


def hopkins(
    data: np.ndarray,
    n_samples=None,
    metric='mahalanobis',
    n_iters=1000,
    reduction=np.median,
    *args,
    **kwargs
    ):
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

    dim = np.atleast_2d(data).shape[1]

    if n_samples is None:
        n_samples = int(data.shape[0]*.1)
    elif n_samples > data.shape[0]:
        raise Exception('The number of sample of sample is bigger than the shape of D')

    results = []
    for i in range(n_iters):
        sample = resample(data, n_samples=n_samples, replace=False)
        if metric == 'mahalanobis':
            kwargs['V'] = np.cov(sample, rowvar=False)
        tree = BallTree(sample, leaf_size=2, metric=metric, *args, **kwargs)
        dist, _ = tree.query(sample, k=2)
        sample_nn_distance = dist[:, 1]

        max_data = data.max(axis=0)
        min_data = data.min(axis=0)
        uniform_sample = np.random.uniform(
            low=min_data, high=max_data,
            size=(n_samples, dim)
        )

        dist, _ = tree.query(uniform_sample, k=1)
        uniform_nn_distance = dist[:, 0]

        sample_sum = reduction(sample_nn_distance)
        uniform_sum = reduction(uniform_nn_distance)
        if sample_sum + uniform_sum == 0:
            raise Exception('The denominator of the hopkins statistics is null')
        results.append(uniform_sum / (uniform_sum + sample_sum))

    return np.median(np.array(results))


def dip(data, n_samples=None, metric='mahalanobis', *args, **kwargs):

    ''' dip test of unimodality over multidimensional data based on distance metric'''

    dim = np.atleast_2d(data).shape[1]

    if n_samples is None:
        n_samples = min(int(data.shape[0]*.1), 100)
    elif n_samples > data.shape[0]:
        raise Exception('The number of sample of sample is bigger than the shape of D')

    sample = resample(data, n_samples=n_samples, replace=False)
    
    dist = np.ravel(np.tril(pairwise_distances(sample, metric=metric)))

    dist = np.msort(dist[dist > 0])
    _, pval, _ = diptst(dist, *args, **kwargs)
    return pval



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