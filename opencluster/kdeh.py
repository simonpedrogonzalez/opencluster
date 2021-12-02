
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '/home/simon/repos/opencluster'))
from opencluster.fetcher import load_remote, simbad_search, load_file, remote_info
import scipy.integrate as integrate
import math
from scipy.stats import norm
import time
import multiprocessing
from dask.array import apply_along_axis, from_array
import math
from attr import attrs
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott
from statsmodels.stats.correlation_tools import corr_nearest
import seaborn as sns
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.stats import halfnorm
from typing import Optional, Tuple, List, Union, Callable, Type, Optional

def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

def get_corr_coefs(data: np.ndarray, corr: np.ndarray=None):
    obs, dims = data.shape
    if corr is not None:
        # correlation is given
        if corr.shape == (dims, dims):
            # correlation is given as global correlation matrix per dims
            return np.repeat(corr, repeats=obs, axis=1).reshape((dims, dims, obs)).T
        elif corr.shape == (obs, int(dims*(dims-1)/2)):
            # correlation is given per observation per obs, per dims
            # pairwise corr coef (no need for the 1s given by corr(samevar, samevar)) per observation
            # example: for 1 obs and 4 vars, lower triangle of corr matrix looks like
            # 12
            # 13 23 
            # 14 24 34
            # method should receive obs1 => [12, 13, 23, 14, 24, 34]
            n_corrs = corr.shape[1]
            corrs = np.zeros((obs, dims, dims))
            tril_idcs = tuple(map(tuple, np.vstack((
                np.arange(obs).repeat(n_corrs),
                np.tile(np.array(np.tril_indices(dims, k=-1)), (obs,))
            ))))
            corrs[tril_idcs] = corr.ravel()
            corrs = corrs + np.transpose(corrs, (0,2,1))
            diag_idcs = tuple(map(tuple, np.vstack((
                np.arange(obs).repeat(dims),
                np.tile(np.array(np.diag_indices(dims)), (obs,))
            ))))
            corrs[diag_idcs] = 1
            return corrs
        else:
            return ValueError('Wrong correlation parameter')
    else:
        # correlation is not given, calculate from data
        return np.repeat(
            corr_nearest(np.corrcoef(data, rowvar=False)),
            repeats=obs,
            axis=1).reshape((dims, dims, obs)).T

def get_sigmas(data: np.ndarray, errors: np.ndarray = None, bw: Union[np.ndarray, str, float]='silverman'):
    obs, dims = data.shape
    # silverman bw is supposed to be sigma^2 coeficient for the diagonal of the matrix
    # TODO: check if above statement is true
    if isinstance(bw, str):
        if bw == 'silverman':
            bw = bw_silverman(data)
        elif bw == 'scott':
            bw = bw_scott(data)
        else:
            raise NotImplementedError(f'Bandwidth {bw} method not implemented.')
    elif isinstance(bw, float):
        bw = np.ones(dims) * bw
    elif isinstance(bw, np.ndarray):
        if bw.shape != (dims,):
            raise ValueError('Wrong shape in bw array')
    if errors is None:
        return np.repeat(bw[:, np.newaxis], obs, 1).T
    elif data.shape != errors.shape:
        raise ValueError('Errors and data matrices shapes do not match')
    else:
        return np.sqrt(errors**2 + bw)

def get_cov_matrices(data, errors, corr, bw):
    sigmas = get_sigmas(data, errors)
    corr_coefs = get_corr_coefs(data, corr)
    sigmas_2 = np.apply_along_axis(lambda x: x*np.atleast_2d(x).T, -1, sigmas)
    return sigmas_2 * corr_coefs

@attrs(auto_attribs=True)
class HKDE:

    kernels: np.ndarray = None

    def fit(self, data: np.ndarray, errors: np.ndarray=None, corr: np.ndarray=None, bw: Union[np.ndarray, float, str]='silverman'):
        print('fitting')
        obs, dims = data.shape
        cov_matrices = get_cov_matrices(data, errors, corr, bw)
        self.kernels = [
            multivariate_normal(
                data[i],
                cov_matrices[i],
                allow_singular=True,
            ) for i in range(obs)
        ]        
        return self

    def pdf(self, data):
        print('calculating')
        obs, dims = data.shape
        pdf = np.zeros(obs)
        for i, k in enumerate(self.kernels):
            applied_k = k.pdf(data)
            applied_k[i] = 0
            pdf += applied_k
        return pdf/(obs-1)

""" obs = 3
dims = 3
x = np.linspace(-1, 1, 10)
y = x
z = x
x,y,z = np.mgrid[-1:1:50j, -1:1:50j, -1:1:50j]
p = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

data = np.random.normal(size=(obs,dims))
errors = halfnorm().rvs((obs, dims))

ex = HKDE().fit(data, errors).rvs(100)
pdf = HKDE().fit(data, errors).pdf(ex)
sns.scatterplot(ex[:,0], ex[:,1], hue=pdf)
plt.show()
sns.scatterplot(p[:,0], p[:,1], pdf)

plt.show()
print('coso')
 """