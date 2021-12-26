
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '/home/simon/repos/opencluster'))
from opencluster.fetcher import load_remote, simbad_search, load_file, remote_info
from opencluster.synthetic import *
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
from KDEpy.bw_selection import improved_sheather_jones, silvermans_rule, scotts_rule
from KDEpy import NaiveKDE
from scipy.stats import gaussian_kde
from sklearn.preprocessing import RobustScaler
from statsmodels.nonparametric.kernel_density import KDEMultivariate

def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

def get_corr_coefs(data: np.ndarray, corr: np.ndarray=None):
    obs, dims = data.shape
    if corr is not None:
        # correlation is given
        if isinstance(corr, float) or isinstance(corr, int):
            # is a float or int
            corrs = np.zeros((obs, dims, dims))
            diag_idcs = tuple(map(tuple, np.vstack((
                    np.arange(obs).repeat(dims),
                    np.tile(np.array(np.diag_indices(dims)), (obs,))
                ))))
            corrs[diag_idcs] = 1
            return corrs
        elif isinstance(corr, np.ndarray):
            # is array
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

def get_sigmas(data: np.ndarray, bw: Union[np.ndarray, str, float], errors: np.ndarray = None):
    obs, dims = data.shape
    # silverman bw is supposed to be sigma^2 coeficient for the diagonal of the matrix
    # TODO: check if above statement is true
    if isinstance(bw, str):
        if bw == 'silverman':
            func = silvermans_rule
        elif bw == 'scott':
            func = scotts_rule
            func = lambda x: gaussian_kde(data).covariance_factor()
        elif bw == 'ISJ':
            func = improved_sheather_jones
        else:
            raise NotImplementedError(f'Bandwidth {bw} method not implemented.')
        bw = np.array([func(np.atleast_2d(data[:,i]).T) for i in range(dims)])
    elif isinstance(bw, float) or isinstance(bw, int):
        bw = np.ones(dims) * bw
    elif isinstance(bw, np.ndarray):
        if bw.shape != (dims,):
            raise ValueError('Wrong shape in bw array')
    if errors is None:
        return np.repeat(bw[:, np.newaxis], obs, 1).T
    elif data.shape != errors.shape:
        raise ValueError('Errors and data matrices shapes do not match')
    else:
        return np.sqrt(errors**2 + bw**2)

def get_cov_matrices(data, errors, corr, bw):
    # its correct
    sigmas = get_sigmas(data, bw, errors)
    corr_coefs = get_corr_coefs(data, corr)
    # get sigma matrices Sdxd per observations, where each element is 
    # sigmai*sigmaj for i,j = 1,...,dims
    sigmas_2 = np.apply_along_axis(lambda x: x*np.atleast_2d(x).T, -1, sigmas)
    return sigmas_2 * corr_coefs

@attrs(auto_attribs=True)
class HKDE:

    kernels: np.ndarray = None
    covariances: np.ndarray = None
    # determinants: np.ndarray = None
    n: int = None

    def fit(
        self,
        data: np.ndarray,
        errors: np.ndarray=None,
        corr: Union[np.ndarray, float, int]=None,
        bw: Union[np.ndarray, float, str]='ISJ',
        *args, **kwargs
        ):
        print('fitting')
        obs, dims = data.shape
        self.n = obs
        cov = get_cov_matrices(data, errors, corr, bw)
        self.covariances = cov
        # self.determinants = det
        """ det = np.apply_along_axis(
            lambda x: np.linalg.det(x.reshape((dims, dims))),
            1, cov.reshape((obs, dims*dims))
        ) """
        self.kernels = [
            multivariate_normal(
                data[i],
                cov[i],
                *args, **kwargs,
            ) for i in range(obs)
        ]
        
        return self

    def pdf(self, data):
        print('calculating')
        obs, dims = data.shape
        pdf = np.zeros(obs)
        cov = self.covariances
        #det = self.determinants

        # leave one out kde
        for i, k in enumerate(self.kernels):
            applied_k = k.pdf(data)# / det[i]
            applied_k[i] = 0
            pdf += applied_k
        return pdf/(self.n-1)

def test_corr():
    obs = 3
    dims = 3
    x = np.linspace(-1, 1, 10)
    y = x
    z = x
    x,y,z = np.mgrid[-1:1:50j, -1:1:50j, -1:1:50j]
    p = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    data = np.random.normal(size=(obs,dims))
    errors = halfnorm().rvs((obs, dims))

    pdf = HKDE().fit(data, errors).pdf(data)
    sns.scatterplot(data[:,0], data[:,1], hue=pdf)
    plt.show()
    sns.scatterplot(p[:,0], p[:,1], pdf)

    plt.show()
    print('coso')

def test_bandwidth():
    field = Field(
    pm=stats.multivariate_normal(mean=(0., 0.), cov=20),
    space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)),
    radius=10), star_count=int(1e3))
    clusters = [
        Cluster(
            space=stats.multivariate_normal(mean=polar_to_cartesian([120.7, -28.5, 5]), cov=.5),
            pm=stats.multivariate_normal(mean=(.5, 0), cov=1./35),
            star_count=200
        ),
    ]
    s = Synthetic(field=field, clusters=clusters).rvs().to_numpy()
    obs, dims = s.shape
    funcs = [scotts_rule, bw_scott, silvermans_rule, bw_silverman, improved_sheather_jones]
    data = []
    for f in funcs:
        for i in range(dims):
            bw = f(np.atleast_2d(s[:,i]).T)
            if isinstance(bw, np.ndarray):
                bw = bw[0]
            data.append({ 'var': i+1, 'func': f.__name__, 'bw': bw})
    df = pd.DataFrame(data)
    sns.lineplot(data=df, x='var', y='bw', hue='func')
    print(s)

def test_kdeh():
    field = Field(
    pm=stats.multivariate_normal(mean=(0., 0.), cov=20),
    space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)),
    radius=10), star_count=int(1e3))
    clusters = [
        Cluster(
            space=stats.multivariate_normal(mean=polar_to_cartesian([120.7, -28.5, 5]), cov=.5),
            pm=stats.multivariate_normal(mean=(.5, 0), cov=1./35),
            star_count=200
        ),
    ]
    df = Synthetic(field=field, clusters=clusters).rvs()
    s = df.to_numpy()[:,0:3]
    obs, dims = s.shape
    
    kdeh = HKDE().fit(s, bw='ISJ', corr=0)
    kdeh_pdf = kdeh.pdf(s)
    bws = [scotts_rule(np.atleast_2d(s[:,i]).T) for i in range(dims)]

    #ss = RobustScaler().fit(s).transform(s)
    # with scipy i cannot set different bws for different dims
    # nor different bw per dim
    # default is cov(data) * scott_factor**2
    scipykde = gaussian_kde(s.T, bw_method=1)
    scipy_pdf = scipykde.pdf(s.T)
    # with KDEpy i cannot set different bws per dims
    # I can set different bw per obs but no different bw per obs per dim
    # default is 1
    kdepykde = NaiveKDE().fit(s)
    kdepy_pdf = kdepykde.evaluate(s)
    # with statsmodels i can set different bws per dims, but not different bws per dim per obs
    stmodkde = KDEMultivariate(s, 'c'*dims, bw=[1]*dims)
    stmod_pdf = stmodkde.pdf(s)


    x = s[:,0]
    y = s[:,2]
    
    # assert np.allclose(kdeh_pdf, kdepy_pdf, atol=.001)
    
    plt.figure()
    sns.scatterplot(x,y,hue=scipy_pdf).set(title='scipy')
    plt.figure()
    sns.scatterplot(x,y,hue=kdeh_pdf).set(title='kdeh')
    plt.figure()
    sns.scatterplot(x,y,hue=kdepy_pdf).set(title='kdepy')
    plt.figure()
    sns.scatterplot(x,y,hue=stmod_pdf).set(title='statsmodels')
    """ plt.figure()
    sns.histplot(np.sqrt(kdeh_pdf**2-kdepy_pdf**2), bins=50).set(title='diff') """
    plt.figure()
    sns.histplot(scipy_pdf, bins=50).set(title='dens scipy')
    plt.figure()
    sns.histplot(kdepy_pdf, bins=50).set(title='dens kdepy')
    plt.figure()
    sns.histplot(kdeh_pdf, bins=50).set(title='dens kdeh')
    plt.figure()
    sns.histplot(stmod_pdf, bins=50).set(title='dens statsmodels')
    plt.show()
    print('coso')

# test_kdeh()