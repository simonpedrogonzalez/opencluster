
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
from attr import attrs, asdict
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
from abc import abstractmethod
import copy
from rpy2.robjects import r
from rutils import *

def pair(data, mem=None, labels=None):
    df = pd.DataFrame(data)
    if (data.shape[1] == 3):
        df.columns = ['pmra', 'pmdec', 'parallax']
    elif (data.shape[1] == 5):
        df.columns = ['pmra', 'pmdec', 'parallax', 'ra', 'dec']
    else:
        raise Exception('wrong col number')
    if mem is None and labels is None:
        return sns.pairplot(df)
    if mem is not None:
        hue = np.round(mem, 2)
    else:
        hue = labels
    return sns.pairplot(
        df, plot_kws={'hue': hue , 'hue_norm': (0,1)},
        diag_kind='kde', diag_kws={'hue':labels},
        corner=True,
        ).map_lower(sns.kdeplot, levels=4, color=".1")


def rkde(data):
    from rpy2.robjects import r
    from rpy2.robjects import packages as rpackages
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    #from rpy2.robjects.vectors import StrVector
    #rpackages_names = StrVector(('ks', ... ))
    try:
        ks = importr('ks')
    except error:
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages('ks')
        ks = importr('ks')
    obs, dims = data.shape
    r('library("ks")')
    r.assign('data', data)
    r('result <- kde(data, eval.points=data, H=Hpi(data, pilot="unconstr"))')
    H = r2np(r('result["H"]'), (dims, dims))
    pdf = r2np(r('result["estimate"]'), (obs,))
    return pdf, H


class Bandwidth:
    @abstractmethod
    def H(data, *args, **kwargs):
        pass


@attrs(auto_attribs=True)
class PluginBandwidth(Bandwidth):
    nstage: int = None
    pilot: str = 'unconstr'
    binned: bool = True
    diag: bool = False
    
    def H(self, data):
        
        #from rpy2.robjects import numpy2ri
        #numpy2ri.activate()
        #from rpy2.robjects.vectors import StrVector
        #rpackages_names = StrVector(('ks', ... ))
    
        _, dims = data.shape
        params = copy.deepcopy(self.__dict__)
        diag = params.pop('diag')

        # prepare packages
        rhardload(r, ['ks'])
        # delete all previous session variables
        rclean(r, 'var')
        _, rparams = pyargs2r(r, x=data, **params)
        result = r(f'ks::Hpi{".diag" if diag else ""}({rparams})')
        return r2np(result, (dims,dims))

@attrs(auto_attribs=True)
class HKDE:

    kernels: np.ndarray = None
    covariances: np.ndarray = None
    n: int = None
    d: int = None
    bw: Union[Bandwidth, int, float] = PluginBandwidth()
    weights: np.ndarray = None

    def get_sigmas(self, data: np.ndarray, err: np.ndarray):
        obs, dims = data.shape
        # inherent distribution variance values
        if isinstance(bw, Bandwidth):
            self.bw.diag=True
            variance = np.diag(self.bw.H(data))
        elif isinstance(self.bw, (int, float)):
            variance = np.ones(dims) * bw    
        elif isinstance(bw, (np.ndarray, list)):
            bw = np.array(bw)
            if bw.shape != (dims,):
                raise ValueError('Wrong shape in bw array')
            variance = bw
        if err is None:
            sigmas = np.sqrt(variance)
            return np.repeat(sigmas[:, np.newaxis], obs, 1).T
        elif data.shape != err.shape:
            raise ValueError('error matrix and data matrix shapes do not match')
        else:
            # TODO: check
            variance_with_err = np.sqrt(err**2 + variance**2)
            sigmas = np.sqrt(variance_with_errors)
            return sigmas

    def get_corr_coefs(self, data: np.ndarray, corr: Union[np.ndarray, str]):
        obs, dims = data.shape
        # correlation is given
        if isinstance(corr, str):
            if corr == 'zero':
                corrs = np.zeros((obs, dims, dims))
                diag_idcs = tuple(map(tuple, np.vstack((
                        np.arange(obs).repeat(dims),
                        np.tile(np.array(np.diag_indices(dims)), (obs,))
                    ))))
                corrs[diag_idcs] = 1
                return corrs
            elif corr=='auto': # correlation is not given, calculate from data
                return np.repeat(
                    corr_nearest(np.corrcoef(data, rowvar=False)),
                    repeats=obs,
                    axis=1).reshape((dims, dims, obs)).T

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
                raise ValueError('Wrong corr dimensions')
        raise ValueError('Wrong corr parameter')

    def get_cov_matrices(self, data, err, corr):
        obs, dims = data.shape
        # checked
        if err is None and corr == 'auto' and isinstance(self.bw, Bandwidth):
            # calculate full H from plugin method
            cov = self.bw.H(data)
            # repeat covariance for each obs
            return np.repeat(cov[:,np.newaxis], obs, 1).swapaxes(0,1)
        # TODO: check
        else:
            # get sigma value for each dimension
            sigmas = self.get_sigmas(data, err)
            # get correlation coefficients
            corr_coefs = self.get_corr_coefs(data, corr)
            # get covariance matrices Sdxd per observations, where each element is 
            # sigmai*sigmaj for i,j = 1,...,dims
            cross_sigmas = np.apply_along_axis(lambda x: x*np.atleast_2d(x).T, -1, sigmas)
            cov = cross_sigmas * corr_coefs
        return cov
            
    def fit(
        self,
        data: np.ndarray,
        err: np.ndarray=None,
        corr: Union[np.ndarray, str]='auto',
        weights: np.ndarray=None,
        *args, **kwargs
        ):
        obs, dims = data.shape
        if weights is not None:
            self.weights = weights
        if self.weights is not None:
            if len(self.weights.shape) != 1:
                raise ValueError('Weights must be 1d np ndarray.')
            if self.weights.shape[0] != obs:
                raise ValueError('Weights must have same n as data.')
            if np.any(self.weights > 1):
                raise ValueError('Weight values must belong to [0,1].')
            data = data[self.weights > 0]
            self.weights = self.weights[self.weights > 0]
            obs, dims = data.shape
        else:
            self.weights = np.ones(obs)

        if obs == 0:
            raise ValueError('Data matrix is empty or all points are weighted 0')
        self.n = obs
        self.d = dims
        print('getting cov')
        self.covariances = self.get_cov_matrices(data, err, corr)
        
        print('getting kern')
        self.kernels = np.array([
            multivariate_normal(
                data[i],
                self.covariances[i],
                *args, **kwargs,
            ) for i in range(obs)
        ])
        print('done fit')
        return self

    def pdf(self, eval_points: np.ndarray, leave1out=False):
        if self.kernels is None:
            raise Exception('Model not fitted. Try excecuting fit function first.')
        obs, dims = eval_points.shape
        if dims != self.d:
            raise ValueError('Eval points must have same dims as data.')
        print('eval')
        pdf = np.zeros(obs)
        if leave1out:
        # leave one out kde
            norm_factor = self.n-1
            for i, k in enumerate(self.kernels):
                applied_k = k.pdf(eval_points)# * self.weights[i]
                applied_k[i] = 0
                pdf += applied_k
        else:
            norm_factor = self.n
            for i, k in enumerate(self.kernels):
                    applied_k = k.pdf(eval_points)# * self.weights[i]
                    pdf += applied_k
        if obs == 1:
            # return as float value
            return (pdf/norm_factor)[0]
        return pdf/norm_factor

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

def one_cluster_sample():
    field = Field(
    pm=stats.multivariate_normal(mean=(0., 0.), cov=20),
    space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)),
    radius=10), star_count=int(1e4))
    clusters = [
        Cluster(
            space=stats.multivariate_normal(mean=polar_to_cartesian([120.7, -28.5, 5]), cov=.5),
            pm=stats.multivariate_normal(mean=(.5, 0), cov=1./35),
            star_count=200
        ),
    ]
    df = Synthetic(field=field, clusters=clusters).rvs()
    return df

def test_hkde():
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
    s = df.to_numpy()[:,0:2]
    obs, dims = s.shape
    
    kdeh = HKDE().fit(s)
    Hkdeh = kdeh.covariances[0]
    kdeh_pdf = kdeh.pdf(s)
    bws = [scotts_rule(np.atleast_2d(s[:,i]).T) for i in range(dims)]

    #ss = RobustScaler().fit(s).transform(s)
    # with scipy i cannot set different bws for different dims
    # nor different bw per dim
    # default is cov(data) * scott_factor**2
    scipykde = gaussian_kde(s.T, bw_method='scott')
    scipy_pdf = scipykde.pdf(s.T)
    # with KDEpy i cannot set different bws per dims
    # I can set different bw per obs but no different bw per obs per dim
    # default is 1
    kdepykde = NaiveKDE().fit(s)
    kdepy_pdf = kdepykde.evaluate(s)
    # with statsmodels i can set different bws per dims, but not different bws per dim per obs
    stmodkde = KDEMultivariate(s, 'c'*dims, bw=[1]*dims)
    stmod_pdf = stmodkde.pdf(s)

    rkde_pdf, rH = rkde(s)


    x = s[:,0]
    y = s[:,1]
    
    # assert np.allclose(kdeh_pdf, kdepy_pdf, atol=.001)
    
    plt.figure()
    sns.scatterplot(x,y,hue=scipy_pdf).set(title='scipy')
    plt.figure()
    sns.scatterplot(x,y,hue=kdeh_pdf).set(title='kdeh')
    plt.figure()
    sns.scatterplot(x,y,hue=kdepy_pdf).set(title='kdepy')
    plt.figure()
    sns.scatterplot(x,y,hue=stmod_pdf).set(title='statsmodels')
    plt.figure()
    sns.scatterplot(x,y,hue=rkde_pdf).set(title='ks')
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
    plt.figure()
    sns.histplot(rkde_pdf, bins=50).set(title='dens ks')
    plt.figure()
    # ES IDéntico!!! Siiiiii
    sns.histplot(kdeh_pdf**2-rkde_pdf**2, bins=50).set(title='diff')
    plt.show()
    
    print('coso')

def test_diff_bw_options():
    df = one_cluster_sample()
    d = df.to_numpy()[:,0:3]

    x = d[:,0]
    y = d[:,1]

    obs, dims = d.shape
    kde_default = HKDE().fit(d)
    H_default = kde_default.covariances[0]
    pdf_default = kde_default.pdf(d)
    kde_unconstr = HKDE(bw=PluginBandwidth(pilot='unconstr')).fit(d)
    H_unconstr = kde_unconstr.covariances[0]
    pdf_unconstr = kde_default.pdf(d)
    kde_binned = HKDE(bw=PluginBandwidth(binned=True)).fit(d)
    H_binned = kde_binned.covariances[0]
    pdf_binned = kde_default.pdf(d)

    plt.figure()
    sns.scatterplot(x,y,hue=pdf_default).set(title='default')
    plt.figure()
    sns.scatterplot(x,y,hue=pdf_binned).set(title='binned')
    plt.figure()
    sns.scatterplot(x,y,hue=pdf_unconstr).set(title='unconstr')
    
    plt.figure()
    sns.histplot(pdf_default, bins=50).set(title='dens ks')
    plt.figure()
    sns.histplot(pdf_binned, bins=50).set(title='dens ks')
    plt.figure()
    sns.histplot(pdf_unconstr, bins=50).set(title='dens ks')
    plt.show()
    print('coso')    

def test_performance():
    # it takes about 
    df = three_clusters_sample(int(1e5))
    s = df.to_numpy()[:,0:3]
    obs, dims = s.shape
    pdf = HKDE().fit(s).pdf(s)

def test_weigths():
    df = three_clusters_sample(1000)
    s = df.to_numpy()[:,0:3]
    obs, dims = s.shape
    pdf1 = HKDE().fit(s, weights=np.ones(obs)).pdf(s)
    # raises error
    # pdf2 = HKDE().fit(s, weights=np.zeros(obs)).pdf(s)
    pdf3 = HKDE().fit(s, weights=np.ones(obs)*.5).pdf(s)
    assert np.allclose(pdf1, pdf3*2)
    print('coso')

test_weigths()
