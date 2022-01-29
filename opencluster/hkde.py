import copy
import os
import sys
from abc import abstractmethod
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from attr import attrs
from KDEpy import NaiveKDE
from KDEpy.bw_selection import (
    improved_sheather_jones,
    scotts_rule,
    silvermans_rule,
)
from rpy2.robjects import r
from rutils import pyargs2r, r2np, rclean, rhardload
from scipy.stats import gaussian_kde, halfnorm, multivariate_normal
from statsmodels.nonparametric.bandwidths import bw_scott, bw_silverman
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.stats.correlation_tools import corr_nearest
sys.path.append(
    os.path.join(
        os.path.dirname("opencluster"), "/home/simon/repos/opencluster"
    )
)
from opencluster.synthetic import (
    Cluster,
    Field,
    Synthetic,
    UniformSphere,
    one_cluster_sample,
    polar_to_cartesian,
    stats,
    three_clusters_sample,
)

# prepare r packages
rhardload(r, ["ks"])


# TODO: move to testing file
def rkde(data):
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import packages as rpackages
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr

    numpy2ri.activate()
    # from rpy2.robjects.vectors import StrVector
    # rpackages_names = StrVector(('ks', ... ))
    try:
        importr("ks")
    except Exception:
        utils = rpackages.importr("utils")
        utils.chooseCRANmirror(ind=1)
        utils.install_packages("ks")
        importr("ks")
    obs, dims = data.shape

    r.assign("data", data)
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
    pilot: str = None
    binned: bool = None
    diag: bool = False

    def H(self, data):

        _, dims = data.shape
        params = copy.deepcopy(self.__dict__)
        diag = params.pop("diag")

        # delete all previous session variables
        rclean(r, "var")
        _, rparams = pyargs2r(r, x=data, **params)
        result = r(f'ks::Hpi{".diag" if diag else ""}({rparams})')
        return r2np(result, (dims, dims))


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
        if isinstance(self.bwbw, Bandwidth):
            self.bw.diag = True
            variance = np.diag(self.bw.H(data))
        elif isinstance(self.bw, (int, float)):
            variance = np.ones(dims) * self.bw
        elif isinstance(self.bw, (np.ndarray, list)):
            bw = np.array(self.bw)
            if bw.shape != (dims,):
                raise ValueError("Wrong shape in bw array")
            variance = bw
        if err is None:
            sigmas = np.sqrt(variance)
            return np.repeat(sigmas[:, np.newaxis], obs, 1).T
        elif data.shape != err.shape:
            raise ValueError(
                "error matrix and data matrix shapes do not match"
            )
        else:
            # TODO: check
            variance_with_err = err ** 2 + variance ** 2
            sigmas = np.sqrt(variance_with_err)
            return sigmas

    def get_corr_coefs(self, data: np.ndarray, corr: Union[np.ndarray, str]):
        obs, dims = data.shape
        # correlation is given
        if isinstance(corr, str):
            if corr == "zero":
                corrs = np.zeros((obs, dims, dims))
                diag_idcs = tuple(
                    map(
                        tuple,
                        np.vstack(
                            (
                                np.arange(obs).repeat(dims),
                                np.tile(
                                    np.array(np.diag_indices(dims)), (obs,)
                                ),
                            )
                        ),
                    )
                )
                corrs[diag_idcs] = 1
                return corrs
            elif (
                corr == "auto"
            ):  # correlation is not given, calculate from data
                # this may be theorically incorrect in the case
                # of pm plx, but may be useful in other contexts
                return (
                    np.repeat(
                        corr_nearest(np.corrcoef(data, rowvar=False)),
                        repeats=obs,
                        axis=1,
                    )
                    .reshape((dims, dims, obs))
                    .T
                )

        elif isinstance(corr, np.ndarray):
            # is array
            if corr.shape == (dims, dims):
                # correlation is given as global correlation matrix per dims
                return (
                    np.repeat(corr, repeats=obs, axis=1)
                    .reshape((dims, dims, obs))
                    .T
                )
            elif corr.shape == (obs, int(dims * (dims - 1) / 2)):
                # correlation is given per observation per obs, per dims
                # pairwise corr coef
                # (no need for the 1s given by corr(samevar, samevar))
                # per observation. Example: for 1 obs and 4 vars, lower triangle of corr
                # matrix looks like:
                # 12
                # 13 23
                # 14 24 34
                # method should receive obs1 => [12, 13, 23, 14, 24, 34]
                n_corrs = corr.shape[1]
                corrs = np.zeros((obs, dims, dims))
                tril_idcs = tuple(
                    map(
                        tuple,
                        np.vstack(
                            (
                                np.arange(obs).repeat(n_corrs),
                                np.tile(
                                    np.array(np.tril_indices(dims, k=-1)),
                                    (obs,),
                                ),
                            )
                        ),
                    )
                )
                corrs[tril_idcs] = corr.ravel()
                corrs = corrs + np.transpose(corrs, (0, 2, 1))
                diag_idcs = tuple(
                    map(
                        tuple,
                        np.vstack(
                            (
                                np.arange(obs).repeat(dims),
                                np.tile(
                                    np.array(np.diag_indices(dims)), (obs,)
                                ),
                            )
                        ),
                    )
                )
                corrs[diag_idcs] = 1
                return corrs
            else:
                raise ValueError("Wrong corr dimensions")
        raise ValueError("Wrong corr parameter")

    def get_cov_matrices(self, data, err, corr):
        obs, dims = data.shape
        # checked
        if err is None and corr == "auto" and isinstance(self.bw, Bandwidth):
            # calculate full H from plugin method
            cov = self.bw.H(data)
            # repeat covariance for each obs
            return np.repeat(cov[:, np.newaxis], obs, 1).swapaxes(0, 1)
        # TODO: check
        else:
            # get sigma value for each dimension
            sigmas = self.get_sigmas(data, err)
            # get correlation coefficients
            corr_coefs = self.get_corr_coefs(data, corr)
            # get covariance matrices Sdxd per observations, where each element is
            # sigmai*sigmaj for i,j = 1,...,dims
            cross_sigmas = np.apply_along_axis(
                lambda x: x * np.atleast_2d(x).T, -1, sigmas
            )
            cov = cross_sigmas * corr_coefs
        return cov

    def set_weigths(self, weights: np.ndarray):
        if len(weights.shape) != 1:
            raise ValueError("Weights must be 1d np ndarray.")
        if np.any(weights > 1):
            raise ValueError("Weight values must belong to [0,1].")
        self.weights = weights
        self.n = np.sum(self.weights)
        return self

    def fit(
        self,
        data: np.ndarray,
        err: np.ndarray = None,
        corr: Union[np.ndarray, str] = "auto",
        weights: np.ndarray = None,
        *args,
        **kwargs,
    ):
        obs, dims = data.shape
        self.d = dims

        if weights is not None:
            self.set_weigths(weights)

        if self.weights is not None:
            if self.weights.shape[0] != obs:
                raise ValueError("Data must have same n as weigths.")
            data = data[self.weights > 0]
            self.weights = self.weights[self.weights > 0]
            obs, dims = data.shape
        else:
            self.set_weigths(np.ones(obs))

        if obs == 0:
            raise ValueError(
                "Data matrix is empty or all points are weighted 0"
            )

        print("getting cov")
        self.covariances = self.get_cov_matrices(data, err, corr)

        print("getting kern")
        self.kernels = np.array(
            [
                multivariate_normal(
                    data[i],
                    self.covariances[i],
                    *args,
                    **kwargs,
                )
                for i in range(obs)
            ]
        )
        print("done fit")
        return self

    def pdf(self, eval_points: np.ndarray, leave1out=False):
        if self.kernels is None:
            raise Exception(
                "Model not fitted. Try excecuting fit function first."
            )
        obs, dims = eval_points.shape
        if dims != self.d:
            raise ValueError("Eval points must have same dims as data.")
        print("eval")
        pdf = np.zeros(obs)

        # put weigths and normalization toghether in each step
        # pdf(point) = sum(ki(point)*wi/(sum(w)-wi))

        # TODO: include xi dispersion when getting sum(kj(xi))
        # should be sum((kj*ki)(xi)), that is, convolve ki with kj.
        # should be equivalent to aggregating Hi to Hj cuadratically

        norm_weigths = self.weights / (self.n - self.weights)
        if leave1out:
            for i, k in enumerate(self.kernels):
                applied_k = k.pdf(eval_points) * norm_weigths[i]
                applied_k[i] = 0
                pdf += applied_k
        else:
            for i, k in enumerate(self.kernels):
                applied_k = k.pdf(eval_points) * norm_weigths[i]
                pdf += applied_k
        if obs == 1:
            # return as float value
            return pdf[0]
        return pdf


def test_corr():
    obs = 3
    dims = 3
    x = np.linspace(-1, 1, 10)
    y = x
    z = x
    x, y, z = np.mgrid[-1:1:50j, -1:1:50j, -1:1:50j]
    p = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    data = np.random.normal(size=(obs, dims))
    errors = halfnorm().rvs((obs, dims))

    pdf = HKDE().fit(data, errors).pdf(data)
    sns.scatterplot(data[:, 0], data[:, 1], hue=pdf)
    plt.show()
    sns.scatterplot(p[:, 0], p[:, 1], pdf)

    plt.show()
    print("coso")


def test_bandwidth():
    field = Field(
        pm=stats.multivariate_normal(mean=(0.0, 0.0), cov=20),
        space=UniformSphere(
            center=polar_to_cartesian((120.5, -27.5, 5)), radius=10
        ),
        star_count=int(1e3),
    )
    clusters = [
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([120.7, -28.5, 5]), cov=0.5
            ),
            pm=stats.multivariate_normal(mean=(0.5, 0), cov=1.0 / 35),
            star_count=200,
        ),
    ]
    s = Synthetic(field=field, clusters=clusters).rvs().to_numpy()
    obs, dims = s.shape
    funcs = [
        scotts_rule,
        bw_scott,
        silvermans_rule,
        bw_silverman,
        improved_sheather_jones,
    ]
    data = []
    for f in funcs:
        for i in range(dims):
            bw = f(np.atleast_2d(s[:, i]).T)
            if isinstance(bw, np.ndarray):
                bw = bw[0]
            data.append({"var": i + 1, "func": f.__name__, "bw": bw})
    df = pd.DataFrame(data)
    sns.lineplot(data=df, x="var", y="bw", hue="func")
    print(s)


def test_hkde():
    field = Field(
        pm=stats.multivariate_normal(mean=(0.0, 0.0), cov=20),
        space=UniformSphere(
            center=polar_to_cartesian((120.5, -27.5, 5)), radius=10
        ),
        star_count=int(1e3),
    )
    clusters = [
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([120.7, -28.5, 5]), cov=0.5
            ),
            pm=stats.multivariate_normal(mean=(0.5, 0), cov=1.0 / 35),
            star_count=200,
        ),
    ]
    df = Synthetic(field=field, clusters=clusters).rvs()
    s = df.to_numpy()[:, 0:2]
    obs, dims = s.shape

    kdeh = HKDE().fit(s)
    Hkdeh = kdeh.covariances[0]
    print(Hkdeh)
    kdeh_pdf = kdeh.pdf(s)
    bws = [scotts_rule(np.atleast_2d(s[:, i]).T) for i in range(dims)]
    print(bws)
    # ss = RobustScaler().fit(s).transform(s)
    # with scipy i cannot set different bws for different dims
    # nor different bw per dim
    # default is cov(data) * scott_factor**2
    scipykde = gaussian_kde(s.T, bw_method="scott")
    scipy_pdf = scipykde.pdf(s.T)
    # with KDEpy i cannot set different bws per dims
    # I can set different bw per obs but no different bw per obs per dim
    # default is 1
    kdepykde = NaiveKDE().fit(s)
    kdepy_pdf = kdepykde.evaluate(s)
    # with statsmodels i can set different bws per dims,
    # but not different bws per dim per obs
    stmodkde = KDEMultivariate(s, "c" * dims, bw=[1] * dims)
    stmod_pdf = stmodkde.pdf(s)

    rkde_pdf, rH = rkde(s)

    x = s[:, 0]
    y = s[:, 1]

    # assert np.allclose(kdeh_pdf, kdepy_pdf, atol=.001)

    plt.figure()
    sns.scatterplot(x, y, hue=scipy_pdf).set(title="scipy")
    plt.figure()
    sns.scatterplot(x, y, hue=kdeh_pdf).set(title="kdeh")
    plt.figure()
    sns.scatterplot(x, y, hue=kdepy_pdf).set(title="kdepy")
    plt.figure()
    sns.scatterplot(x, y, hue=stmod_pdf).set(title="statsmodels")
    plt.figure()
    sns.scatterplot(x, y, hue=rkde_pdf).set(title="ks")
    """ plt.figure()
    sns.histplot(np.sqrt(kdeh_pdf**2-kdepy_pdf**2), bins=50).set(title='diff') """
    plt.figure()
    sns.histplot(scipy_pdf, bins=50).set(title="dens scipy")
    plt.figure()
    sns.histplot(kdepy_pdf, bins=50).set(title="dens kdepy")
    plt.figure()
    sns.histplot(kdeh_pdf, bins=50).set(title="dens kdeh")
    plt.figure()
    sns.histplot(stmod_pdf, bins=50).set(title="dens statsmodels")
    plt.figure()
    sns.histplot(rkde_pdf, bins=50).set(title="dens ks")
    plt.figure()
    # ES IDéntico!!! Siiiiii
    sns.histplot(kdeh_pdf ** 2 - rkde_pdf ** 2, bins=50).set(title="diff")
    plt.show()

    print("coso")


def test_diff_bw_options():
    df = one_cluster_sample()
    d = df.to_numpy()[:, 0:3]

    x = d[:, 0]
    y = d[:, 1]

    obs, dims = d.shape
    kde_default = HKDE().fit(d)
    H_default = kde_default.covariances[0]
    pdf_default = kde_default.pdf(d)
    kde_unconstr = HKDE(bw=PluginBandwidth(pilot="unconstr")).fit(d)
    H_unconstr = kde_unconstr.covariances[0]
    pdf_unconstr = kde_default.pdf(d)
    kde_binned = HKDE(bw=PluginBandwidth(binned=True)).fit(d)
    H_binned = kde_binned.covariances[0]
    pdf_binned = kde_default.pdf(d)

    plt.figure()
    sns.scatterplot(x, y, hue=pdf_default).set(title="default")
    plt.figure()
    sns.scatterplot(x, y, hue=pdf_binned).set(title="binned")
    plt.figure()
    sns.scatterplot(x, y, hue=pdf_unconstr).set(title="unconstr")

    plt.figure()
    sns.histplot(pdf_default, bins=50).set(title="dens ks")
    plt.figure()
    sns.histplot(pdf_binned, bins=50).set(title="dens ks")
    plt.figure()
    sns.histplot(pdf_unconstr, bins=50).set(title="dens ks")
    plt.show()
    print("binned")
    print(H_binned)
    print("unconstr")
    print(H_unconstr)
    print("default")
    print(H_default)


def test_performance():
    np.random.seed(0)
    df = three_clusters_sample(int(1e3))
    s = df.to_numpy()[:, 0:3]
    obs, dims = s.shape
    for i in range(10):
        HKDE().fit(s).pdf(s)


def test_weigths():
    np.random.seed(0)
    df = three_clusters_sample(1000)
    s = df.to_numpy()[:, 0:3]
    obs, dims = s.shape
    pdf1 = HKDE().fit(s).pdf(s)
    # raises error
    # pdf2 = HKDE().fit(s, weights=np.zeros(obs)).pdf(s)
    pdf3 = HKDE().set_weigths(np.ones(obs) * 0.5).fit(s).pdf(s)
    pdf4 = HKDE().set_weigths(np.ones(obs) * 0.5).fit(s).pdf(s, leave1out=True)
    assert np.allclose(pdf1, pdf3)
    assert np.allclose(pdf1, pdf4)
    print("coso")


# test_performance()
# test_weigths()
