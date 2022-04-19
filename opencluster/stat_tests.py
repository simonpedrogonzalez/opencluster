import os
import sys
from abc import abstractmethod

from astropy.stats import RipleysKEstimator
from attr import attrs, validators, attrib
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from unidip.dip import diptst
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler

sys.path.append(os.path.join(os.path.dirname("opencluster"), "."))
from opencluster.synthetic import (
    Cluster,
    Field,
    Synthetic,
    UniformSphere,
    polar_to_cartesian,
    stats,
    case2_sample0c,
    case2_sample1c,
    case2_sample2c, BivariateUnifom,
)
from opencluster.utils import combinations


@attrs(auto_attribs=True)
class TestResult:
    value: float=None
    passed: bool=None
    pvalue: float=None


class StatTest:
    @abstractmethod
    def test(self, data: np.ndarray, *args, **kwargs) -> TestResult:
        pass


@attrs(auto_attribs=True)
class HopkinsTest(StatTest):
    n_samples: int = None
    metric: str = "euclidean"
    n_iters: int = 100
    # reduction: Callable = np.median
    # interpretation:
    # H0: data comes from uniform distribution
    # H1: data does not come from uniform distribution
    # if h = u/(u+w) ~ 1 => w = 0 luego hay estructura
    # if h = u/(u+w) ~ .5 => w ~ u luego no hay estructura
    # if h > .75 => reject H0, and in general  indicates a clustering
    # tendency at the 90% confidence level.
    threshold: float = None
    pvalue_threshold: float = 0.05

    def get_pvalue(self, value, n_samples):
        """
        Parameters
        ----------
        value : float
            The hopkins score of the dataset (between 0 and 1)
        n_samples : int
            The number of samples used to compute the hopkins score

        Returns
        ---------------------
        pvalue : float
            The pvalue of the hopkins score
        """
        beta = stats.beta(n_samples,n_samples)
        if value > .5:
            return 1 - (beta.cdf(value) - beta.cdf(1-value))
        else:
            return 1 - (beta.cdf(1-value) - beta.cdf(value))


    def test(self, data: np.ndarray, *args, **kwargs):
        """Assess the clusterability of a dataset. A score
        between 0 and 1, a score around 0.5 express
        no clusterability and a score tending to 1
        express a high cluster tendency.

        Parameters
        ----------
        data : numpy array
            The input dataset

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
            n_samples = int(obs * 0.1)
        else:
            n_samples = min(obs, self.n_samples)

        results = []
        for i in range(self.n_iters):
            sample = resample(data, n_samples=n_samples, replace=False)
            if self.metric == "mahalanobis":
                kwargs["V"] = np.cov(sample, rowvar=False)
            tree = BallTree(
                sample, leaf_size=2, metric=self.metric, *args, **kwargs
            )
            dist, _ = tree.query(sample, k=2)
            sample_nn_distance = dist[:, 1]

            max_data = data.max(axis=0)
            min_data = data.min(axis=0)
            uniform_sample = np.random.uniform(
                low=min_data, high=max_data, size=(n_samples, dims)
            )

            dist, _ = tree.query(uniform_sample, k=1)
            uniform_nn_distance = dist[:, 0]

            sample_sum = np.sum(sample_nn_distance ** dims)
            uniform_sum = np.sum(uniform_nn_distance ** dims)
            # sample_sum = self.reduction(sample_nn_distance)
            # uniform_sum = self.reduction(uniform_nn_distance)
            if sample_sum + uniform_sum == 0:
                raise Exception(
                    "The denominator of the hopkins statistics is null"
                )
            results.append(uniform_sum / (uniform_sum + sample_sum))

        value = np.median(np.array(results))
        pvalue = self.get_pvalue(value, n_samples)
        if self.threshold is not None:
            passed = value >= self.threshold
        else:
            passed = pvalue <= self.pvalue_threshold
        return TestResult(value=value, passed=passed, pvalue=pvalue)


@attrs(auto_attribs=True)
class DipTest(StatTest):
    n_samples: int = None
    metric: str = "euclidean"
    pvalue_threshold: float = 0.05

    def test(self, data: np.ndarray, *args, **kwargs):
        """dip test of unimodality over multidimensional
        data based on distance metric"""
        assert len(data.shape) == 2

        obs, dims = data.shape

        if self.n_samples is None:
            n_samples = min(obs, 100)
        else:
            n_samples = min(obs, self.n_samples)

        sample = resample(data, n_samples=n_samples, replace=False)
        dist = np.ravel(
            np.tril(pairwise_distances(sample, metric=self.metric))
        )
        dist = np.msort(dist[dist > 0])
        _, pval, _ = diptst(dist, *args, **kwargs)
        # sns.histplot(dist).set(title=str(pval))
        # plt.show()
        # print(pval)
        passed = pval < self.pvalue_threshold
        return TestResult(pvalue=pval, passed=passed)


@attrs(auto_attribs=True)
class RipleysKTest(StatTest):
    rk_estimator: RipleysKEstimator = None
    scaler = None
    pvalue_threshold: int = attrib(validator=validators.in_([.05, .01]), default=0.05)
    csr_factors = {
        .05: 1.45,
        .01: 1.68,
    }

    def empiric_csr_rule(self, l_function, radii, area, n):
        supremum = np.max(np.abs(l_function-radii))
        factor = self.csr_factors[self.pvalue_threshold]
        return supremum >= factor * np.sqrt(area)/n

    def test(
        self, data: np.ndarray, radii: np.ndarray = None, *args, **kwargs
    ):

        obs, dims = data.shape
        if dims != 2:
            raise ValueError("Data must be bidimensional.")

        if self.scaler is not None:
            data = self.scaler.fit_transform(data)

        x_min=data[:, 0].min()
        x_max=data[:, 0].max()
        y_min=data[:, 1].min()
        y_max=data[:, 1].max()

        if radii is None:
            # considers rectangular window
            short_side = min(x_max - x_min, y_max - y_min)
            radii_max_ripley = short_side / 4
            radii_max_large = np.sqrt(100/(np.pi*obs))
            radii_max = min(radii_max_ripley, radii_max_large)
            step = radii_max / 128 / 4
            radii = np.arange(0, radii_max, step)

        if self.rk_estimator is None:
            area = ConvexHull(data).volume
            self.rk_estimator = RipleysKEstimator(
                area=area,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
        else:
            area = self.rk_estimator.area

        if kwargs.get("mode") is None:
            # best mode for rectangular window
            kwargs["mode"] = "ripley"

        l_function = self.rk_estimator.Lfunction(data, radii, *args, **kwargs)

        value = np.max(np.abs(l_function - radii))
        
        passed = self.empiric_csr_rule(l_function, radii, area, obs)

        return TestResult(value=value, passed=passed)


def test_dip():
    ns = [100, 1000, int(1e4)]
    # case uniform
    uniforms = [
        UniformSphere(
            center=polar_to_cartesian((120.5, -27.5, 5)), radius=10
        ).rvs(n)
        for n in ns
    ]

    results_euclidean = np.array(
        [DipTest(metric="euclidean").test(u) for u in uniforms]
    )
    results_mahalanobis = np.array([DipTest().test(u) for u in uniforms])

    print(results_euclidean)
    print(results_mahalanobis)
    assert np.all(results_euclidean > 0.05)
    assert np.all(results_mahalanobis > 0.05)

    # case 1 k
    any_pm = stats.multivariate_normal(mean=(7.5, 7), cov=1.0 / 35)
    c_f_mixes = [0.1, 0.5, 0.9]
    cov_diag = 0.5
    random_matrix = np.random.rand(3, 3)
    cov_full = [np.dot(rm, rm.T) for rm in random_matrix]
    covs = [cov_diag, cov_full]
    metrics = ["euclidean", "mahalanobis"]
    parameters = combinations([ns, c_f_mixes, covs, metrics])
    oneclusters = [
        Synthetic(
            field=Field(
                space=UniformSphere(
                    center=polar_to_cartesian((120.5, -27.5, 5)), radius=10
                ),
                pm=any_pm,
                star_count=int(p[0] * (1 - p[1])),
            ),
            clusters=[
                Cluster(
                    space=stats.multivariate_normal(
                        mean=polar_to_cartesian([120.7, -28.5, 5]),
                        cov=p[2],
                    ),
                    pm=any_pm,
                    star_count=int(p[0] * p[1]),
                ),
            ],
            representation_type="cartesian",
        )
        .rvs()[["x", "y", "z"]]
        .to_numpy()
        for p in parameters
    ]
    results = [
        DipTest(metric=p[3]).test(oc) for oc, p in zip(oneclusters, parameters)
    ]
    a = [(p[0], p[1], r) for p, r in zip(parameters, results)]
    print(results)
    print("coso")
    assert np.all(results < 0.2)

    # NOTE: "fails" when mix is too imbalanced, e.g. .1 to .9, or viceversa
    # Meaning:
    # if dip > .1
    # there is unimodal tendency, there are no clusters or there is only
    # one cluster and no noise
    # if dip < .1
    # if there is multimodal tendency
    #   there are several clusters
    #   or one cluster + noise

    # case 2 k
    any_pm = stats.multivariate_normal(mean=(7.5, 7), cov=1.0 / 35)
    c_f_mixes = [0.1, 0.5, 0.9]
    cov_diag = 0.5
    random_matrix = np.random.rand(3, 3)
    cov_full = [np.dot(rm, rm.T) for rm in random_matrix]
    covs = [cov_diag, cov_full]
    metrics = ["euclidean", "mahalanobis"]
    parameters = combinations([ns, c_f_mixes, covs, metrics])

    twoclusters = [
        Synthetic(
            field=Field(
                space=UniformSphere(
                    center=polar_to_cartesian((120.5, -27.5, 5)), radius=10
                ),
                pm=any_pm,
                star_count=int(p[0] * (1 - p[1])),
            ),
            clusters=[
                Cluster(
                    space=stats.multivariate_normal(
                        mean=polar_to_cartesian([119.5, -28.5, 4.8]),
                        cov=p[2],
                    ),
                    pm=any_pm,
                    star_count=int(p[0] * p[1]),
                ),
                Cluster(
                    space=stats.multivariate_normal(
                        mean=polar_to_cartesian([121.5, -26.5, 5.2]),
                        cov=p[2],
                    ),
                    pm=any_pm,
                    star_count=int(p[0] * p[1]),
                ),
            ],
            representation_type="cartesian",
        )
        .rvs()[["x", "y", "z"]]
        .to_numpy()
        for p in parameters
    ]
    results = [
        DipTest(metric=p[3]).test(oc) for oc, p in zip(twoclusters, parameters)
    ]
    a = [(p[0], p[1], r) for p, r in zip(parameters, results)]
    print(a)
    print(results)
    print("coso")


def test_hopkins():
    ns = [100, 1000, int(1e4)]
    # case uniform
    uniforms = [
        UniformSphere(
            center=polar_to_cartesian((120.5, -27.5, 5)), radius=10
        ).rvs(n)
        for n in ns
    ]
    metrics = ["euclidean", "mahalanobis"]
    reductions = [np.median, np.mean, np.sum, lambda x: np.sum(x ** 3)]
    cases = combinations([metrics, reductions, uniforms])
    results = [
        (len(d), m, f, HopkinsTest(metric=m, reduction=f).test(d))
        for m, f, d in cases
    ]

    false_positives = [r for r in results if r[3].passed]
    df = pd.DataFrame([(r[3].value, r[2].__name__) for r in false_positives])
    df.columns = ["v", "f"]
    sns.kdeplot(df.v, hue=df.f)
    plt.show()

    print(results)

    # assert np.all(np.array([r.passed for _,_,r in results]) == False)

    # all good except for np.sum(x**3), which makes the test pass in
    # some cases while it should not

    any_pm = stats.multivariate_normal(mean=(7.5, 7), cov=1.0 / 35)
    c_f_mixes = [0.1, 0.5, 0.9]
    cov_diag = 0.5
    random_matrix = np.random.rand(3, 3)
    cov_full = [np.dot(rm, rm.T) for rm in random_matrix]
    covs = [cov_diag, cov_full]
    cases = combinations([metrics, reductions, ns, c_f_mixes, covs])

    results = [
        (
            n,
            p,
            m,
            f,
            c,
            HopkinsTest(metric=m, reduction=f).test(
                Synthetic(
                    field=Field(
                        space=UniformSphere(
                            center=polar_to_cartesian((120.5, -27.5, 5)),
                            radius=10,
                        ),
                        pm=any_pm,
                        star_count=int(n * (1 - p)),
                    ),
                    clusters=[
                        Cluster(
                            space=stats.multivariate_normal(
                                mean=polar_to_cartesian([120.7, -28.5, 5]),
                                cov=c,
                            ),
                            pm=any_pm,
                            star_count=int(n * p),
                        ),
                    ],
                    representation_type="cartesian",
                )
                .rvs()[["x", "y", "z"]]
                .to_numpy()
            ),
            Synthetic(
                field=Field(
                    space=UniformSphere(
                        center=polar_to_cartesian((120.5, -27.5, 5)), radius=10
                    ),
                    pm=any_pm,
                    star_count=int(n * (1 - p)),
                ),
                clusters=[
                    Cluster(
                        space=stats.multivariate_normal(
                            mean=polar_to_cartesian([120.7, -28.5, 5]),
                            cov=c,
                        ),
                        pm=any_pm,
                        star_count=int(n * p),
                    ),
                ],
                representation_type="cartesian",
            )
            .rvs()[["x", "y", "z"]]
            .to_numpy(),
        )
        for m, f, n, p, c in cases
    ]

    # 1/3 of results are false negatives !!!
    # whats wrong here
    false_negatives = [r for r in results if r[5].passed is False]
    print(false_negatives)
    df = pd.DataFrame([(r[5].value, r[3].__name__) for r in false_positives])
    df.columns = ["v", "f"]
    sns.kdeplot(df.v, hue=df.f)
    plt.show()
    assert np.all(np.array([r.passed for _, _, r in results]) is True)


def test_hopkins2():
    f_ratios = [0.6, 0.7, 0.8, 0.9]

    space = ['ra', 'dec']
    pm = ['pmra', 'pmdec']
    space_plx = ['ra', 'dec', 'parallax']
    pm_plx = ['pmra', 'pmdec', 'parallax']
    pm_plx_space = ['ra', 'dec', 'pmra', 'pmdec', 'parallax']
    no_plx = ['ra', 'dec', 'pmra', 'pmdec']
    xyz = ['x', 'y', 'z']

    cols_2d = [space, pm]
    cols_3d = [space_plx, pm_plx]
    cols_5d = [pm_plx_space]
    cols_all = [space, pm, space_plx, pm_plx, pm_plx_space, no_plx]
    cols_all = [pm, space]

    funcs = [case2_sample0c, case2_sample1c, case2_sample2c]

    cs = combinations([funcs, f_ratios, cols_all])
    from rpy2.robjects import r
    from opencluster.rutils import rhardload, pyargs2r, rclean
    rhardload(r, 'hopkins')
    res = []
    for c in cs:

        sample = c[0](c[1])[c[2]].to_numpy()
        sample = RobustScaler().fit_transform(sample)
        

        rclean(r, "var")
        _, rparams = pyargs2r(r, data=sample, n=100)
        r_res = np.asarray(r(f'hv = hopkins(data,n)'))
        r_res_pval = np.asarray(r(f'hopkins.pval(hv,n)'))
        # r_res2 = np.asarray(r(f'get_clust_tendency(data,n)$hopkins_stat'))

        test_result = HopkinsTest().test(sample)
        if c[0].__name__ == 'case2_sample0c':
            n_clu = 0
            passed = False
        else:
            passed = True
            if c[0].__name__ == 'case2_sample1c':
                n_clu = 1
            else:
                n_clu = 2
        err = passed == test_result.passed
        r_passed = r_res_pval.ravel()[0] <= .05
        r_err = r_passed == passed
        res.append((n_clu, c[0].__name__, c[1], str(c[2]), len(c[2]), test_result.value, test_result.pvalue, test_result.passed, passed, test_result.passed == passed, r_res.ravel()[0], r_res_pval.ravel()[0], err, r_err))
    
    df = pd.DataFrame(res)
    df.columns = ['n_clu', 'func', 'f_ratio', 'cols', 'dims', 'value', 'pvalue', 'passed', 'expected', 'passed_equal', 'r_value', 'r_pval', 'err', 'r_err' ]
    df['err'] = df.err.astype(int)
    df['r_err'] = df.err.astype(int)
    print(df)
    # if p value > .05, the likelihood of the null hypothesis is > .05, then there is no enough evidence of the alternative hypothesis
    # H0: sample comes from random distribution

    # NO FUNCIONA BIEN EN 2D PORQUE ES EL UNICO CASO DONDE NO ESTA EL PARALAJE QUE ES LO QUE MEJOR AGRUPA
    # DA FALSO POSITIVO CUANDO SE USA PARALAJE PORQUE NO ESTA UNIFORMEMENTE DISTRIBUIDO
    # DA FALSO NEGATIVO CUANDO NO SE USA PARALAJE PORQUE ESTA MENOS UNIFORMEMENTE DISTRIBUIDO
    # SACAR EL PARALAJE SOLUCIONARIA LOS DOS PROBLEMAS
    # ALTERNATIVAMENTE, SE PUEDE PROBAR CON XYZ EN LUGAR DE RA DEC PLX

def test_hopkins3():
    from sklearn.datasets import load_iris
    data = load_iris().data
    np.random.seed(0)
    test_result = HopkinsTest().test(data)
    print(test_result)

def rhopkins():
    from rpy2.robjects import r
    from opencluster.rutils import rhardload, pyargs2r, rclean
    rhardload(r, 'hopkins')
    rhardload(r, 'pdist')
    r("""hopkins2 <- function (X, m=ceiling(nrow(X)/10), d=ncol(X), U=NULL) {
  if (!(is.matrix(X)) & !(is.data.frame(X))) 
    stop("X must be data.frame or matrix")

  if (m > nrow(X)) 
    stop("m must be no larger than num of samples")

  if(missing(U)) {
    # U is a matrix of column-wise uniform values sampled from the space of X
    colmin <- apply(X, 2, min)
    colmax <- apply(X, 2, max)    
    U <- matrix(0, ncol = ncol(X), nrow = m)
    for (i in 1:ncol(X)) {
      U[, i] <- runif(m, min = colmin[i], max = colmax[i])
    }
  } else {
    # The user has provided the uniform values.
  }

  # Random sample of m rows in X (without replacement)
  k <- sample(1:nrow(X), m)
  W <- X[k, , drop=FALSE]   # Need 'drop' in case X is single-column
  
  # distance between each row of W and each row of X
  dwx <- as.matrix(pdist(W,X))
  # Caution: W[i,] is the same point as X[k[i],] and the distance between them is 0,
  # but we do not want to consider that when calculating the minimum distance
  # between W[i,] and X, so change the distance from 0 to Inf
  for(i in 1:m) dwx[i,k[i]] <- Inf
  # distance from each row of W to the NEAREST row of X
  dwx <- apply(dwx, 1, min)
  
  # distance between each row of U and each row of X
  dux <- as.matrix(pdist(U,X)) # rows of dux refer to U, cols refer to X
  # distance from each row of U to the NEAREST row of X
  dux <- apply(dux, 1, min)

  # You would think this would be faster, but it is not for our test cases:
  # stat = 1 / (1 + sum(dwx^d) / sum( dux^d ) )
  
  return( sum(dux^d) / sum( dux^d + dwx^d ) )
}""")

    from sklearn import datasets
    iris = datasets.load_iris()
    min_data = np.min(iris.data, axis=0)
    max_data = np.max(iris.data, axis=0)
    dims=4
    np.random.seed(0)
    uniform_sample = np.random.uniform(
                low=min_data, high=max_data, size=(150, dims)
            )
    pyargs2r(r, U=uniform_sample)
    pyargs2r(r, X=iris.data)
    r_res = np.asarray(r(f'hv = hopkins2(X,m=150,U=U)')).ravel()[0]
    r_pval = np.asarray(r('hvpval = hopkins.pval(hv, 150)')).ravel()[0]
    np.random.seed(0)
    test_result = HopkinsTest(n_iters=1,n_samples=150).test(iris.data)
    print(test_result.value)
    print(r_res)
    print(r_pval)

def uniform_sample():
    return BivariateUnifom(locs=(0,0), scales=(1, 1)).rvs(1000)

def cluster_structure_sample():
    sample = BivariateUnifom(locs=(0,0), scales=(1, 1)).rvs(100)
    sample2 = stats.multivariate_normal(mean=(.5, .5), cov=1./500).rvs(900)
    return np.concatenate((sample, sample2))

def test_ripleys():
    us = uniform_sample()
    usr = RipleysKTest().test(data=us)
    cl = cluster_structure_sample()
    clr = RipleysKTest().test(data=cl)
    prin

def rripley():
    from rpy2.robjects import r
    from opencluster.rutils import rhardload, pyargs2r, rclean
    rhardload(r, 'spatstat')

    n = 1000

    u = uniform_sample()
    pyargs2r(r, u=u)
    r('W <- owin(c(0,1), c(0,1))')
    r('u <- as.ppp(as.matrix(u), W=W)')
    r('plot(u)')
    r('dev.off()')
    r('le = Lest(as.ppp(u, c(0,1,0,1)), correction="Ripley")')
    radii = np.asarray(r('le$r'))
    lf = np.asarray(r('le$iso'))
    diff = np.abs(lf-radii)
    value = diff.max()
    threshold = 1.68 * np.sqrt(1)/n
    passed = value >= threshold
    correct = not passed
    print(f'uniform {correct}')

    lfa = RipleysKEstimator(area=1, x_max=1, x_min=0, y_min=0, y_max=1).Lfunction(u, radii, mode='ripley')
    diffa = np.abs(lfa-radii)
    valuea = diffa.max()
    thresholda = 1.68 * np.sqrt(1)/n
    passeda = valuea >= thresholda
    correcta = not passeda

    clu = cluster_structure_sample()
    pyargs2r(r, clu=clu)
    r('clu <- as.ppp(as.matrix(clu), W=W)')
    r('plot(clu)')
    r('dev.off()')
    r('le = Lest(as.ppp(as.data.frame(clu), owin(xrange=c(0,1), yrange=c(0,1))), correction="Ripley")')
    radii = np.asarray(r('le$r'))
    lf = np.asarray(r('le$iso'))
    diff = np.abs(lf-radii)
    value = diff.max()
    threshold = 1.68 * np.sqrt(1)/n
    passed = value >= threshold
    correct = passed
    print(f'clu {correct}')



    print(r_res)

test_ripleys()
# rripley()