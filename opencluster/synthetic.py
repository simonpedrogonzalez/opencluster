import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from typing_extensions import TypedDict
from typing import Optional, Tuple, List, Union
from attr import attrib, attrs, validators
import copy
import math
from astropy.coordinates import SkyCoord
import astropy.units as u

np.seterr(over='raise')
# np.seterr(invalid='raise')
# np.seterr(invalid='divide')
np.random.seed(0)


class EDSD(stats.rv_continuous):
    def _pdf(self, x, wl, w0, wf):
        return np.piecewise(x, [(x <= w0) + (x >= wf)], [
            0,
            lambda x: wl**3/(2*(x-w0))**4 * np.exp(-wl/(x-w0))
        ])

    """  def _simple_ppf(self, y, wl, w0, wf):
        return (
            .1 +
            w0 + .4*y**4 +
            2*wl /
            (((np.log10(
                -5/4*np.log10(1.-y))
                - 3)**2)
                - 6)
        ) """

    def _ppf(self, y, wl, w0, wf):
        return (
            (-0.007*wl*2 + 0.087*wl - 0.12) +
            w0 +
            (0.17 + 0.076*wl)*y**4 +
            (2 - 0.037*wl + 0.0048*wl**2)*wl /
            (((np.log10(
                -5/4*np.log10(1.-y))
                - 3)**2)
                - 6)
        )

    def _cdf(self, x, wl, w0, wf):
        return np.piecewise(x, [x <= w0, x >= wf], [
            0,
            1,
            lambda x: ((2*x**2 + (2*wl - 4*w0)*x + 2*w0**2 - 2*wl*w0 + wl**2) *
                       np.exp(-wl/(x-w0)) /
                       (2*(x-w0)**2))
        ])

    def _argcheck(self, wl, w0, wf):
        return (w0 < wf)

    """ def check_ppf_error(self, wl, w0, wf):
        x = np.linspace(w0, wf, 10000)
        fig, ax = plt.subplots(2, sharex=True)
        y_correcto = self._cdf(x, wl, w0, wf)
        y_pdf = self._pdf(x, wl, w0, wf)
        xy = np.vstack((x, y_correcto, y_pdf)).T
        # remove problematic points
        xy = xy[xy[:, 1] < 0.9985]
        x = xy[:, 0]
        y_correcto = xy[:, 1]
        y_pdf = xy[:, 2]
        ax[0].plot(x, y_correcto, label='cdf')
        ax[0].plot(x, y_pdf, label='pdf')

        x_aprox = self._ppf(y_correcto, wl, w0, wf)
        ax[0].plot(x_aprox, y_correcto, label='fast ppf aprox')
        ax[0].legend()

        y_aprox = self._cdf(x_aprox, wl, w0, wf)
        err = (y_aprox - y_correcto)**2
        ax[1].plot(x, err, label='err')
        ax[1].plot(x, np.zeros_like(x), '--')
        ax[1].plot(x, np.ones_like(x)*.0005, '--')
        plt.setp(ax, xlim=(w0, wf))
        plt.show() """

    def rvs(self, wl, w0, wf, size):
        if not self._argcheck(wl, w0, wf):
            raise ValueError('wf must be greater than w0')
        limits = np.array([max(w0+1e-10, self.a), min(wf-1e-10, self.b)]).astype('float64')
        rv_limits = self._cdf(limits, wl, w0, wf)
        sample = np.array([])
        while(sample.shape[0] < size):
            y = np.random.uniform(
                low=rv_limits[0], high=rv_limits[1], size=size)
            new_sample = self._ppf(y, wl, w0, wf)
            new_sample = new_sample[(new_sample >= limits[0]) & (
                new_sample <= limits[1])]
            sample = np.concatenate((sample, new_sample), axis=0)
        return sample[:size]


def uniform_cube(lows:tuple, highs:tuple, size:int=1):
    return np.random.uniform(low=lows, high=highs, size=(size, 3))


def uniform_sphere(center:tuple, radius:float, size:int=1):
    phi = np.random.uniform(0, 2*np.pi, size)
    cos_theta = np.random.uniform(-1, 1, size)
    theta = np.arccos(cos_theta)
    r = np.cbrt(np.random.uniform(0, radius, size))
    x = r * np.sin(theta) * np.cos(phi) + center[0]
    y = r * np.sin(theta) * np.sin(phi) + center[1]
    z = r * np.cos(theta) + center[2]
    return np.vstack((x, y, z)).T


def norm2d(mean, cov, n):
    return np.random.multivariate_normal(mean, cov, n)


def skewnorm(a, loc, scale, n):
    y = stats.skewnorm.rvs(a, loc, scale, n)
    df = pd.DataFrame(y)
    df.columns = ['plx']
    return df


def norm(mean, sigma, n):
    return np.random.normal(mean, sigma, n)


def truncated_gumbel(lock, scale, low, high, n):
    rv_limits = (stats.gumbel_r.cdf(low, lock, scale),
                 stats.gumbel_r.cdf(high, lock, scale))
    y = np.random.rand(n)*(rv_limits[1] - rv_limits[0]) + rv_limits[0]
    return stats.gumbel_r.ppf(y, lock, scale)

# TODO: make faster: instead of line modifying sampling range, line modifies the results directly.
def plx_error(w0, wl, wf, n, y0=None, x0=None, plx=None):
    n = plx.shape[0]
    err = np.zeros_like(plx)
    table = np.vstack((plx, err)).T
    below_zero = table[table[:, 0] <= 0, 0]
    if plx is not None and y0 and x0 and below_zero.shape[0]:
        table[table[:, 0] > 0, 1] = EDSD(a=0, b=wf).rvs(
            wl, w0, wf, size=(n - below_zero.shape[0]))
        err_low = -y0 / abs(x0 - below_zero.min()) * below_zero
        err_below_zero = np.apply_along_axis(
            lambda x: EDSD(a=x[0], b=wf).rvs(wl, w0, wf, 1),
            axis=1, arr=np.expand_dims(err_low, 1)).flatten()
        table[table[:, 0] <= 0, 1] = err_below_zero
        return table[:, 1]
    else:
        return EDSD(a=0, b=wf).rvs(wl, w0, wf, size=n)


class EDSDParams(TypedDict):
    wl: float
    w0: float
    wf: float
    a: Optional[float]
    b: Optional[float]


class NormParams(TypedDict):
    mean: float
    sigma: float


class PlxErrorParams(EDSDParams):
    y0: Optional[float]
    x0: Optional[float]


class PlxFieldParams(TypedDict):
    dist_params: EDSDParams
    err_params: PlxErrorParams


class PlxClusterParams(TypedDict):
    dist_params: NormParams
    err_params: PlxErrorParams


class Norm2DParams(TypedDict):
    means: Tuple[float, float]
    cov: List[List[float]]


class PmParams(TypedDict):
    dist_params: Norm2DParams
    err_params: Tuple[EDSDParams, EDSDParams]


class ConeRegionParams(TypedDict):
    center: Tuple[float, float]
    radius: float


class SquareRegionParams(TypedDict):
    ra_range: Tuple[float, float]
    dec_range: Tuple[float, float]


class FieldParams(TypedDict):
    plx_params: Optional[PlxFieldParams]
    pm_params: Optional[PmParams]
    space_params: Optional[Union[ConeRegionParams, SquareRegionParams]]

@attrs(auto_attribs=True)
class Field:

    plx_params: Optional[PlxFieldParams]
    pm_params: Optional[PmParams]
    space_params: Optional[Union[ConeRegionParams, SquareRegionParams]]

    def rvs(self, size: float = 1):
        data = pd.DataFrame()
        if self.space_params:
            if 'center' in self.space_params:
                ra_dec = cone_region(self.space_params.get(
                    'center'), self.space_params.get('radius'), size)
            else:
                ra_dec = square_region(self.space_params.get(
                    'ra_range'), self.space_params.grt('dec_range'), size)
            data[['ra', 'dec']] = pd.DataFrame(ra_dec)
        if self.pm_params:
            dist_params = self.pm_params.get('dist_params')
            pm = norm2d(dist_params.get('means'), dist_params.get('cov'), size)
            err_params = copy.deepcopy(self.pm_params.get('error_params'))
            wl, w0, wf = err_params[0].pop('wl'), err_params[0].pop(
                'w0'), err_params[0].pop('wf')
            pmra_err = EDSD(**err_params[0]).rvs(wl, w0, wf, size=size)
            wl, w0, wf = err_params[1].pop('wl'), err_params[1].pop(
                'w0'), err_params[1].pop('wf')
            pmdec_err = EDSD(**err_params[1]).rvs(wl, w0, wf, size=size)
            data[['pmra', 'pmra_error', 'pmdec', 'pmdec_error']] = pd.DataFrame(
                np.vstack((pm[:, 0], pmra_err, pm[:, 1], pmdec_err)).T
            )
        if self.plx_params:
            dist_params = self.plx_params.get('dist_params')
            wl, w0, wf = dist_params.pop('wl'), dist_params.pop(
                'w0'), dist_params.pop('wf')
            plx = EDSD(**dist_params).rvs(wl, w0, wf, size=size)
            err_params = copy.deepcopy(self.plx_params.get('error_params'))
            wl, w0, wf = err_params.pop('wl'), err_params.pop(
                'w0'), err_params.pop('wf')
            y0, x0 = err_params.pop('y0', None), err_params.pop('x0', None)
            if y0:
                plx_arg = plx
            else:
                plx_arg = None
            plx_err = plx_error(w0, wl, wf, size, y0, x0, plx_arg)
            data[['parallax', 'parallax_error']] = pd.DataFrame(
                np.vstack((plx, plx_err)).T)
        return data


@attrs(auto_attribs=True)
class Cluster:

    plx_params: Optional[PlxClusterParams]
    pm_params: Optional[PmParams]
    space_params: Optional[Norm2DParams]

    def rvs(self, size: float = 1):
        data = pd.DataFrame()
        if self.space_params:
            ra_dec = norm2d(self.space_params.get('means'),
                            self.space_params.get('cov'), size)
            data[['ra', 'dec']] = pd.DataFrame(ra_dec)
        if self.pm_params:
            dist_params = self.pm_params.get('dist_params')
            pm = norm2d(dist_params.get('means'), dist_params.get('cov'), size)
            err_params = copy.deepcopy(self.pm_params.get('error_params'))
            wl, w0, wf = err_params[0].pop('wl'), err_params[0].pop(
                'w0'), err_params[0].pop('wf')
            pmra_err = EDSD(**err_params[0]).rvs(wl, w0, wf, size=size)
            wl, w0, wf = err_params[1].pop('wl'), err_params[1].pop(
                'w0'), err_params[1].pop('wf')
            pmdec_err = EDSD(**err_params[1]).rvs(wl, w0, wf, size=size)
            data[['pmra', 'pmra_error', 'pmdec', 'pmdec_error']] = pd.DataFrame(
                np.vstack((pm[:, 0], pmra_err, pm[:, 1], pmdec_err)).T
            )
        if self.plx_params:
            dist_params = self.plx_params.get('dist_params')
            plx = norm(dist_params.get('mean'), dist_params.get('sigma'), size)
            err_params = copy.deepcopy(self.plx_params.get('error_params'))
            wl, w0, wf = err_params.pop('wl'), err_params.pop(
                'w0'), err_params.pop('wf')
            y0, x0 = err_params.pop('y0', None), err_params.pop('x0', None)
            if y0:
                plx_arg = plx
            else:
                plx_arg = None
            plx_err = plx_error(w0, wl, wf, size, y0, x0, plx_arg)
            data[['parallax', 'parallax_error']] = pd.DataFrame(
                np.vstack((plx, plx_err)).T)
        return data


""" @attrs(auto_attribs=True)
class SkySample:
    field_params: """
# radec
# df = square_region((118, 123), (-25, -30), int(4e5))
# df = cone_region((120.5, -27.5), 5, int(4e5))

# pm
# df = norm2d([0, 0], [[10, -10], [-50, 10]], int(4e5))

# sns.histplot(data=df, bins=100)
# sns.scatterplot(x=df['x'], y = df['plx'])
# sns.jointplot(data=df, x='pmra', y='pmdec', marker='.', marginal_kws=dict(bins = 160, fill= False))
# plx
""" w0 = -.1
wl = 1.2
wf = 5
plx = EDSD().rvs(wl, w0, wf, size=int(4e5))
df = pd.DataFrame(plx)
sns.histplot(data=df, bins=100)
plt.show()
 """

# err pm
# option 1 with gumbel function
""" scale, lock = .27, .05
df = pd.DataFrame(truncated_gumbel(lock, scale, 0, 3.4, int(2e5))) """

# option 2 with same as plx
""" w0, wl, wf = -.15, 1.1, 3.4
df = pd.DataFrame(EDSD(a=0, b=3.4).rvs(wl, w0, wf, size=int(2e5)))
sns.histplot(data=df, bins=1000)
plt.show() """

# plx error

""" w0 = -.1
wl = 1.2
wf = 5
n = int(4e5)
plx = EDSD().rvs(wl, w0, wf, size=n)

w0, wl, wf = -.15, 1.1, 4.1
y0, x0 = .15, .15
plx_err = plx_error(w0, wl, wf, y0, x0, plx)
df = pd.DataFrame(np.vstack((plx, plx_err)).T)
df.columns = ['plx', 'plx_err']
df = df[df['plx'] < 1.6]
#sns.histplot(data=df['plx_err'], bins=1000)
# sns.scatterplot(x=df['plx'], y=df['plx_err'], s=.5)
sns.jointplot(data=df, x='plx', y='plx_err', marker='.') # , marginal_kws=dict(bins = 160, fill= False))
plt.show() """

""" plx_error_params = {'w0': -.15, 'wl': 1.1, 'wf': 4.1, 'y0': .15, 'x0': .05}
pm_error_params = (
    {'w0': -.15, 'wl': 1.1, 'wf': 3.4, 'a': 0, 'b': 3.4},
    {'w0': -.15, 'wl': 1.1, 'wf': 3.4, 'a': 0, 'b': 3.4})

field = Field(
    plx_params={
        'dist_params': {'w0': -.1, 'wl': 1.2, 'wf': 5},
        'error_params': plx_error_params
    },
    pm_params={
        'dist_params': {'means': (0., 0.), 'cov': [[10, -10], [-50, 10]]},
        'error_params': pm_error_params
    },
    space_params={'center': (120.5, -27.5), 'radius': 5}
)
field_data = field.rvs(int(1e4))
field_data['tag'] = pd.DataFrame(np.zeros((int(1e4),)))

cluster = Cluster(
    plx_params={
        'dist_params': {'mean': .7, 'sigma': .05},
        'error_params': plx_error_params
    },
    pm_params={
        'dist_params': {'means': (-2.4, 2), 'cov': [[.3, 0], [0, .3]]},
        'error_params': pm_error_params
    },
    space_params={
        'means': (119.5, -26.5),
        'cov': [[.08, 0], [0, .08]]}
) """

""" cluster_data = cluster.rvs(size=200)
cluster_data['tag'] = pd.DataFrame(np.ones((200,)))

data = pd.concat([field_data, cluster_data], axis=0)

print('drawing graphs')

sns.jointplot(data=data, x='parallax', y='parallax_error', hue='tag')
# sns.jointplot(data=data, x='pmra', y='pmra_error', hue='tag')
# sns.jointplot(data=data, x='pmdec', y='pmdec_error', hue='tag')
sns.jointplot(data=data, x='pmra', y='pmdec', hue='tag')
sns.jointplot(data=data, x='ra', y='dec', hue='tag')

plt.show()  # , marginal_kws=dict(bins = 160, fill= False))
 """

# EDSD(a=0, b=3).check_ppf_error(wl=1.1, w0=-.15, wf=4.1)
""" EDSD(a=0, b=3).rvs(wl=1.1, w0=-.15, wf=4.1, size=100)


print('coso') """

def cartesian_to_polar(coords):
    coords = SkyCoord(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        unit='kpc', representation_type='cartesian', frame='icrs'
        )
    coords.representation_type = 'spherical'
    return np.vstack((coords.ra.deg, coords.dec.deg, coords.distance.parallax.mas)).T
    
def polar_to_cartesian(coords):
    coords = SkyCoord(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        unit='kpc', representation_type='spherical', frame='icrs'
        )
    coords.representation_type = 'cartesian'
    return np.vstack((coords.ra.deg, coords.dec.deg, coords.distance.parallax.mas)).T

r = uniform_sphere(center=(0,0,0), radius=1, size=1000)

y = cartesian_to_polar(r)
""" d = pd.DataFrame(r)
d.columns = ['x', 'y', 'z']
sns.pairplot(d) """
plt.show()

