
import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))


from opencluster.fetcher import load_file
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from typing_extensions import TypedDict
from typing import Optional, Tuple, List, Union, Callable
from attr import attrib, attrs, validators
import copy
import math
from astropy.coordinates import Distance, SkyCoord
import astropy.units as u

# np.seterr(over='raise')
# np.seterr(invalid='raise')
# np.seterr(invalid='divide')

# Coordinate transformation
def cartesian_to_polar(coords):
    coords = SkyCoord(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        unit='pc', representation_type='cartesian', frame='icrs'
    )
    coords.representation_type = 'spherical'
    return np.vstack((coords.ra.deg, coords.dec.deg, coords.distance.parallax.mas)).T

def polar_to_cartesian(coords):
    coords = SkyCoord(
        ra=coords[:, 0]*u.degree, dec=coords[:, 1]*u.degree, distance=Distance(parallax=coords[:, 2]*u.mas),
        representation_type='spherical', frame='icrs'
    )
    coords.representation_type = 'cartesian'
    return np.vstack((coords.x.value, coords.y.value, coords.z.value)).T

# Custom validators
def in_range(min_value, max_value):
    def range_validator(instance, attribute, value):
        if value < float(min_value):
            raise ValueError(f'{attribute.name} attribute must be >= than {min_value}')
        if value > float(max_value):
            raise ValueError(f'{attribute.name} attribute must be <= than {max_value}')
    return range_validator

def dist_has_n_dimensions(n: int):
    def dist_has_n_dimensions_validator(instance, attribute, value):
        if not value.dim:
            raise TypeError(f'{attribute.name} attribute does not have dim property')
        elif value.dim != n:
            raise ValueError(f'{attribute.name} attribute must have {n} dimensions, but has {value.dim} dimensions')
    return dist_has_n_dimensions_validator

def has_len(length: int):
    def has_len_validator(instance, attribute, value):
        if len(value) != length:
            raise ValueError(f'{attribute.name} attribute must have length {length}, but has length {len(value)}')
    return has_len_validator

# Custom distributions
@attrs(auto_attribs=True, init=False)
class EDSD(stats.rv_continuous):
    
    w0: float
    wl: float
    wf: float
   
    def __init__(self, w0:float, wl:float, wf:float, **kwargs):
        super().__init__(**kwargs)
        if not self._argcheck(w0, wl, wf):
            raise ValueError('Incorrect parameters types or values')
        self.w0 = w0
        self.wl = wl
        self.wf = wf

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
        if not isinstance(wl, (float, int)):
            raise TypeError('wl parameter expected int or float')
        if not isinstance(w0, (float, int)):
            raise TypeError('w0 parameter expected int or float')
        if not isinstance(wf, (float, int)):
            raise TypeError('wf parameter expected int or float')
        if not (w0 < wf):
            raise ValueError('w0 must be < than wf')
        if self.a:
            if self.b and (self.a > self.b) or np.isclose(self.a, self.b):
                raise ValueError('a must be < than b')
            if np.isclose(self.a, wf):
                raise ValueError('a must be < than wf')
        return True

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

    def rvs(self, size: int=1):
        wl, w0, wf = self.wl, self.w0, self.wf
        self._argcheck(wl, w0, wf)
        limits = np.array(
            [max(w0+1e-10, self.a), min(wf-1e-10, self.b)]).astype('float64')
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


@attrs(auto_attribs=True)
class UniformSphere(stats._multivariate.multi_rv_frozen):
    center: Tuple[float,float,float] = attrib(validator=[
        validators.deep_iterable(member_validator=validators.instance_of((int, float))),
        has_len(3)],
        default=(0., 0., 0.))
    radius: float = attrib(
        validator=validators.instance_of((float, int)),
        default=1.)
    dim: float = attrib(default=3, init=False) 

    def rvs(self, size: int = 1):
        phi = stats.uniform().rvs(size) * 2 * np.pi
        cos_theta = stats.uniform(-1, 2).rvs(size)
        theta = np.arccos(cos_theta)
        r = np.cbrt(stats.uniform().rvs(size)) * self.radius
        x = r * np.sin(theta) * np.cos(phi) + self.center[0]
        y = r * np.sin(theta) * np.sin(phi) + self.center[1]
        z = r * np.cos(theta) + self.center[2]
        return np.vstack((x, y, z)).T

@attrs(auto_attribs=True)
class UniformCircle(stats._multivariate.multi_rv_frozen):
    center: Tuple[float,float] = attrib(validator=[
        validators.deep_iterable(member_validator=validators.instance_of((int, float))),
        has_len(2)],
        default=(0., 0.))
    radius: float = attrib(
        validator=validators.instance_of((float, int)),
        default=1.)
    dim: float = attrib(default=2, init=False) 

    def rvs(self, size: int=1):
        theta = stats.uniform().rvs(size=size) * 2 * np.pi
        r = self.radius * stats.uniform().rvs(size=size) ** .5
        x = r * np.cos(theta) + self.center[0]
        y = r * np.sin(theta) + self.center[1]
        return np.vstack((x, y)).T


def skewnorm(a, loc, scale, n):
    y = stats.skewnorm.rvs(a, loc, scale, n)
    df = pd.DataFrame(y)
    df.columns = ['plx']
    return df

def truncated_gumbel(lock, scale, low, high, n):
    rv_limits = (stats.gumbel_r.cdf(low, lock, scale),
                 stats.gumbel_r.cdf(high, lock, scale))
    y = np.random.rand(n)*(rv_limits[1] - rv_limits[0]) + rv_limits[0]
    return stats.gumbel_r.ppf(y, lock, scale)


# TODO: make faster: instead of line modifying sampling range, line modifies the results directly.
""" def plx_error(w0, wl, wf, n, y0=None, x0=None, plx=None):
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
        return EDSD(a=0, b=wf).rvs(wl, w0, wf, size=n) """

# Data generators
@attrs(auto_attribs=True)
class Cluster:

    space: stats._multivariate.multi_rv_frozen = attrib(
        validator=[ validators.instance_of(stats._multivariate.multi_rv_frozen), dist_has_n_dimensions(n=3) ])
    pm: stats._multivariate.multi_rv_frozen = attrib(
        validator=[validators.instance_of(stats._multivariate.multi_rv_frozen), dist_has_n_dimensions(n=2)])
    representation_type: str= attrib(
        validator=validators.in_(['cartesian', 'spherical']),
        default='spherical')
    star_count: int= attrib(
        validator=[ validators.instance_of(int), in_range(0, 'inf') ],
        default=200)

    def rvs(self):
        size = self.star_count
        data = pd.DataFrame()
        xyz = self.space.rvs(size)
        if self.representation_type == 'spherical':
            data[['ra', 'dec', 'parallax']] = pd.DataFrame(cartesian_to_polar(xyz))
        else:
            data[['x', 'y', 'z']] = pd.DataFrame(xyz)
        pm = self.pm.rvs(size)
        data[['pmra', 'pmdec']] = pd.DataFrame(pm)
        return data


@attrs(auto_attribs=True)
class Field:
    space: stats._multivariate.multi_rv_frozen = attrib(
        validator=[ validators.instance_of(stats._multivariate.multi_rv_frozen), dist_has_n_dimensions(n=2) ])
    plx: stats.rv_continuous = attrib(validator=validators.instance_of(stats.rv_continuous))
    pm: stats._multivariate.multi_rv_frozen = attrib(
        validator=[validators.instance_of(stats._multivariate.multi_rv_frozen), dist_has_n_dimensions(n=2)])
    representation_type: str= attrib(
        validator=validators.in_(['cartesian', 'spherical']),
        default='spherical')
    star_count: int= attrib(
        validator=[ validators.instance_of(int), in_range(0, 'inf') ],
        default=int(1e5))

    def rvs(self):
        size = self.star_count
        data = pd.DataFrame()
        ra_dec = self.space.rvs(size)
        pm = self.pm.rvs(size)
        data[['pmra', 'pmdec']] = pd.DataFrame(
                np.vstack((pm[:, 0], pm[:, 1])).T)
        plx = self.plx.rvs(size)
        ra_dec_plx = np.vstack((ra_dec[:,0], ra_dec[:,1], plx)).T
        if self.representation_type == 'cartesian':
            xyz = polar_to_cartesian(ra_dec_plx)
            data[['x', 'y', 'z']] = pd.DataFrame(xyz)
        else:
            data[['ra', 'dec', 'parallax']] = pd.DataFrame(ra_dec_plx)
        return data


@attrs(auto_attribs=True)
class Synthetic:
    field: Field = attrib(validator=validators.instance_of(Field))
    clusters: List[Cluster] = attrib(validator=validators.deep_iterable(
        member_validator=validators.instance_of(Cluster)
        ))

    def rvs(self):
        self.field.representation_type = 'spherical'
        field_data = self.field.rvs()
        field_data['label'] = pd.DataFrame(np.zeros(field_data.shape[0], dtype=int))
        for i in range(len(self.clusters)):
            label = i+1
            self.clusters[i].representation_type = 'spherical'
            cluster_data = self.clusters[i].rvs()
            idx = (field_data['parallax'] >= cluster_data['parallax'].min()) & (field_data['parallax'] <= cluster_data['parallax'].max())
            data_columns = ['ra', 'dec', 'pmra', 'pmdec']
            for column in data_columns:
                idx = idx & (field_data[column] >= cluster_data[column].min()) & (field_data[column] <= cluster_data[column].max())
            
            # TODO: check for every x in field[idx] if 
            # A: is_from_distribution(polar_to_cartesian(x['ra', 'dec', 'parallax']),
            #   multivariate_normal(polar_to_cartesian(cluster_center), cluster_cov))
            # and B: is_from_distribution(x['pmra', 'pmdec'], multivariate_normal(cluster_pm_center, cluster_pm_cov))
            # then eliminate x from field

            field_data.drop(field_data[idx].index, inplace=True)
            cluster_data['label'] = pd.DataFrame(np.ones(cluster_data.shape[0], dtype=int)*label)
            field_data = pd.concat([field_data, cluster_data], axis=0)

        return field_data

def is_from_dist(cdf: Callable, x, *args, **kwargs):
    return np.isclose(cdf(x, args, kwargs), np.zeros_like(x)) 
        
def is_inside_circle(center, radius, data):
    dx = np.abs(data[:,0]-center[0])
    dy = np.abs(data[:,1]-center[1])
    return (dx < radius) & (dy < radius) & ((dx + dy <= radius) | (dx**2 + dy**2 <= radius**2))

def is_inside_sphere(center, radius, data):
    dx = np.abs(data[:,0]-center[0])
    dy = np.abs(data[:,1]-center[1])
    dz = np.abs(data[:,2]-center[2])
    return (dx < radius) & (dy < radius) & ((dx + dy + dz <= radius) | (dx**2 + dy**2 + dx**2 <= radius**2))


""" center = (0,0,0)
radius = 1.
size = 1000000
data = UniformSphere(center, radius).rvs(size)
dx = np.abs(data[:,0]-center[0])
dy = np.abs(data[:,1]-center[1])
dz = np.abs(data[:,2]-center[2])
tag = np.zeros(data.shape[0])


k = radius / math.sqrt(3)
cube = data[(dx <= k) & (dy <= k) & (dz <= k)]
sx, sy, sz = cube[:,0], cube[:,1], cube[:,2]

tag[(dx <= k) & (dy <= k) & (dz <= k)] = 1
# data[np.sqrt(dx**2 + dy**2 + dx**2) > radius]

stats.kstest(sx, 'uniform', args=(sx.min(), sx.max() - sx.min())).pvalue > .05
stats.kstest(sy, 'uniform', args=(sy.min(), sy.max() - sy.min())).pvalue > .05
stats.kstest(sz, 'uniform', args=(sz.min(), sz.max() - sz.min())).pvalue > .05

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
data = pd.DataFrame({"x": data[:,0].T, "y": data[:,1].T, 'z': data[:,2].T, "label": tag})
groups = data.groupby("label")

for name, group in groups:
    if name == 0:
        ax.scatter(group['x'], group['y'], group['z'], label=name)
plt.show()
print()

rt = 'spherical'
field = Field(
    plx=EDSD(a=0, w0=1, wl=2, wf=12),
    pm=stats.multivariate_normal(mean=(0., 0.), cov=10),
    space=UniformCircle(center=(120.5, -27.5), radius=5),
    representation_type=rt,
    star_count=int(1e4)
)# .rvs()
# field_data['tag'] = pd.DataFrame(np.zeros((int(1e3),)))
 """
""" cluster = Cluster(
    pm_params={'means': (-2.4, 2), 'cov': [[.3, 0], [0, .3]]},
    space_params={'means':(121.5, -26.5, 3), 'cov':[[.8, 0, 0], [0, .8, 0], [0, 0, .8]]},
    representation_type=rt,
    star_count=200
) """# .rvs()

""" sample = Sample(field, [cluster]).rvs()

cluster_data['tag'] = pd.DataFrame(np.ones((2000,)))
data = pd.concat([field_data, cluster_data], axis=0) """
# data = cluster_data
# print('drawing graphs')
# sns.jointplot(data=data, x='parallax', y='parallax_error', hue='tag')
# sns.jointplot(data=data, x='pmra', y='pmra_error', hue='tag')
# sns.jointplot(data=data, x='pmdec', y='pmdec_error', hue='tag')
""" sns.jointplot(data=data, x='ra', y='dec', hue='tag')
sns.jointplot(data=data, x='parallax', y='ra', hue='tag') """
""" sns.jointplot(data=data, x='x', y='y', hue='tag')
sns.jointplot(data=data, x='x', y='z', hue='tag')
sns.jointplot(data=data, x='y', y='z', hue='tag') """
# plt.show()  # , marginal_kws=dict(bins = 160, fill= False))

data = load_file('scripts/data/NGC_2477/NGC_2477_data_2021-05-10_08-02-15')

# print(data)

uni = stats.uniform(0,10)
nor = stats.norm(5,1)
umix = .8
nmix = .2
scale = 1000

unidata = uni.rvs(int(umix*scale))
nordata = nor.rvs(int(nmix*scale))
data = np.concatenate([unidata, nordata])
norprob = nor.pdf(data)*nmix
uniprob = uni.pdf(data)*umix
norprob = norprob/(uniprob+norprob)
uniprob = 1. - norprob
data = np.vstack((data, uniprob, norprob)).T
print(data)