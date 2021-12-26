
import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))


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

# Helper functions
def is_inside_circle(center, radius, data):
    dx = np.abs(data[:,0]-center[0])
    dy = np.abs(data[:,1]-center[1])
    return (dx < radius) & (dy < radius) & ((dx + dy <= radius) | (dx**2 + dy**2 <= radius**2))

def is_inside_sphere(center, radius, data):
    dx = np.abs(data[:,0]-center[0])
    dy = np.abs(data[:,1]-center[1])
    dz = np.abs(data[:,2]-center[2])
    return dx**2 + dy**2 + dz**2 <= radius**2

# Coordinate transformation
# TODO: change to take three arrays or numbers as input
def cartesian_to_polar(coords):
    coords = np.array(coords)
    if len(coords.shape) == 1:
        coords = SkyCoord(
            x=coords[0], y=coords[1], z=coords[2],
            unit='pc', representation_type='cartesian', frame='icrs'
        )
        coords.representation_type = 'spherical'
        return np.array([coords.ra.deg, coords.dec.deg, coords.distance.parallax.mas])
    else:
        coords = SkyCoord(
            x=coords[:,0], y=coords[:,1], z=coords[:,2],
            unit='pc', representation_type='cartesian', frame='icrs'
        )
        coords.representation_type = 'spherical'
        return np.vstack((coords.ra.deg, coords.dec.deg, coords.distance.parallax.mas)).T

# TODO: change to take three arrays or numbers as input
def polar_to_cartesian(coords):
    coords = np.array(coords)
    if len(coords.shape) == 1:
        coords = SkyCoord(
            ra=coords[0]*u.degree, dec=coords[1]*u.degree,
            distance=Distance(parallax=coords[2]*u.mas),
            representation_type='spherical', frame='icrs'
        )
        coords.representation_type = 'cartesian'
        return np.array([coords.x.value, coords.y.value, coords.z.value])
    else:
        coords = SkyCoord(
            ra=coords[:,0]*u.degree, dec=coords[:,1]*u.degree,
            distance=Distance(parallax=coords[:,2]*u.mas),
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

    # TODO: test
    def pdf(self, x: list):
        is_inside = is_inside_sphere(self.center, self.radius, x)
        res = np.array(is_inside, dtype=float)
        res[res > 0] = 1./(4./3.*np.pi*self.radius**3)
        return res


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

    # TODO: fails when n is 1
    def rvs(self):
        size = self.star_count
        data = pd.DataFrame()
        xyz = self.space.rvs(size)
        if self.representation_type == 'spherical':
            data['ra'], data['dec'], data['parallax'] = cartesian_to_polar(xyz).T
        else:
            data[['x', 'y', 'z']] = pd.DataFrame(xyz)
        pm = self.pm.rvs(size)
        data[['pmra', 'pmdec']] = pd.DataFrame(pm)
        return data

    # test
    def pdf(self, data):
        pm_pdf = self.pm.pdf(data[['pmra', 'pmdec']].to_numpy())
        if set(['x', 'y', 'z']).issubset(set(data.columns)):
            space_pdf = self.space.pdf(data[['x' ,'y', 'z']].to_numpy())
        else:
            xyz = polar_to_cartesian(data['ra', 'dec', 'parallax'].to_numpy())
            space_pdf = self.space.pdf(xyz)
        return pm_pdf*space_pdf


@attrs(auto_attribs=True)
class Field:
    space: stats._multivariate.multi_rv_frozen = attrib(
        validator=[ validators.instance_of(stats._multivariate.multi_rv_frozen), dist_has_n_dimensions(n=3)])
    pm: stats._multivariate.multi_rv_frozen = attrib(
        validator=[validators.instance_of(stats._multivariate.multi_rv_frozen), dist_has_n_dimensions(n=2)])
    representation_type: str= attrib(
        validator=validators.in_(['cartesian', 'spherical']),
        default='spherical')
    star_count: int= attrib(
        validator=[ validators.instance_of(int), in_range(0, 'inf') ],
        default=int(1e5))

    # TODO: test
    def rvs(self):
        size = self.star_count
        data = pd.DataFrame()
        xyz = self.space.rvs(size)
        pm = self.pm.rvs(size)
        data[['pmra', 'pmdec']] = pd.DataFrame(
                np.vstack((pm[:, 0], pm[:, 1])).T)
        if self.representation_type == 'spherical':
            ra_dec_plx = cartesian_to_polar(xyz)
            data[['ra', 'dec', 'parallax']] = pd.DataFrame(ra_dec_plx)
        else:
            data[['x', 'y', 'z']] = pd.DataFrame(xyz)
        return data

    # TODO: test
    def pdf(self, data):
        pm_pdf = self.pm.pdf(data[['pmra', 'pmdec']].to_numpy())
        if set(['x', 'y', 'z']).issubset(set(data.columns)):
            space_pdf = self.space.pdf(data[['x' ,'y', 'z']].to_numpy())
        else:
            xyz = polar_to_cartesian(data['ra', 'dec', 'parallax'].to_numpy())
            space_pdf = self.space.pdf(xyz)
        return pm_pdf*space_pdf


@attrs(auto_attribs=True)
class Synthetic:
    field: Field = attrib(validator=validators.instance_of(Field))
    clusters: List[Cluster] = attrib(validator=validators.deep_iterable(
        member_validator=validators.instance_of(Cluster)
        ))
    representation_type: str= attrib(
        validator=validators.in_(['cartesian', 'spherical']),
        default='spherical')

    # TODO: test
    def rvs(self):
        self.field.representation_type = 'cartesian'
        data = self.field.rvs()
        for i in range(len(self.clusters)):
            self.clusters[i].representation_type = 'cartesian'
            cluster_data = self.clusters[i].rvs()
            data = pd.concat([data, cluster_data], axis=0)
        
        # TODO: improve
        total_stars = sum([c.star_count for c in self.clusters]) + self.field.star_count
        field_mixing_ratio = float(self.field.star_count)/float(total_stars)
        field_p = self.field.pdf(data)*field_mixing_ratio
        clusters_mixing_ratios = [float(c.star_count)/float(total_stars) for c in self.clusters]
        cluster_ps = np.array([self.clusters[i].pdf(data)*clusters_mixing_ratios[i] for i in range(len(self.clusters))])
        total_p = cluster_ps.sum(axis=0) + field_p
        total_clusters_probs = 0
        for i in range(len(self.clusters)):
            data[f'p_cluster{i+1}'] = cluster_ps[i]/total_p
            total_clusters_probs += cluster_ps[i]/total_p
        data['p_field'] = 1 - total_clusters_probs
        if self.representation_type == 'spherical':
            xyz = data[['x', 'y', 'z']].to_numpy()
            data['ra'], data['dec'], data['parallax'] = cartesian_to_polar(xyz).T
            data['log10_parallax'] = np.log10(data['parallax'])
        return data


""" rt = 'spherical'
field = Field(
    pm=stats.multivariate_normal(mean=(0., 0.), cov=3),
    # space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=700),
    space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=3),
    representation_type=rt,
    star_count=int(60)
)
cluster = Cluster(
    space=stats.multivariate_normal(
        mean=polar_to_cartesian([120.7, -28.5, 5]),
        cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ),
    pm=stats.multivariate_normal(mean=(.5, .5), cov=.5),
    representation_type=rt,
    star_count=40
)
synthetic = Synthetic(field=field, clusters=[cluster])
data = synthetic.rvs()
data[['ra', 'dec', 'parallax']] = cartesian_to_polar(data[['x', 'y', 'z']].to_numpy())
sns.scatterplot(data=data, x='ra', y='dec', hue='p_cluster1')
plt.show() """