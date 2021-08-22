from opencluster.synthetic import EDSD, polar_to_cartesian, cartesian_to_polar, is_inside_circle, UniformCircle, UniformSphere, Cluster, Field, Synthetic
from scipy.stats import kstest, multivariate_normal
import pandas as pd
import math
import numpy as np
import pytest

class Ok:
    pass

class TestEDSD:
    def test_EDSD_argv_check(self):
        # needs more checks!!!
        with pytest.raises(ValueError):
            EDSD(wl=1.1, w0=4.1, wf=-.15).rvs(size=100)
    
    def test_EDSD_pdf_cdf(self):
        # calculate pdf for 100 points, integrate
        # calculate cdf, compare cdf with pdf integral
        assert True

    def test_EDSD_cdf_ppf(self):
        # calculate y = cdf(x0) for 100 points
        # calculate x1 = ppf(y), compare x0 and x1 for minimal error
        assert True

    def test_EDSD_rvs(self):
        sample = EDSD(wl=1.1, w0=-.15, wf=4.1).rvs(size=100)
        assert sample.dtype == 'float64'
        assert sample.shape == (100,)
        assert sample.min() >= -.15
        assert sample.max() <= 4.1
        sample = EDSD(a=0, b=3, wl=1.1, w0=-.15, wf=4.1).rvs(size=100)
        assert sample.min() >= max(0, -.15)
        assert sample.max() <= min(3, 4.1)

class TestHelpers:
    def test_coord_transform(self):
        cartesian = np.random.uniform(low=-16204., high=16204., size=(1000, 3))
        polar = cartesian_to_polar(cartesian)
        assert np.allclose(cartesian, polar_to_cartesian(polar))

    def test_uniform_circle(self):
        center = np.random.uniform(size=2)
        radius = np.random.uniform()
        size = int(1e7)
        assert UniformCircle().dim == 2
        data = UniformCircle(center=center, radius=radius).rvs(size)
        dx = np.abs(data[:,0]-center[0])
        dy = np.abs(data[:,1]-center[1])
        assert data.shape == (size, 2)
        assert data[np.sqrt(dx**2 + dy**2) > radius].shape[0] == 0
        assert data[~is_inside_circle(center, radius, data)].shape[0] == 0
        k = radius/math.sqrt(2)
        square = data[(dx <= k) & (dy <= k)]
        sx, sy = square[:,0], square[:,1]
        assert kstest(sx, 'uniform', args=(sx.min(), sx.max() - sx.min())).pvalue > .05
        assert kstest(sy, 'uniform', args=(sy.min(), sy.max() - sy.min())).pvalue > .05

    def test_uniform_sphere(self):
        center = np.random.uniform(size=3)
        radius = np.random.uniform()
        size = int(1e5)
        assert UniformSphere().dim == 3
        data = UniformSphere(center, radius).rvs(size)
        assert data.shape == (size, 3)
        dx = np.abs(data[:,0]-center[0])
        dy = np.abs(data[:,1]-center[1])
        dz = np.abs(data[:,2]-center[2])
        assert data[np.sqrt(dx**2 + dy**2 + dz**2) > radius].shape[0] == 0
        assert data[~is_inside_circle(center, radius, data)].shape[0] == 0
        k = 2 * radius / math.sqrt(3)
        cube = data[(dx <= k) & (dy <= k) & (dz <= k)]
        sx, sy, sz = cube[:,0], cube[:,1], cube[:,2]
        assert kstest(sx, 'uniform', args=(sx.min(), sx.max() - sx.min())).pvalue > .05
        assert kstest(sy, 'uniform', args=(sy.min(), sy.max() - sy.min())).pvalue > .05
        assert kstest(sz, 'uniform', args=(sz.min(), sz.max() - sz.min())).pvalue > .05

class TestField:
    @pytest.mark.parametrize(
        'plx, space, pm, star_count, representation_type, test', [
            (EDSD(1,2,3), UniformCircle(), multivariate_normal((0,0)), 1, 'cartesian', Ok),
            (UniformCircle(), UniformCircle(), multivariate_normal((0,0)), 1, 'cartesian', TypeError),
            (EDSD(1,2,3), multivariate_normal((0,0)), multivariate_normal((0,0)), 1, 'cartesian', Ok),
            (EDSD(1,2,3), EDSD(1,2,3), multivariate_normal((0,0)), 1, 'cartesian', TypeError),
            (EDSD(1,2,3), multivariate_normal(), multivariate_normal((0,0)), 1, 'cartesian', ValueError),
            (EDSD(1,2,3), UniformCircle(), EDSD(1,2,3), 1, 'cartesian', TypeError),
            (EDSD(1,2,3), UniformCircle(), multivariate_normal(), 1, 'cartesian', ValueError),
            (EDSD(1,2,3), UniformCircle(), multivariate_normal((0,0)), 1., 'cartesian', TypeError),
            (EDSD(1,2,3), UniformCircle(), multivariate_normal((0,0)), -1, 'cartesian', ValueError),
            (EDSD(1,2,3), UniformCircle(), multivariate_normal((0,0)), 1, 'spherical', Ok),
            (EDSD(1,2,3), UniformCircle(), multivariate_normal((0,0)), 1, 'other', ValueError),
            ])
    def test_attrs(self, plx, space, pm, star_count, representation_type, test):
        if issubclass(test, Exception):
            with pytest.raises(test):
                Field(plx=plx, space=space, pm=pm, star_count=star_count, representation_type=representation_type)
        else:
            Field(plx=plx, space=space, pm=pm, star_count=star_count, representation_type=representation_type)
    
    def test_rvs(self):
        field_data = Field(
            plx=EDSD(1,2,3),
            space=UniformCircle(),
            pm=multivariate_normal((0,0)),
            star_count=int(1e5),
            representation_type='spherical'
        ).rvs()
        assert isinstance(field_data, pd.DataFrame)
        assert field_data.shape == (int(1e5), 5)
        assert sorted(list(field_data.columns)) == sorted(['ra', 'dec', 'parallax', 'pmra', 'pmdec'])
        field_data = Field(
            plx=EDSD(1,2,3),
            space=UniformCircle(),
            pm=multivariate_normal((0,0)),
            star_count=int(1e5),
            representation_type='cartesian'
        ).rvs()
        assert isinstance(field_data, pd.DataFrame)
        assert field_data.shape == (int(1e5), 5)
        assert sorted(list(field_data.columns)) == sorted(['x', 'y', 'z', 'pmra', 'pmdec'])

class TestCluster:
    @pytest.mark.parametrize(
        'space, pm, star_count, representation_type, test',
        [
           (multivariate_normal((0,0,0)), multivariate_normal((0,0)), 1, 'cartesian', Ok),
            (EDSD(1,2,3), multivariate_normal((0,0)), 1, 'cartesian', TypeError),
            (UniformCircle(), multivariate_normal((0,0)), 1, 'cartesian', ValueError),
            (multivariate_normal((0,0,0)), EDSD(1,2,3), 1, 'cartesian', TypeError),
            (multivariate_normal((0,0,0)), multivariate_normal(), 1, 'cartesian', ValueError),
            (multivariate_normal((0,0,0)), multivariate_normal((0,0)), .1, 'cartesian', TypeError),
            (multivariate_normal((0,0,0)), multivariate_normal((0,0)), -1, 'cartesian', ValueError),
            (multivariate_normal((0,0,0)), multivariate_normal((0,0)), 1, 'other', ValueError),
            (multivariate_normal((0,0,0)), multivariate_normal((0,0)), 1, 'spherical', Ok),
        ])
    def test_attrs(self, space, pm, star_count, representation_type, test):
        if issubclass(test, Exception):
            with pytest.raises(test):
                Cluster(space=space, pm=pm, star_count=star_count, representation_type=representation_type)
        else:
            Cluster(space=space, pm=pm, star_count=star_count, representation_type=representation_type)

    def test_rvs(self):
        cluster_data = Cluster(
            space=multivariate_normal((0,0,0)),
            pm=multivariate_normal((0,0)),
            star_count=100,
            representation_type='spherical'
        ).rvs()
        assert isinstance(cluster_data, pd.DataFrame)
        assert cluster_data.shape == (100, 5)
        assert sorted(list(cluster_data.columns)) == sorted(['ra', 'dec', 'parallax', 'pmra', 'pmdec'])
        cluster_data = Cluster(
            space=multivariate_normal((0,0,0)),
            pm=multivariate_normal((0,0)),
            star_count=100,
            representation_type='cartesian'
        ).rvs()
        assert isinstance(cluster_data, pd.DataFrame)
        assert cluster_data.shape == (100, 5)
        assert sorted(list(cluster_data.columns)) == sorted(['x', 'y', 'z', 'pmra', 'pmdec'])

class TestSynthetic:
    def test_attrs(self):
        assert False

    def test_rvs(self):
        assert False

class TestCrop:
    def test_draw_3Dcontour(self):
        # generate gaussian 3d data
        # draw shape
        # assert data in sphere
        # assert at least one point in ellipse border
        assert True
    
    def test_draw_2Dcontour(self):
        # generate gaussian 2d data
        # draw shape (ellipse)
        # assert data in ellipse
        # assert at least one point in ellipse border
        assert True
