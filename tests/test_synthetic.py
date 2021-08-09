from opencluster.synthetic import EDSD, polar_to_cartesian, cartesian_to_polar, is_inside_circle, cone_region, Cluster, Field, Sample
from scipy.stats import kstest
import pandas as pd
import math
import numpy as np
import pytest


class TestEDSD:
    def test_EDSD_argv_check(self):
        with pytest.raises(ValueError):
            EDSD().rvs(wl=1.1, w0=4.1, wf=-.15, size=100)
    
    def test_EDSD_pdf_cdf(self):
        # calculate pdf for 100 points, integrate
        # calculate cdf, compare cdf with pdf integral
        assert False

    def test_EDSD_cdf_ppf(self):
        # calculate y = cdf(x0) for 100 points
        # calculate x1 = ppf(y), compare x0 and x1 for minimal error
        assert False

    def test_EDSD_rvs(self):
        sample = EDSD().rvs(wl=1.1, w0=-.15, wf=4.1, size=100)
        assert sample.dtype == 'float64'
        assert sample.shape == (100,)
        assert sample.min() >= -.15
        assert sample.max() <= 4.1
        sample = EDSD(a=0, b=3).rvs(wl=1.1, w0=-.15, wf=4.1, size=100)
        assert sample.min() >= max(0, -.15)
        assert sample.max() <= min(3, 4.1)

class TestHelpers:
    def test_coord_transform(self):
        cartesian = np.random.uniform(low=-16204., high=16204., size=(1000, 3))
        polar = cartesian_to_polar(cartesian)
        assert np.allclose(cartesian, polar_to_cartesian(polar))

    def test_cone_region(self):
        center = np.random.uniform(size=2)
        radius = np.random.uniform()
        size = int(1e5)
        data = cone_region(center, radius, size)
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

    def test_is_in_dist(self):
        center = np.random.uniform(size=2)
        radius = np.random.uniform()
        size = int(1e5)
        data = cone_region(center, radius, size)

class Field:
    @pytest.mark.parametrize(
        'plx, space, pm, star_count, representation_type, test',
        [(UniformCircle(), UniformCircle(), Norm2D(), 1, 'cartesian', TypeError),
        (EDSD(), Norm2D(), Norm2D(), 1, 'spherical', TypeError),
        (EDSD(), UniformCircle(), UniformCircle(), 1, 'cartesian', TypeError),
        (EDSD(), UniformCircle(), Norm2D(), 1., 'cartesian', TypeError),
        (EDSD(), UniformCircle(), Norm2D(), -1, 'cartesian', ValueError),
        (EDSD(), UniformCircle(), Norm2D(), 1, 'other', ValueError),
        (EDSD(), UniformCircle(), Norm2D(), int(5e6), 'cartesian', 'ok'),
        (EDSD(), UniformCircle(), Norm2D(), 200, 'spherical', 'ok'),
        ])
    def test_attrs(self, plx, space, pm, star_count, representation_type, test):
        if issubclass(test, Exception):
            with pytest.raises(test):
                Field(plx=plx, space=space, pm=pm, star_count=star_count, representation_type=representation_type)
        else:
            field = Field(plx=plx, space=space, pm=pm, star_count=star_count, representation_type=representation_type)
    
    def test_rvs(self):
        field_data = Field(
            plx=EDSD(),
            space=UniformCircle(),
            pm=Norm2D(),
            star_count=int(1e5),
            representation_type='spherical'
        ).rvs()
        assert isinstance(field_data, pd.DataFrame)
        assert field_data.shape == (int(1e5), 5)
        assert sorted(list(field_data.columns)) == sorted(['ra', 'dec', 'parallax', 'pmra', 'pmdec'])
        field_data = Field(
            plx=EDSD(),
            space=UniformCircle(),
            pm=Norm2D(),
            star_count=int(1e5),
            representation_type='cartesian'
        ).rvs()
        assert isinstance(field_data, pd.DataFrame)
        assert field_data.shape == (int(1e5), 5)
        assert sorted(list(field_data.columns)) == sorted(['ra', 'dec', 'x', 'y', 'z'])


class TestCrop:
    def test_draw_3Dcontour(self):
        # generate gaussian 3d data
        # draw shape
        # assert data in sphere
        # assert at least one point in ellipse border
        return True
    
    def test_draw_2Dcontour(self):
        # generate gaussian 2d data
        # draw shape (ellipse)
        # assert data in ellipse
        # assert at least one point in ellipse border
        return False
