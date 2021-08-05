from opencluster.synthetic import EDSD, polar_to_cartesian, cartesian_to_polar
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

class TestCoordTransform:
    def test_coord_transform(self):
        cartesian = np.random.uniform(low=-16204., high=16204., size=(1000, 3))
        polar = cartesian_to_polar(cartesian)
        assert np.allclose(cartesian, polar_to_cartesian(polar))

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
