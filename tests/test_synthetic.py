from opencluster.synthetic import EDSD
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

class TestCrop5DSphere:
    def test_corp_5d_sphere_params_check(self):
        return True
