from opencluster.synthetic import EDSD, polar_to_cartesian, cartesian_to_polar, is_inside_circle, UniformCircle, UniformSphere, Cluster, Field, Synthetic
from scipy.stats import kstest, multivariate_normal
from opencluster.detection import histogram, default_mask, convolve, count_based_outlier_removal, nyquist_offsets, fast_std_filter, Peak
import pandas as pd
import math
import numpy as np
import pytest

class TestDetection:

    @pytest.mark.parametrize(
        'data, bin_shape',
        [
           (multivariate_normal((0,0,0)), multivariate_normal((0,0)), 1, 'cartesian', Ok),
        ])
    def test_histogram():
        # caso muy pequeño
        # caso feliz
        # histogram(data, bin_shape: list)
        return False

    def test_default_mask():
        return False

    def test_convolve():
        # test with mask and test with function
        return False

    def test_count_based_outlier_removal():
        # test happy case and test all bins removed
        # test small case: its limits enclose less than one bin, but shoud always return at least 1 bin
        return False

    def test_nyquist_offsets():
        # random bin_shape
        return False
    
    def test_fast_std_filter():
        # test is equal than normal
        return False
    
    def test_fast_var_filter():
        # test is equal than normal
        return False

    def test_peak_is_in_neighbourhood():
        # case that it is
        # case that it is not
        # case is near border
        return False

    def test_best_peaks():
        # test equal peaks
        # test none peaks
        # test happy case
        return False

    def test_detect():
        # test parameter check
        # test uniform case (no peaks espected)
        # test no thresholds provided set in None ()
        # test thresholds provided happy case
        return False