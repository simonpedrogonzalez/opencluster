from opencluster.synthetic import EDSD, polar_to_cartesian, cartesian_to_polar, is_inside_circle, UniformCircle, UniformSphere, Cluster, Field, Synthetic
from scipy.stats import kstest, multivariate_normal
from opencluster.stat_tests import HopkinsTest, DipTest
import pandas as pd
import math
import numpy as np
import pytest

class TestDetection:
    def test_hopkins():
        # increasingly clearer clusters
        return True

    def test_dip():
        # increasingly clearer multimodal distribution
        return True