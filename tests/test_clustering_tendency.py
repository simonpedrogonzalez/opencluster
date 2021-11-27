from opencluster.synthetic import EDSD, polar_to_cartesian, cartesian_to_polar, is_inside_circle, UniformCircle, UniformSphere, Cluster, Field, Synthetic
from scipy.stats import kstest, multivariate_normal
from opencluster.clustering_tendency import hopkins, dip
import pandas as pd
import math
import numpy as np
import pytest

class TestDetection:
    def test_hopkins():
        # increasingly clearer clusters
        return False

    def test_dip():
        # increasingly clearer multimodal distribution
        return False