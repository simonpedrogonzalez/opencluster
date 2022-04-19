import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))

from opencluster.synthetic import *
from opencluster.detection import *
from opencluster.membership import *
from opencluster.fetcher import *
from opencluster.stat_tests import *
from opencluster.pipeline import *

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from astropy.coordinates import Distance, SkyCoord
import astropy.units as u
from pprint import pprint

test_clusters = [
    {
        'name': 'Stock 8',
        'description': 'rafael',
        'filters': { 'parallax': '> 0'},
        'area_radius': 1,
    },
    { 
        'name': 'ngc2527',
        'description': 'OK. 1m stars' ,
        'filters': { 'parallax': '> 0.2'},
        'area_radius': 2.5,
    },
    { 
        'name': 'ic2395',
        'description': 'interesting structures',
        'filters': {
            'parallax': '> 0',
            'phot_g_mean_mag': '< 17'
            },
        'area_radius': 4
    },
    { 
        'name': 'ngc2323',
        'description': 'OK',
        'filters': {
            'parallax': '> 0',
            'phot_g_mean_mag': '< 18'
            },
        'area_radius': 0.7
    },
    { 
        'name': 'ic2602',
        'description': 'small star count',
        'filters': {
            'parallax': '> 0',
            'phot_g_mean_mag': '< 18'
        },
        'area_radius': 0.7
    },
    { 
        'name': 'ngc2168',
        'description': 'OK',
        'filters': {
            'parallax': '> 0',
            'phot_g_mean_mag': '< 18'
        },
        'area_radius': 0.5
    }
]

sample_number = 1
sample = test_clusters[sample_number]
base_path = "/home/simon/repos/opencluster/scripts/data/clusters/"
file_path = base_path + f"{sample.get('name')}.vot"

print('reading file')
table = load_file(file_path)
data = table.to_pandas()

result = PMPlxPipeline().process(data)
p = result.p

_, n_clus = p.shape
data_to_plot = data[['pmra', 'pmdec', 'log10_parallax']].to_numpy()

for n in range(n_clus):
    """ sns.scatterplot(data.ra, data.dec, hue=p[:, n], size=p[:, n], hue_norm=(0, 1)).set(
        title=f"p(x∈C{n}) ra-dec"
    )
    plt.figure() """
    sns.scatterplot(data.pmra, data.pmdec, hue=p[:, n], hue_norm=(0, 1), size=p[:, n]).set(
        title=f"p(x∈C{n}) pm"
    )
    plt.figure()
    """ sns.scatterplot(
        data.pmra, data.parallax, hue=p[:, n], hue_norm=(0, 1), size=p[:, n], 
    ).set(title=f"p(x∈C{n}) pmra-plx") """

print('coso')
    