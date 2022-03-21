import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))

from opencluster.synthetic import *
from opencluster.detection import *
from opencluster.membership import *
from opencluster.fetcher import *
from opencluster.stat_tests import *
from sklearn.preprocessing import RobustScaler


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy import ndimage
from typing_extensions import TypedDict
from typing import Optional, Tuple, List, Union, Callable
from attr import attrib, attrs, validators
import copy
import math
from astropy.coordinates import Distance, SkyCoord
import astropy.units as u
from skimage.feature import peak_local_max
from statsmodels.robust.scale import huber, hubers_scale
from astropy.stats import biweight_location, biweight_scale, mad_std
from astropy.stats.sigma_clipping import sigma_clipped_stats
from pprint import pprint

""" desc, colnames = remote_info(default_table())
print('REMOTE TABLE INFO')
pprint(desc)
print('REMOTE TABLE COLUMNS')
pprint(colnames) """

 
test_clusters = [
    {
        'name': 'Stock 8',
        'description': 'rafael',
        'filters': { 'parallax': '> 0'},
        'area_radius': 1,
    },
    { 
        'name': 'ngc2527',
        'description': 'OK' ,
        'where': [('parallax', '>',  .2), ('phot_g_mean_mag', '<', '18')],
        # low noise_sig means "do not rely on excess noise, do not check it"
        # high noise_sig means "you should check that excess noise is small"
        'or_where': [('astrometric_excess_noise_sig', '<=', 2), ('astrometric_excess_noise', '<', 2)],
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
    # { 'name': 'ic4665', 'description': 'Ok with 2.8 plx' },
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


# data downloading
i = 1
cluster = test_clusters[i]
radius = cluster.get('area_radius')*u.deg
filters = cluster.get('filters')

# center
""" simbad_search(
        cluster.get('name'),
        dump_to_file=True,
        output_file=f'/home/simon/repos/opencluster/scripts/data/meta-{cluster.get("name")}.vot'
        ) """

""" print('counting...')
# count before download!!!
count_query_result = (
        query_region(name=cluster.get('name'), radius=radius)
        .where(filters)
        .count()
    )
# control
print(f'Query star count: {count_query_result["count_all"][0]}')
assert count_query_result['count_all'][0] < int(2e6) """

print('downloading...')
# download data and write file

query = (
        query_region(cluster.get('name'), radius)
        .select(
            'ra', 'dec', 'ra_error', 'dec_error', 'ra_dec_corr',
            'pmra', 'pmra_error', 'ra_pmra_corr', 'dec_pmra_corr',
            'pmdec', 'pmdec_error', 'ra_pmdec_corr', 'dec_pmdec_corr', 'pmra_pmdec_corr',
            'parallax', 'parallax_error', 'parallax_pmra_corr', 'parallax_pmdec_corr',  'ra_parallax_corr', 'dec_parallax_corr',
            'phot_g_mean_mag', 'bp_rp', 'astrometric_excess_noise', 'astrometric_excess_noise_sig',
            )
        .where(cluster.get('where'))
        .or_where(cluster.get('or_where'))
    )

table = query.get() 
table['log10_parallax'] = np.log10(table.to_pandas()['parallax'].to_numpy())

print('writing file...')
table.write(f'/home/simon/repos/opencluster/scripts/data/clusters_phot/{cluster.get("name")}.xml', format="votable")
print(f'{cluster.get("name")} ready')