import sys
import os
sys.path.append(os.path.join(os.path.dirname('opencluster'), '.'))

from opencluster.synthetic import *
from opencluster.detection import *
from opencluster.membership import *
from opencluster.fetcher import *

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
pprint(colnames)
 """
 
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
""" 
for i in range(1):
    cluster = test_clusters[i]
    radius = cluster.get('area_radius')*u.deg
    filters = cluster.get('filters')

    # center
    simbad_search(
        cluster.get('name'),
        dump_to_file=True,
        output_file=f'/home/simon/repos/opencluster/scripts/data/meta-{cluster.get("name")}.vot'
        )

    print('counting...')
    # count before download!!!
    count_query_result = (
            query_region(name=cluster.get('name'), radius=radius)
            .where(filters)
            .count()
        )
    # control
    print(f'Query star count: {count_query_result["count_all"][0]}')
    assert count_query_result['count_all'][0] < int(2e6)

    print('downloading...')
    # download data and write file
    table = OCTable((
            query_region(name=cluster.get('name'), radius=radius)
            .select([
                'ra', 'dec',
                'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
                'parallax', 'parallax_error',
                'phot_g_mean_mag'
            ])
            .where(filters)
            .get()
        ))
    table.table['log10_parallax'] = np.log10(table.to_pandas()['parallax'].to_numpy())

    print('writing file...')
    table.write_to(f'/home/simon/repos/opencluster/scripts/data/{cluster.get("name")}.vot')
    print(f'{cluster.get("name")} ready')
 """
i = 0
cluster = test_clusters[i]

print('reading file')
data = load_file(f'/home/simon/repos/opencluster/scripts/data/{cluster.get("name")}.vot').to_pandas()

detection_data = data[['pmra', 'pmdec', 'parallax']].to_numpy()
detection_data[:,2] = np.log10(detection_data[:,2])
bin_shape = [.5, .5, .05]

print('detecting')


result = find_clusters(
    data=detection_data,
    bin_shape=bin_shape,
    mask=default_mask(3)
)

coords = []
data['log_parallax'] = np.log10(data['parallax'].to_numpy())
check_data = data[['pmra', 'pmdec', 'log_parallax', 'ra', 'dec']].to_numpy()
for peak in res.peaks:
    limits = np.vstack((peak.center-peak.sigma/2, peak.center+peak.sigma/2)).T
    s = subset(check_data, limits)
    coords.append((np.median(s[:,3]), np.median(s[:,4])))

print('Sobredensidades detectadas en coordenadas:')
pprint(res.peaks)
pprint(coords)