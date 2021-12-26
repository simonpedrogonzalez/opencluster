import numpy as np
import itertools
from attr import attrs
from typing import List


@attrs(auto_attribs=True, init=False)
class Colnames:

    var: list
    var_corr: list
    var_err: list
    var_err_corr: list
    err: list
    corr: list

    def __init__(self, colnames: List[str]):
        errors = [c for c in colnames if c.endswith('_error')]
        correlations = [c for c in colnames if c.endswith('_corr')]
        variables = [c for c in colnames if c not in errors and c not in correlations]
        self.var = variables
        self.err = sorted_err(variables, errors)
        var_err = []
        for var in variables:
            for err in errors:
                if var in err:
                    var_err.append(var)
                    break
        self.var_err = var_err
        
        vc = []
        for var in variables:
            for corr in correlations:
                if var in corr:
                    vc.append(var)
                    break
        vc = list(dict.fromkeys(vc))
        self.var_corr = vc
        self.corr = sorted_corr(vc, correlations)

        vce = []
        for var in vc:
            for err in self.err:
                if var in err:
                    vce.append(var)
                    break
        self.var_err_corr = vce

def sorted_corr(variables: list, correlations: list):
    vc = variables
    vc_count = len(variables)
    corr_matrix = np.ndarray(shape=(vc_count, vc_count), dtype=f'|S{max([len(c) for c in variables + correlations])}')
    for i1, var1 in enumerate(vc):
        for i2, var2 in enumerate(vc):
            corr1 = f'{var1}_{var2}_corr'
            corr2 = f'{var2}_{var1}_corr'
            corr = corr1 if corr1 in correlations else corr2 if corr2 in correlations else ''
            corr_matrix[i1, i2] = corr
    return list(corr_matrix[np.tril_indices(vc_count, k=-1)].astype(str))

def sorted_err(variables: list, errors: list):
    ordered_errors = []
    for var in variables:
        for err in errors:
            if err.startswith(var):
                ordered_errors.append(err)
                errors.remove(err)
                break
    return ordered_errors

def subset(data: np.ndarray, limits: list):
    for i in range(len(limits)):
        data = data[(data[:,i] > limits[i][0]) & (data[:,i] < limits[i][1])]
    return data

def combinations(items: list):
    return list(itertools.product(*items))

def dict_combinations(items: list):
    "items is a list of dicts"
    return combinations([[{ k: v } for (k, v) in d.items()] for d in items])

def get_colnames(colnames: list):
    error_c = [c for c in colnames if c.endswith('_error')]

    return [f'{var_name}_error' for var_name in var_colnames]

""" 
c = Colnames([
                'ra', 'dec', 'ra_error', 'dec_error', 'ra_dec_corr',
                'pmra', 'pmra_error', 'ra_pmra_corr', 'dec_pmra_corr',
                'pmdec', 'pmdec_error', 'ra_pmdec_corr', 'dec_pmdec_corr', 'pmra_pmdec_corr',
                'parallax', 'parallax_error', 'parallax_pmra_corr', 'parallax_pmdec_corr', 'ra_parallax_corr', 'dec_parallax_corr',
                'phot_g_mean_mag'
            ])

print('coso') """
