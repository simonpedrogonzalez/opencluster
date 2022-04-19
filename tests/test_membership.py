from opencluster.membership2 import DensityBasedMembershipEstimator
import pytest
import numpy as np

from opencluster.utils import Colnames2

df = Table.read("tests/data/ngc2527_small.xml").to_pandas()
cnames = Colnames2(df.columns.to_list())
datanames = cnames.get_data_names(["pmra", "pmdec", "parallax"])
errornames, _ = cnames.get_error_names()
corrnames, _ = cnames.get_corr_names()
data = df[datanames].to_numpy()
err = df[errornames].to_numpy()
corr = df[corrnames].to_numpy()
n, d = data.shape
w = np.ones(n)

def test_fit_predict():
    dbme = DensityBasedMembershipEstimator()
    result = dbme.fit_predict(data, err, corr)
    assert 0