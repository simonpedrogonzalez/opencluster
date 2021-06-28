# Opencluster

[![Build Status](https://travis-ci.com/simonpedrogonzalez/opencluster.svg?branch=develop)](https://travis-ci.com/simonpedrogonzalez/opencluster)

load_remote: easy to use wrapper for simple radial search queries, over some of the most commonly used catalogs for coordinates, proper motions, parallax and magnitudes (gaia DR1 & DR2, hipparcos, tychos2, etc from the astroquery Gaia web service).
```python
import astropy.units as u

from opencluster.fetcher import load_remote

octable = load_remote(
            table="gaiadr2.gaia_source",
            name="ic2395",
            radius=u.Quantity("30", u.arcminute),
            limit=55,
            columns=["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"],
            filters={"phot_g_mean_mag": "<12"},
        )

load_remote(
            name="ic2395",
            radius=u.Quantity("30", u.arcminute),
            columns=["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"],
            filters={"phot_g_mean_mag": "<12"},
            dump_to_file=True,
            output_file="ic2395.vot",
        )
```
load_file: load an OCTable from VOTable file
```python
loaded_table = load_file(file)
```
Fitting parallax distribution, based on a logarithmic and negative exponential functions to model the field, and a normal function to model the cluster. Initial parameters must be introduced, by observing the distribution. Parameters are (complete this section)

```python
from opencluster import load_file

octable = load_file("ic2395.vot")

octable.densities_plot()

result = octable.fit_plx(
    lower_lim=0,
    upper_lim=2,
    bins=30,
    initial_params=[4.0, 0.1, 0.1, 1.0, 0.07, 1.3, 0.1],
)
result.get("fit")
result.get("params")
```

