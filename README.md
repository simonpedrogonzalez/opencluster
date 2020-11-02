# Opencluster

[![Build Status](https://travis-ci.com/simonpedrogonzalez/opencluster.svg?branch=master)](https://travis-ci.com/simonpedrogonzalez/opencluster)

load_remote: easy to use wrapper for simple radial search queries, over some of the most commonly used catalogs for coordinates, proper motions, parallax and magnitudes (gaia DR1 & DR2, hipparcos, tychos2, etc from the astroquery Gaia web service).
```python
import astropy.units as u

from opencluster import load_remote

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
Getting membership probabilities based on multiple variables (under development)
