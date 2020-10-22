# Opencluster

[![Build Status](https://travis-ci.com/simonpedrogonzalez/opencluster.svg?branch=master)](https://travis-ci.com/simonpedrogonzalez/opencluster)

query_region: easy to use wrapper for simple radial search queries, over some of the most commonly used catalogs for coordinates, proper motions, parallax and magnitudes (gaia DR1 & DR2, hipparcos, tychos2, etc from the astroquery Gaia web service). Examples:
```
from opencluster import region
import astropy.units as u

votable = region(ra=130.62916667, dec=-48.1, radius=u.Quantity("30", u.arcminute))
.select(["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"])
.where({"phot_g_mean_mag": "<15"})
.top(55)
.get()

region(name="ic2395", radius=u.Quantity("30", u.arcminute))
.select("*")
.from_table("public.hipparcos", ra_name="ra", dec_name="de")
.top(50)
.get(dump_to_file=True, output_file="test.vot")
```
