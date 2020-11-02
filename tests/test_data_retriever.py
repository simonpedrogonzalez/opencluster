# <Opencluster, a package for open star cluster probabilities calculations>
# Copyright (C) 2020  González Simón Pedro

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.utils.diff import report_diff_values

from opencluster.opencluster import (
    CatalogNotFoundException,
    list_remotes,
    load_file,
    load_remote,
    remote_info,
    simbad_search,
)

import pytest


class TestDataRetriever:
    def test_correct_simbad_search(self):
        # check exact values
        assert simbad_search("ic2395").to_string("hmsdms") == SkyCoord(
            ra=130.62916667, dec=-48.1, frame="icrs", unit="deg"
        ).to_string("hmsdms")

    def test_non_existent_catalog_simbad_search(self):
        with pytest.raises(CatalogNotFoundException):
            simbad_search("non existent table")

    def test_load(self):
        name = "ic2395"
        file = name + ".vot"
        original_table = load_remote(
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
            limit=55,
            columns=["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"],
            filters={"phot_g_mean_mag": "<12"},
            dump_to_file=True,
            output_file=name + ".vot",
        )
        loaded_table = load_file(file)
        identical = report_diff_values(original_table, loaded_table)
        assert identical
        if os.path.exists(file):
            os.remove(file)

    def test_info(self):
        desc, columns = remote_info("public.hipparcos")
        assert len(list_remotes()) == 95
        assert len(columns) == 78
        assert (
            desc
            == """table = TAP Table name: public.public.hipparcos
Description: hipparcos original catalogue (J1991.25)
Num. columns: 78"""
        )
