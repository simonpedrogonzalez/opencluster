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
from astropy.table.table import Table
from astropy.utils.diff import report_diff_values

from opencluster.opencluster import (
    CatalogNotFoundException,
    cone_search,
    load_VOTable,
    query_region,
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

    def test_cone_search(self):
        table = cone_search(name="ic2395", radius=0.1)
        assert isinstance(table, Table)
        assert len(table) == 50
        table2 = cone_search(name="ic2395", radius=0.01, row_limit=-1)
        assert isinstance(table2, Table)
        assert len(table2) == 52
        table3 = cone_search(
            ra=130.62916667, dec=-48.1, radius=0.1, row_limit=80
        )
        assert isinstance(table3, Table)
        assert len(table3) == 80

    def test_wrong_params_conse_search(self):
        with pytest.raises(ValueError):
            cone_search(ra=130.62916667, radius=0.1)
        with pytest.raises(ValueError):
            cone_search(name="ic2395", ra=130.62916667, radius=0.1)
        with pytest.raises(ValueError):
            cone_search(dec=130.62916667, radius=0.1)
        with pytest.raises(ValueError):
            cone_search(name=3, radius=0.1)
        with pytest.raises(ValueError):
            cone_search(name="ic2395", radius=0.1, row_limit=0.1)

    def test_loadVOTable(self):
        name = "ic2395"
        file = "./tests/" + name + ".vot"
        original_table = cone_search(name=name, radius=0.1, row_limit=80)
        cone_search(
            name=name,
            radius=0.1,
            output_file=file,
            dump_to_file=True,
            row_limit=80,
        )
        loaded_table = load_VOTable(file)
        identical = report_diff_values(original_table, loaded_table)
        assert len(loaded_table) == 80
        assert identical
        if os.path.exists(file):
            os.remove(file)

    def test_queries(self):
        table1 = (
            query_region(
                ra=130.62916667,
                dec=-48.1,
                radius=u.Quantity("30", u.arcminute),
            )
            .from_table("gaiadr2.gaia_source")
            .select(["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"])
            .where({"phot_g_mean_mag": "<15"})
            .top(55)
            .get(verbose=True)
        )
        assert isinstance(table1, Table)
        assert len(table1) == 55

        query_region(
            name="ic2395", radius=u.Quantity("30", u.arcminute)
        ).from_table("public.hipparcos", ra_name="ra", dec_name="de").select(
            "*"
        ).top(
            50
        ).get(
            verbose=True, dump_to_file=True, output_file="test.vot"
        )

        table2 = load_VOTable("test.vot")
        assert isinstance(table2, Table)
        assert len(table2) == 7
        if os.path.exists("test.vot"):
            os.remove("test.vot")

        query = (
            query_region(
                ra=130.62916667,
                dec=-48.1,
                radius=u.Quantity("30", u.arcminute),
            )
            .from_table("gaiadr2.gaia_source")
            .select(["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"])
            .where({"phot_g_mean_mag": "<15"})
            .top(55)
        )
        assert len(query.list_tables()) == 95
        assert len(query.list_columns("public.hipparcos")) == 78
        assert (
            query.table_description("public.hipparcos")
            == """table = TAP Table name: public.public.hipparcos
Description: hipparcos original catalogue (J1991.25)
Num. columns: 78"""
        )
