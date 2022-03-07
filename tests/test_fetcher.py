"""Test data fetcher module"""
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
# pylint: disable=missing-docstring

import os
import re
from io import BytesIO
from itertools import chain

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.table.table import Table
from astropy.utils.diff import report_diff_values
from astroquery.utils.tap.model.job import Job
from astroquery.utils.tap.model.tapcolumn import TapColumn
from astroquery.utils.tap.model.taptable import TapTableMeta

from opencluster import fetcher
from opencluster.fetcher import (
    SimbadResult,
    query_region,
    simbad_search,
    table_info,
)


# TODO: change for utils test_if_raises_exception
class Ok:
    pass


def verify_result(test, func):
    if issubclass(test, Exception):
        with pytest.raises(test):
            func()
    else:
        func()


def simbad_query_object():
    content = b'<?xml version="1.0" encoding="utf-8"?>\n<!-- Produced with astropy.io.votable version 4.3.1\n     http://www.astropy.org/ -->\n<VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.ivoa.net/xml/VOTable/v1.3 http://www.ivoa.net/xml/VOTable/VOTable-1.4.xsd">\n <RESOURCE type="results">\n  <TABLE ID="SimbadScript" name="default">\n   <DESCRIPTION>\n    Simbad script executed on 2021.10.17CEST02:20:07\n   </DESCRIPTION>\n   <FIELD ID="MAIN_ID" arraysize="*" datatype="char" name="MAIN_ID" ucd="meta.id;meta.main" width="22">\n    <DESCRIPTION>\n     Main identifier for an object\n    </DESCRIPTION>\n    <LINK href="http://simbad.u-strasbg.fr/simbad/sim-id?Ident=${MAIN_ID}&amp;NbIdent=1" value="${MAIN_ID}"/>\n   </FIELD>\n   <FIELD ID="RA" arraysize="13" datatype="char" name="RA" precision="8" ucd="pos.eq.ra;meta.main" unit="&quot;h:m:s&quot;" width="13">\n    <DESCRIPTION>\n     Right ascension\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="DEC" arraysize="13" datatype="char" name="DEC" precision="8" ucd="pos.eq.dec;meta.main" unit="&quot;d:m:s&quot;" width="13">\n    <DESCRIPTION>\n     Declination\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="RA_PREC" datatype="short" name="RA_PREC" width="2">\n    <DESCRIPTION>\n     Right ascension precision\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="DEC_PREC" datatype="short" name="DEC_PREC" width="2">\n    <DESCRIPTION>\n     Declination precision\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="COO_ERR_MAJA" datatype="float" name="COO_ERR_MAJA" precision="3" ucd="phys.angSize.smajAxis;pos.errorEllipse;pos.eq" unit="mas" width="6">\n    <DESCRIPTION>\n     Coordinate error major axis\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="COO_ERR_MINA" datatype="float" name="COO_ERR_MINA" precision="3" ucd="phys.angSize.sminAxis;pos.errorEllipse;pos.eq" unit="mas" width="6">\n    <DESCRIPTION>\n     Coordinate error minor axis\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="COO_ERR_ANGLE" datatype="short" name="COO_ERR_ANGLE" ucd="pos.posAng;pos.errorEllipse;pos.eq" unit="deg" width="3">\n    <DESCRIPTION>\n     Coordinate error angle\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="COO_QUAL" arraysize="1" datatype="char" name="COO_QUAL" ucd="meta.code.qual;pos.eq" width="1">\n    <DESCRIPTION>\n     Coordinate quality\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="COO_WAVELENGTH" arraysize="1" datatype="char" name="COO_WAVELENGTH" ucd="instr.bandpass;pos.eq" width="1">\n    <DESCRIPTION>\n     Wavelength class for the origin of the coordinates (R,I,V,U,X,G)\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="COO_BIBCODE" arraysize="*" datatype="char" name="COO_BIBCODE" ucd="meta.bib.bibcode;pos.eq" width="19">\n    <DESCRIPTION>\n     Coordinate reference\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="RA_2" arraysize="13" datatype="char" name="RA_2" precision="8" ucd="pos.eq.ra;meta.main" unit="&quot;h:m:s&quot;" width="13">\n    <DESCRIPTION>\n     Right ascension\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="DEC_2" arraysize="13" datatype="char" name="DEC_2" precision="8" ucd="pos.eq.dec;meta.main" unit="&quot;d:m:s&quot;" width="13">\n    <DESCRIPTION>\n     Declination\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="RA_PREC_2" datatype="short" name="RA_PREC_2" width="2">\n    <DESCRIPTION>\n     Right ascension precision\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="DEC_PREC_2" datatype="short" name="DEC_PREC_2" width="2">\n    <DESCRIPTION>\n     Declination precision\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="COO_ERR_MAJA_2" datatype="float" name="COO_ERR_MAJA_2" precision="3" ucd="phys.angSize.smajAxis;pos.errorEllipse;pos.eq" unit="mas" width="6">\n    <DESCRIPTION>\n     Coordinate error major axis\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="COO_ERR_MINA_2" datatype="float" name="COO_ERR_MINA_2" precision="3" ucd="phys.angSize.sminAxis;pos.errorEllipse;pos.eq" unit="mas" width="6">\n    <DESCRIPTION>\n     Coordinate error minor axis\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="COO_ERR_ANGLE_2" datatype="short" name="COO_ERR_ANGLE_2" ucd="pos.posAng;pos.errorEllipse;pos.eq" unit="deg" width="3">\n    <DESCRIPTION>\n     Coordinate error angle\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="COO_QUAL_2" arraysize="1" datatype="char" name="COO_QUAL_2" ucd="meta.code.qual;pos.eq" width="1">\n    <DESCRIPTION>\n     Coordinate quality\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="COO_WAVELENGTH_2" arraysize="1" datatype="char" name="COO_WAVELENGTH_2" ucd="instr.bandpass;pos.eq" width="1">\n    <DESCRIPTION>\n     Wavelength class for the origin of the coordinates (R,I,V,U,X,G)\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="COO_BIBCODE_2" arraysize="*" datatype="char" name="COO_BIBCODE_2" ucd="meta.bib.bibcode;pos.eq" width="19">\n    <DESCRIPTION>\n     Coordinate reference\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PLX_VALUE" datatype="double" name="PLX_VALUE" precision="3" ucd="pos.parallax.trig" unit="mas" width="9">\n    <DESCRIPTION>\n     Parallax\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PLX_PREC" datatype="short" name="PLX_PREC" width="1">\n    <DESCRIPTION>\n     Parallax precision\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PLX_ERROR" datatype="float" name="PLX_ERROR" ucd="stat.error;pos.parallax.trig" unit="mas">\n    <DESCRIPTION>\n     Parallax error\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PLX_QUAL" arraysize="1" datatype="char" name="PLX_QUAL" ucd="meta.code.qual;pos.parallax.trig" width="1">\n    <DESCRIPTION>\n     Parallax quality\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PLX_BIBCODE" arraysize="*" datatype="char" name="PLX_BIBCODE" ucd="meta.bib.bibcode;pos.parallax.trig" width="19">\n    <DESCRIPTION>\n     Parallax reference\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PMRA" datatype="double" name="PMRA" precision="3" ucd="pos.pm;pos.eq.ra" unit="mas.yr-1" width="9">\n    <DESCRIPTION>\n     Proper motion in RA\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PMDEC" datatype="double" name="PMDEC" precision="3" ucd="pos.pm;pos.eq.dec" unit="mas.yr-1" width="9">\n    <DESCRIPTION>\n     Proper motion in DEC\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PMRA_PREC" datatype="short" name="PMRA_PREC" width="2">\n    <DESCRIPTION>\n     Proper motion in RA precision\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PMDEC_PREC" datatype="short" name="PMDEC_PREC" width="2">\n    <DESCRIPTION>\n     Proper motion in DEC precision\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PM_ERR_MAJA" datatype="float" name="PM_ERR_MAJA" precision="3" ucd="phys.angSize.smajAxis;pos.errorEllipse;pos.pm" unit="mas.yr-1" width="5">\n    <DESCRIPTION>\n     Proper motion error major axis\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PM_ERR_MINA" datatype="float" name="PM_ERR_MINA" precision="3" ucd="phys.angSize.sminAxis;pos.errorEllipse;pos.pm" unit="mas.yr-1" width="5">\n    <DESCRIPTION>\n     Proper motion error minor axis\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PM_ERR_ANGLE" datatype="short" name="PM_ERR_ANGLE" ucd="pos.posAng;pos.errorEllipse;pos.pm" unit="deg" width="3">\n    <DESCRIPTION>\n     Proper motion error angle\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PM_QUAL" arraysize="1" datatype="char" name="PM_QUAL" ucd="meta.code.qual;pos.pm" width="1">\n    <DESCRIPTION>\n     Proper motion quality\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="PM_BIBCODE" arraysize="*" datatype="char" name="PM_BIBCODE" ucd="meta.bib.bibcode;pos.pm" width="19">\n    <DESCRIPTION>\n     Proper motion reference\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="RVZ_TYPE" arraysize="1" datatype="char" name="RVZ_TYPE" width="1">\n    <DESCRIPTION>\n     Radial velocity / redshift type\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="RVZ_RADVEL" datatype="double" name="RVZ_RADVEL" precision="3" ucd="spect.dopplerVeloc.opt" unit="km.s-1">\n    <DESCRIPTION>\n     Radial Velocity\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="RVZ_ERROR" datatype="float" name="RVZ_ERROR" precision="3" ucd="stat.error;spect.dopplerVeloc" unit="km.s-1">\n    <DESCRIPTION>\n     Radial velocity / redshift error\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="RVZ_QUAL" arraysize="1" datatype="char" name="RVZ_QUAL" ucd="meta.code.qual;spect.dopplerVeloc" width="1">\n    <DESCRIPTION>\n     Radial velocity / redshift quality\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="RVZ_WAVELENGTH" arraysize="1" datatype="char" name="RVZ_WAVELENGTH" ucd="em.wl.central;spect.dopplerVeloc" width="1">\n    <DESCRIPTION>\n     Radial velocity / redshift wavelength class of the observ.\n     (RIVUXG)\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="RVZ_BIBCODE" arraysize="*" datatype="char" name="RVZ_BIBCODE" ucd="meta.bib.bibcode;spect.dopplerVeloc" width="19">\n    <DESCRIPTION>\n     Radial velocity / redshift reference\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="GALDIM_MAJAXIS" datatype="float" name="GALDIM_MAJAXIS" ucd="phys.angSize.smajAxis" unit="arcmin" width="4">\n    <DESCRIPTION>\n     Angular size major axis\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="GALDIM_MINAXIS" datatype="float" name="GALDIM_MINAXIS" ucd="phys.angSize.sminAxis" unit="arcmin" width="4">\n    <DESCRIPTION>\n     Angular size minor axis\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="GALDIM_ANGLE" datatype="short" name="GALDIM_ANGLE" ucd="pos.posAng" unit="deg" width="3">\n    <DESCRIPTION>\n     Galaxy ellipse angle\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="GALDIM_QUAL" arraysize="1" datatype="char" name="GALDIM_QUAL" ucd="meta.code.qual;phys.angSize" width="1">\n    <DESCRIPTION>\n     Galaxy dimension quality\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="GALDIM_WAVELENGTH" arraysize="1" datatype="char" name="GALDIM_WAVELENGTH" ucd="em.wl.central;phys.angSize" width="1">\n    <DESCRIPTION>\n     Gal. dim. wavelength class of the observ. (RIVUXG)\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="GALDIM_BIBCODE" arraysize="*" datatype="char" name="GALDIM_BIBCODE" ucd="meta.bib.bibcode;phys.angSize" width="19">\n    <DESCRIPTION>\n     Galaxy dimension reference\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="Diameter_diameter" datatype="double" name="Diameter_diameter" precision="2" ucd="PHYS.ANGSIZE" width="8">\n    <DESCRIPTION>\n     Diameter value\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="Diameter_Q" arraysize="1" datatype="char" name="Diameter_Q" ucd="META.CODE.QUAL" width="1">\n    <DESCRIPTION>\n     Quality\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="Diameter_unit" arraysize="4" datatype="char" name="Diameter_unit" ucd="META.UNIT" width="4">\n    <DESCRIPTION>\n     Unit (mas/km)\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="Diameter_error" datatype="double" name="Diameter_error" precision="2" ucd="STAT.ERROR" width="8">\n    <DESCRIPTION>\n     Error\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="Diameter_filter" arraysize="8" datatype="char" name="Diameter_filter" ucd="INSTR.FILTER" width="8">\n    <DESCRIPTION>\n     filter or wavelength\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="Diameter_method" arraysize="8" datatype="char" name="Diameter_method" ucd="INSTR.SETUP" width="8">\n    <DESCRIPTION>\n     calculation method\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="Diameter_bibcode" arraysize="19" datatype="char" name="Diameter_bibcode" ucd="META.BIB.BIBCODE" width="19">\n    <DESCRIPTION>\n     Bibcode\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="SCRIPT_NUMBER_ID" datatype="int" name="SCRIPT_NUMBER_ID" precision="1" ucd="meta.number" width="2"/>\n   <DATA>\n    <TABLEDATA>\n     <TR>\n      <TD>IC 2395</TD>\n      <TD>08 42 31.0</TD>\n      <TD>-48 06 00</TD>\n      <TD>5</TD>\n      <TD>5</TD>\n      <TD/>\n      <TD/>\n      <TD>0</TD>\n      <TD>D</TD>\n      <TD/>\n      <TD>2008NewA...13..370T</TD>\n      <TD>08 42 31.0</TD>\n      <TD>-48 06 00</TD>\n      <TD>5</TD>\n      <TD>5</TD>\n      <TD/>\n      <TD/>\n      <TD>0</TD>\n      <TD>D</TD>\n      <TD/>\n      <TD>2008NewA...13..370T</TD>\n      <TD>    1.379</TD>\n      <TD>3</TD>\n      <TD>0.087</TD>\n      <TD>C</TD>\n      <TD>2018A&amp;A...618A..93C</TD>\n      <TD>   -4.464</TD>\n      <TD>    3.293</TD>\n      <TD>3</TD>\n      <TD>3</TD>\n      <TD>0.237</TD>\n      <TD>0.296</TD>\n      <TD>90</TD>\n      <TD>C</TD>\n      <TD>2018A&amp;A...618A..93C</TD>\n      <TD>v</TD>\n      <TD>17.660</TD>\n      <TD>1.720</TD>\n      <TD>B</TD>\n      <TD>O</TD>\n      <TD>2018A&amp;A...619A.155S</TD>\n      <TD/>\n      <TD/>\n      <TD>0</TD>\n      <TD/>\n      <TD/>\n      <TD/>\n      <TD/>\n      <TD/>\n      <TD/>\n      <TD/>\n      <TD/>\n      <TD/>\n      <TD/>\n      <TD>1</TD>\n     </TR>\n    </TABLEDATA>\n   </DATA>\n  </TABLE>\n </RESOURCE>\n</VOTABLE>\n'
    fake_file = BytesIO()
    fake_file.write(content)
    return Table.read(fake_file, format="votable")


def gaia_load_tables():
    c1 = TapColumn(None)
    c1.name = "ra"
    c1.description = "Right ascension"
    c1.unit = "deg"
    c1.ucd = "pos.eq.ra;meta.main"
    c1.utype = "Char.SpatialAxis.Coverage.Location.Coord.Position2D.Value2.C1"
    c1.datatype = "double"
    c1.data_type = None
    c1.flag = "primary"
    t1 = TapTableMeta()
    t1.name = "gaiaedr3.gaia_source"
    t1.description = "This table has an entry for every Gaia observed source as listed in the\nMain Database accumulating catalogue version from which the catalogue\nrelease has been generated. It contains the basic source parameters,\nthat is only final data (no epoch data) and no spectra (neither final\nnor epoch)."
    t1.columns = [c1]
    t2 = TapTableMeta()
    t2.name = "other.name"
    return [t1, t2]


def simbad_search_mock():
    coords = SkyCoord(
        "08 42 31.0", "-48 06 00", unit=(u.hourangle, u.deg), frame="icrs"
    )
    return SimbadResult(coords=coords)


def gaia_launch_job_async():
    return Job(None)


def gaia_job_get_results():
    fake_file = BytesIO()
    fake_file.write(
        b'<?xml version="1.0" encoding="utf-8"?>\n<!-- Produced with astropy.io.votable version 4.3.1\n     http://www.astropy.org/ -->\n<VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.ivoa.net/xml/VOTable/v1.3 http://www.ivoa.net/xml/VOTable/VOTable-1.4.xsd">\n <RESOURCE type="results">\n  <TABLE>\n   <FIELD ID="ra" datatype="double" name="ra" ucd="pos.eq.ra;meta.main" unit="deg" utype="Char.SpatialAxis.Coverage.Location.Coord.Position2D.Value2.C1">\n    <DESCRIPTION>\n     Right ascension\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="dec" datatype="double" name="dec" ucd="pos.eq.dec;meta.main" unit="deg" utype="Char.SpatialAxis.Coverage.Location.Coord.Position2D.Value2.C2">\n    <DESCRIPTION>\n     Declination\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="parallax" datatype="double" name="parallax" ucd="pos.parallax.trig" unit="mas">\n    <DESCRIPTION>\n     Parallax\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="pmra" datatype="double" name="pmra" ucd="pos.pm;pos.eq.ra" unit="mas.yr-1">\n    <DESCRIPTION>\n     Proper motion in right ascension direction\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="pmdec" datatype="double" name="pmdec" ucd="pos.pm;pos.eq.dec" unit="mas.yr-1">\n    <DESCRIPTION>\n     Proper motion in declination direction\n    </DESCRIPTION>\n   </FIELD>\n   <FIELD ID="dist" datatype="double" name="dist"/>\n   <DATA>\n    <TABLEDATA>\n     <TR>\n      <TD>130.6274659271372</TD>\n      <TD>-48.09917905206921</TD>\n      <TD>0.7769760369768328</TD>\n      <TD>-9.275583230412119</TD>\n      <TD>-2.3170710609933667</TD>\n      <TD>0.0014093212232431582</TD>\n     </TR>\n     <TR>\n      <TD>130.62912580829916</TD>\n      <TD>-48.098221388531556</TD>\n      <TD>0.26055419976133914</TD>\n      <TD>-5.2876656234328</TD>\n      <TD>7.167618779202627</TD>\n      <TD>0.0017847538833669556</TD>\n     </TR>\n     <TR>\n      <TD>130.62745539926073</TD>\n      <TD>-48.09838367780551</TD>\n      <TD>0.32178905361257304</TD>\n      <TD>-2.294342930019437</TD>\n      <TD>2.8861911269698393</TD>\n      <TD>0.001987495669263213</TD>\n     </TR>\n     <TR>\n      <TD>130.6258748619376</TD>\n      <TD>-48.09832997940612</TD>\n      <TD>0.21007317524953678</TD>\n      <TD>-1.2976179809788364</TD>\n      <TD>2.778797497310169</TD>\n      <TD>0.002768705397220442</TD>\n     </TR>\n     <TR>\n      <TD>130.62471992642566</TD>\n      <TD>-48.09891066337167</TD>\n      <TD>0.4130728870763981</TD>\n      <TD>-3.193523817495721</TD>\n      <TD>3.4058573717702565</TD>\n      <TD>0.0031703755189884863</TD>\n     </TR>\n    </TABLEDATA>\n   </DATA>\n  </TABLE>\n </RESOURCE>\n</VOTABLE>\n'
    )
    return Table.read(fake_file)


def remove_last_digits(string, number, precision):
    nstring = str(round(number, precision))
    if nstring in string:
        return re.sub(r"(?<=" + nstring + r")\d*", "", string=string)
    return string


def multiline2singleline(string: str):
    return " ".join(
        list(
            filter(
                len,
                list(
                    chain(*[line.split(" ") for line in string.splitlines()])
                ),
            )
        )
    )


# remove unimportant differences between queries"
# 1. very last digit difference in coordinates due to
# precision in calculation.
# 2. linejumps and several spaces that do not affect
# query semantics
def format_query_string(string):
    sra = -48.1000
    sdec = 130.6291
    prec = 4
    string = remove_last_digits(string, sra, prec)
    string = remove_last_digits(string, sdec, prec)
    string = multiline2singleline(string)
    return string


@pytest.fixture
def mock_simbad_query_object(mocker):
    mocker.patch(
        "astroquery.simbad.Simbad.query_object",
        return_value=simbad_query_object(),
    )


@pytest.fixture
def mock_gaia_load_tables(mocker):
    return mocker.patch(
        "astroquery.gaia.Gaia.load_tables", return_value=gaia_load_tables()
    )


@pytest.fixture
def mock_simbad_search(mocker):
    return mocker.patch(
        "opencluster.fetcher.simbad_search", return_value=simbad_search_mock()
    )


@pytest.fixture
def mock_gaia_launch_job_async(mocker):
    mocker.patch(
        "astroquery.utils.tap.model.job.Job.get_results",
        return_value=gaia_job_get_results(),
    )
    return mocker.patch(
        "astroquery.gaia.Gaia.launch_job_async",
        return_value=gaia_launch_job_async(),
    )


class TestFetcher:
    def test_simbad_search(self, mock_simbad_query_object):
        result = simbad_search("ic2395", cols=["coordinates", "parallax"])
        assert result.coords.to_string("hmsdms", precision=2) == SkyCoord(
            ra=130.62916667, dec=-48.1, frame="icrs", unit="deg"
        ).to_string("hmsdms", precision=2)
        assert isinstance(result.table, Table)
        assert sorted(list(result.table.columns)) == sorted(
            [
                "MAIN_ID",
                "RA",
                "DEC",
                "RA_PREC",
                "DEC_PREC",
                "COO_ERR_MAJA",
                "COO_ERR_MINA",
                "COO_ERR_ANGLE",
                "COO_QUAL",
                "COO_WAVELENGTH",
                "COO_BIBCODE",
                "RA_2",
                "DEC_2",
                "RA_PREC_2",
                "DEC_PREC_2",
                "COO_ERR_MAJA_2",
                "COO_ERR_MINA_2",
                "COO_ERR_ANGLE_2",
                "COO_QUAL_2",
                "COO_WAVELENGTH_2",
                "COO_BIBCODE_2",
                "PLX_VALUE",
                "PLX_PREC",
                "PLX_ERROR",
                "PLX_QUAL",
                "PLX_BIBCODE",
                "SCRIPT_NUMBER_ID",
            ]
        )
        empty_result = simbad_search("invalid_identifier")
        assert empty_result.table is None and empty_result.coords is None

    def test_table_info(self, mock_gaia_load_tables):
        result = table_info("gaiaedr3", only_names=True)
        assert len(result) == 1
        assert result[0].name == "gaiaedr3.gaia_source"
        assert isinstance(result[0].description, str)
        assert result[0].columns == None
        result = table_info("gaiaedr3")
        assert isinstance(result[0].columns, Table)

    def test_query_region(self, mock_simbad_search):
        name = "dummy_name"
        coords = simbad_search_mock().coords
        ra = coords.ra.value
        dec = coords.dec.value
        table = "gaiaedr3.gaia_source"

        correct = "SELECT *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291, -48.1) ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS( POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) ORDER BY dist ASC"

        queries = [
            query_region(name, 0.5),
            query_region(coords, 0.5),
            query_region((ra, dec), 0.5),
            query_region((ra, dec), 30 * u.arcmin),
            query_region((ra, dec), 0.5 * u.deg),
        ]

        for q in queries:
            q = format_query_string(q.build())
            assert correct == q

    def test_select(self, mock_simbad_search):
        correct = "SELECT parallax, pmra, pmdec, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291, -48.1) ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS( POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) ORDER BY dist ASC"
        q = query_region("dummy_name", 0.5).select("parallax", "pmra", "pmdec")
        assert format_query_string(q.build()) == correct

    def test_from_table(self, mock_simbad_search):
        correct = "SELECT *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291, -48.1) ) AS dist FROM table WHERE 1 = CONTAINS( POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) ORDER BY dist ASC"
        q = query_region("dummy_name", 0.5).from_table("table")
        assert format_query_string(q.build()) == correct

    @pytest.mark.parametrize(
        "column, operator, value, test, correct",
        [
            ("parallax", ">>", 3, ValueError, None),
            (
                "string_column",
                "=",
                "'string'",
                Ok,
                "SELECT *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291, -48.1) ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS( POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) AND string_column = 'string' ORDER BY dist ASC",
            ),
            (
                "string_column",
                "LIKE",
                "'string'",
                Ok,
                "SELECT *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291, -48.1) ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS( POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) AND string_column LIKE 'string' ORDER BY dist ASC",
            ),
            (
                "parallax",
                ">=",
                5,
                Ok,
                "SELECT *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291, -48.1) ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS( POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) AND parallax >= 5 ORDER BY dist ASC",
            ),
        ],
    )
    def test_single_where(
        self, column, operator, value, test, correct, mock_simbad_search
    ):
        q = query_region("dummy_name", 0.5)
        verify_result(test, lambda: q.where(column, operator, value))
        if test == Ok:
            assert format_query_string(q.build()) == correct

    def test_multiple_where(self, mock_simbad_search):
        correct = "SELECT *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291, -48.1) ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS( POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) AND column1 >= 5 AND column2 <= 17 AND column3 LIKE 'string' ORDER BY dist ASC"
        q = (
            query_region("dummy_name", 0.5)
            .where("column1", ">=", 5)
            .where("column2", "<=", 17)
            .where("column3", "LIKE", "'string'")
            .where("column1", ">=", 5)
        )
        assert format_query_string(q.build()) == correct

    def test_top(self, mock_simbad_search):
        correct = "SELECT TOP 31 *, DISTANCE( POINT('ICRS', ra, dec), POINT('ICRS', 130.6291, -48.1) ) AS dist FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS( POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) ORDER BY dist ASC"
        q = query_region("dummy_name", 0.5).top(10).top(31)
        assert format_query_string(q.build()) == correct

    def test_build_count(self, mock_simbad_search):
        correct = "SELECT COUNT(*) FROM gaiaedr3.gaia_source WHERE 1 = CONTAINS( POINT('ICRS', ra, dec), CIRCLE('ICRS', 130.6291, -48.1, 0.5)) AND column1 >= 5 AND column2 <= 17 AND column3 LIKE 'string'"
        q = (
            query_region("dummy_name", 0.5)
            .where("column1", ">=", 5)
            .where("column2", "<=", 17)
            .where("column3", "LIKE", "'string'")
            .where("column1", ">=", 5)
        )
        assert format_query_string(q.build_count()) == correct

    def test_get(self, mock_simbad_search, mock_gaia_launch_job_async, mocker):
        q = (
            query_region("gaiaedr3.gaia_source", 0.5)
            .select("ra", "dec", "parallax", "pmra", "pmdec")
            .top(5)
        )
        query = q.build()
        result = q.get()
        mock_gaia_launch_job_async.assert_called_with(query=query)
        assert isinstance(result, Table)
        result = q.get(dump_to_file=True, output_file="test_file.xml")
        mock_gaia_launch_job_async.assert_called_with(
            query=query, dump_to_file=True, output_file="test_file.xml"
        )
        assert result is None

    def test_count(
        self, mock_simbad_search, mock_gaia_launch_job_async, mocker
    ):
        q = (
            query_region("gaiaedr3.gaia_source", 0.5)
            .select("ra", "dec", "parallax", "pmra", "pmdec")
            .top(5)
        )
        query = q.build_count()
        result = q.count()
        mock_gaia_launch_job_async.assert_called_with(query=query)
        assert isinstance(result, Table)
        result = q.count(dump_to_file=True, output_file="test_file.xml")
        mock_gaia_launch_job_async.assert_called_with(
            query=query, dump_to_file=True, output_file="test_file.xml"
        )
        assert result is None
