""" <Opencluster, a package for open star cluster probabilities calculations>
    Copyright (C) 2020  González Simón Pedro

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import warnings

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.utils.commons import coord_to_radec, radius_to_unit

from attr import attrib, attrs

import numpy as np


class CatalogNotFoundException(Exception):
    def __init__(self, message="No known catalog could be found"):
        self.message = message
        super().__init__(self.message)


def unsilence_warnings():
    def decorator(func):
        def wrapper(*args, **kwargs):
            warnings.filterwarnings("error")
            try:
                return func(*args, **kwargs)
            except Exception as exception:
                if "No known catalog could be found" in str(exception):
                    raise CatalogNotFoundException from None
                warnings.warn(str(exception))

        return wrapper

    return decorator


@unsilence_warnings()
def simbad_search(id):
    result = Simbad.query_object(id)
    ra = np.array(result["RA"])[0].replace(" ", "h", 1).replace(" ", "m", 1) + "s"
    dec = np.array(result["DEC"])[0].replace(" ", "d", 1).replace(" ", "m", 1) + "s"
    return SkyCoord(ra, dec, frame="icrs")


def cone_search(
    *,
    name=None,
    radius=None,
    ra=None,
    dec=None,
    columns=[
        "ra",
        "ra_error",
        "dec",
        "dec_error",
        "parallax",
        "parallax_error",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
    ],
    dump_to_file=False,
    row_limit=50,
    **kwargs,
):
    if not isinstance(radius, (float, int)):
        raise ValueError("radious must be specified with int or float")
    else:
        radius = u.Quantity(radius, u.degree)
    if not ((name is not None) ^ (ra is not None and dec is not None)):
        raise ValueError("'name' or 'ra' and 'dec' are required (not both)")
    if name is not None:
        if not isinstance(name, str):
            raise ValueError("name must be string")
        else:
            coord = simbad_search(name)
    if (ra, dec) != (None, None):
        if not isinstance(ra, (float, int)) or not isinstance(dec, (float, int)):
            raise ValueError("ra and dec must be numeric")
        else:
            coord = SkyCoord(ra, dec, unit=(u.degree, u.degree), frame="icrs")
    if not isinstance(row_limit, int):
        raise ValueError("row_limit must be int")
    """ elif row_limit == -1:
        warnings.warn("Row limit set to unlimited.")
    """
    Gaia.ROW_LIMIT = row_limit

    job = Gaia.cone_search_async(
        coordinate=coord,
        radius=radius,
        columns=columns,
        dump_to_file=dump_to_file,
        **kwargs,
    )

    if not dump_to_file:
        table = job.get_results()
        return table


def load_VOTable(path):
    table = (
        parse(path, pedantic=False).get_first_table().to_table(use_names_over_ids=True)
    )
    return table


@attrs
class Query:
    table = attrib(default=Gaia.MAIN_GAIA_TABLE)
    column_filters = attrib(default=dict())
    row_limit = attrib(default=-1)
    radius = attrib(default=None)
    coords = attrib(default=None)
    ra_name = attrib(default=Gaia.MAIN_GAIA_TABLE_RA)
    dec_name = attrib(default=Gaia.MAIN_GAIA_TABLE_DEC)
    columns = attrib(
        default=[
            "ra",
            "ra_error",
            "dec",
            "dec_error",
            "parallax",
            "parallax_error",
            "pmra",
            "pmra_error",
            "pmdec",
            "pmdec_error",
        ]
    )
    available_tables = attrib(default=None)

    def where(self, condition):
        if not isinstance(condition, dict):
            raise ValueError("condition must be dict: {'column': '> value'}")
        self.column_filters = {**self.column_filters, **condition}
        return self

    def select(self, columns):
        if columns != "*" and (
            not isinstance(columns, list)
            or not all(isinstance(elem, str) for elem in columns)
        ):
            raise ValueError("columns must be list of strings")
        self.columns = columns
        return self

    def list_tables(self, table=None):
        if self.available_tables is None:
            self.available_tables = Gaia.load_tables()
        return [table.get_qualified_name() for table in self.available_tables]

    def list_columns(self, table=None):
        if table is None:
            table = self.table
        elif not isinstance(table, str):
            raise ValueError("table must be string")
        return [column.name for column in Gaia.load_table(table).columns]

    def table_description(self, table=None):
        if table is None:
            table = self.table
        elif not isinstance(table, str):
            raise ValueError("table must be string")
        return f"table = {Gaia.load_table(table)}"

    def from_table(self, table, ra_name=None, dec_name=None):
        if not isinstance(table, str):
            raise ValueError("table must be string")
        if (ra_name, dec_name) != (None, None):
            if not isinstance(ra_name, str) or not isinstance(dec_name, str):
                raise ValueError("ra and dec parameter names in table must be string")
            else:
                self.ra_name = ra_name
                self.dec_name = dec_name
        self.table = table
        return self

    def build(self):
        """parse things and do de query"""

        if self.columns != "*":
            columns = ",".join(map(str, self.columns))
        else:
            columns = "*"

        if self.radius is not None and self.coords is not None:
            raHours, dec = coord_to_radec(self.coords)
            ra = raHours * 15.0

        query = """
                SELECT
                  {row_limit}
                  {columns},
                  DISTANCE(
                    POINT('ICRS', {ra_column}, {dec_column}),
                    POINT('ICRS', {ra}, {dec})
                  ) AS dist
                FROM
                  {table_name}
                WHERE
                  1 = CONTAINS(
                    POINT('ICRS', {ra_column}, {dec_column}),
                    CIRCLE('ICRS', {ra}, {dec}, {radius}))
                """.format(
            **{
                "ra_column": self.ra_name,
                "row_limit": "TOP {0}".format(self.row_limit)
                if self.row_limit > 0
                else "",
                "dec_column": self.dec_name,
                "columns": columns,
                "ra": ra,
                "dec": dec,
                "radius": self.radius,
                "table_name": self.table,
            }
        )

        if self.column_filters:
            query_filters = "".join(
                [
                    """AND {column} {condition}
                """.format(
                        **{"column": column, "condition": condition}
                    )
                    for column, condition in self.column_filters.items()
                ]
            )
            query += query_filters

        query += """ORDER BY
                dist ASC
                 """
        return query

    def top(self, row_limit):
        """set row limit"""
        if not isinstance(row_limit, int):
            raise ValueError("row_limit must be int")
        self.row_limit = row_limit
        return self

    def get(
        self,
        dump_to_file=False,
        output_file=None,
        output_format="votable",
        verbose=False,
    ):
        query = self.build()
        job = Gaia.launch_job_async(
            query=query,
            output_file=output_file,
            output_format=output_format,
            verbose=verbose,
            dump_to_file=dump_to_file,
        )
        if not dump_to_file:
            table = job.get_results()
            return table


def region(*, ra=None, dec=None, name=None, radius):
    if not isinstance(radius, u.quantity.Quantity):
        raise ValueError("radious must be astropy.units.quantity.Quantity")
    if not ((name is not None) ^ (ra is not None and dec is not None)):
        raise ValueError("'name' or 'ra' and 'dec' are required (not both)")
    if name is not None:
        if not isinstance(name, str):
            raise ValueError("name must be string")
        else:
            coord = simbad_search(name)
    if (ra, dec) != (None, None):
        if not isinstance(ra, (float, int)) or not isinstance(dec, (float, int)):
            raise ValueError("ra and dec must be numeric")
        else:
            coord = SkyCoord(ra, dec, unit=(u.degree, u.degree), frame="icrs")
    radiusDeg = radius_to_unit(radius, unit="deg")
    query = Query(radius=radiusDeg, coords=coord)
    return query
