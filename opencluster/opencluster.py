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

import inspect
import io
import warnings

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse
from astropy.table.table import Table

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.utils.commons import coord_to_radec, radius_to_unit

from attr import attrib, attrs, validators

import numpy as np

import pandas as pd

import scipy.optimize as opt
from scipy.stats import norm


class CatalogNotFoundException(Exception):
    def __init__(self, message="No known catalog could be found"):
        self.message = message
        super().__init__(self.message)


def checkargs(function):
    def wrapper(*args, **kwargs):
        argnames = inspect.getfullargspec(function).args
        arguments = dict(kwargs)
        for index, argvalue in enumerate(args):
            if index < len(argnames):
                arguments[argnames[index]] = argvalue
        annotations = function.__annotations__
        for name, value in arguments.items():
            annotation = annotations.get(name)
            if annotation:
                if not isinstance(value, annotation):
                    raise TypeError(
                        "{} is not of type {}".format(name, annotation)
                    )
        return function(*args, **kwargs)

    return wrapper


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
    ra = (
        np.array(result["RA"])[0].replace(" ", "h", 1).replace(" ", "m", 1)
        + "s"
    )
    dec = (
        np.array(result["DEC"])[0].replace(" ", "d", 1).replace(" ", "m", 1)
        + "s"
    )
    return SkyCoord(ra, dec, frame="icrs")


@attrs
class Query:

    QUERY_TEMPLATE = """
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
    """

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

    def from_table(self, table, ra_name=None, dec_name=None):
        if not isinstance(table, str):
            raise ValueError("table must be string")
        if (ra_name, dec_name) != (None, None):
            if not isinstance(ra_name, str) or not isinstance(dec_name, str):
                raise ValueError(
                    "ra and dec parameter names in table must be string"
                )
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

        row_limit = f"TOP {self.row_limit}" if self.row_limit > 0 else ""

        query = self.QUERY_TEMPLATE.format(
            row_limit=row_limit,
            columns=columns,
            ra_column=self.ra_name,
            dec_column=self.dec_name,
            ra=ra,
            dec=dec,
            table_name=self.table,
            radius=self.radius,
        )

        if self.column_filters:
            query_filters = "".join(
                [
                    """AND {column} {condition}
                """.format(
                        column=column, condition=condition
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

    def get(self, **kwargs):
        query = self.build()
        job = Gaia.launch_job_async(query=query, **kwargs)
        if not kwargs.get("dump_to_file"):
            table = job.get_results()
            return table


def query_region(*, ra=None, dec=None, name=None, radius):
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
        if not isinstance(ra, (float, int)) or not isinstance(
            dec, (float, int)
        ):
            raise ValueError("ra and dec must be numeric")
        else:
            coord = SkyCoord(ra, dec, unit=(u.degree, u.degree), frame="icrs")
    radiusDeg = radius_to_unit(radius, unit="deg")
    query = Query(radius=radiusDeg, coords=coord)
    return query


def list_remotes():
    available_tables = Gaia.load_tables()
    return [table.get_qualified_name() for table in available_tables]


@checkargs
def remote_info(table: str):
    table = Gaia.load_table(table)
    description = f"table = {table}"
    cols = [column.name for column in table.columns]
    return description, cols


@checkargs
def load_file(filepath_or_buffer: (str, io.IOBase) = None):
    table = (
        parse(filepath_or_buffer)
        .get_first_table()
        .to_table(use_names_over_ids=True)
    )
    return OCTable(table)


@checkargs
def load_remote(
    *,
    table: str = None,
    columns=None,
    filters=None,
    ra=None,
    dec=None,
    name=None,
    radius,
    limit=-1,
    **kwargs,
):

    query = (
        query_region(name=name, ra=ra, dec=dec, radius=radius)
        .select(columns)
        .where(filters)
        .top(limit)
    )

    if table is not None:
        query.from_table(table)

    result = query.get(**kwargs)

    if not kwargs.get("dump_to_file"):
        return OCTable(result)


def lognegexp_normal(params, x):
    return (
        params[0] * np.log(x / params[1])
        + params[2] * np.exp(-x / params[3])
        + params[4] * norm(params[5], params[6]).pdf(x)
    )


def normalnegexp_normal():
    """alternative that uses piecewise defined function
    with normal|negative exponential for field, normal for cluster"""
    ...


@attrs(frozen=True)
class OCTable:
    table = attrib(validator=validators.instance_of(Table))

    def __getattr__(self, a):
        return getattr(self.table, a)

    @checkargs
    def fit_plx(
        self,
        initial_params,
        plx_col: str = "parallax",
        pdfs: str = "lognegexp_normal",
        bins: (int, str) = "fd",
        ftol: float = 1.0e-12,
        gtol: float = 1.0e-12,
        xtol: float = 1.0e-12,
        **kwargs,
    ):
        plx = self.table[plx_col].data
        plx = plx[plx.mask is False].data

        his, bin_edges = np.histogram(plx, bins=bins, density=True, **kwargs)
        his = his / np.sum(his)
        import ipdb

        ipdb.set_trace()
        bin_edges = bin_edges - (bin_edges[-1] - bin_edges[0]) / 2.0

        solution = opt.least_squares(
            fun=lambda params, x, y: lognegexp_normal(params, x) - y,
            x0=np.array(initial_params),
            method="lm",
            args=(bin_edges[1:], his),
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
        )

        linspace = np.linspace(bin_edges[0] + 0.1, bin_edges[-1], 10000)
        results = lognegexp_normal(solution.x, linspace)
        return {
            "params": solution.x,
            "fit": pd.DataFrame({"plx": linspace, "pdf": results}),
        }
