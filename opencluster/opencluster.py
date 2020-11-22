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

# =============================================================================
# DOCS
# =============================================================================

"""Package for membership probability calculation from remote or local data."""

# =============================================================================
# IMPORTS
# =============================================================================

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

import scipy.optimize as opt


def checkargs(function):
    """Check arguments match their annotated type.

    Parameters
    ----------
    function : function

    Returns
    -------
    function result

    Raises
    ------
    TypeError
    If argument does not match annotated type.
    """

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


@checkargs
def simbad_search(id: str):
    """Search an identifier in Simbad catalogues.

    Parameters
    ----------
    id : str

    Returns
    -------
    coordinates : astropy.coordinates.SkyCoord
        Coordinates of object if found, None otherwise

    Warns
    ------
    Identifier not found
        If the identifier has not been found in Simbad Catalogues.
    """
    result = Simbad.query_object(id)

    if result is None:
        warnings.warn("Identifier not found.")
        return None
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
    """Query class to retrieve data from remote gaia catalogues.

    Attributes
    ----------
    table : str, optional
        Name of the Gaia catalogues table.
        (default is astroquery.gaia.Gaia.MAIN_GAIA_TABLE)
    ra_name : str, optional
        Name of the column of right ascension column in the selected table.
        (default is astroquery.gaia.Gaia.MAIN_GAIA_TABLE_RA)
    dec_name : str, optional
        Name of the declination column in the selected table.
        (default is astroquery.gaia.Gaia.MAIN_GAIA_TABLE_DEC)
    columns : str or list of str
        If str, must be '*', indicating all columns
        If list of str, must be list of valid column names.
    column_filters : dict, optional
        Dictionary of filters: {'column_name' : 'column_filter'}
        column_name: str
            Valid column in the selected table
        column_filter: str
            Must start with '<', '>', '<=', '>=' and end with a numeric value.
    row_limit : int, optional
        Limit of rows to retrieve from the remote table.
        (default is -1, meaning all found rows)
    radius : u.quantity.Quantity
        Radius of the cone search.
    coords : astropy.coordinates.SkyCoord
        Coordinates of the center of the cone search.
    QUERY_TEMPLATE : multiline str template used for the query
    """

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
    column_filters = attrib(factory=dict)
    row_limit = attrib(default=-1)
    radius = attrib(default=None)
    coords = attrib(default=None)
    ra_name = attrib(default=Gaia.MAIN_GAIA_TABLE_RA)
    dec_name = attrib(default=Gaia.MAIN_GAIA_TABLE_DEC)
    columns = attrib(default="*")

    def where(self, column_filters):
        """Add filters or conditions to the query.

        Parameters
        ----------
        column_filters : dict, optional
            Dictionary of filters: {'column_name' : 'column_filter'}
            column_name: str
                Valid column in the selected table
            column_filter: str
                Must start with '<', '>', '<=', '>='and end with a
                numeric value.

        Returns
        -------
        query : Query class instance

        Raises
        ------
        ValueError if an attribute does not match type.
        KeyError if columns is not '*' and filter has invalid column.
        """
        if not isinstance(column_filters, dict):
            raise ValueError(
                "column_filters must be dict: {'column': '> value'}"
            )
        if self.columns != '*':
            for col in column_filters.keys():
                if col not in self.columns:
                    raise KeyError("invalid column in filter")

        self.column_filters = {**self.column_filters, **column_filters}
        return self

    def select(self, columns):
        """Select table columns to be retrieved.

        Parameters
        ----------
        columns : str or list of str
            If str, must be '*', indicating all columns
            If list of str, must be list of valid column names.

        Returns
        -------
        query : Query class instance

        Raises
        ------
        ValueError if an attribute does not match type.
        """
        if columns != "*" and (
            not isinstance(columns, list)
            or not all(isinstance(elem, str) for elem in columns)
        ):
            raise ValueError("columns must be list of strings")
        self.columns = columns
        return self

    @checkargs
    def from_table(self, table, ra_name:str=None, dec_name:str=None):
        """Select Gaia table.

        Parameters
        ----------
        table : str, optional
            Name of the Gaia catalogues table
            (default is astroquery.gaia.Gaia.MAIN_GAIA_TABLE)
        ra_name : str, optional
            Name of the column of right ascension column in the selected table.
            (default is astroquery.gaia.Gaia.MAIN_GAIA_TABLE_RA)
        dec_name : str, optional
            Name of the declination column in the selected table.
            (default is astroquery.gaia.Gaia.MAIN_GAIA_TABLE_DEC)

        Returns
        -------
        query : Query class instance

        Raises
        ------
        ValueError if an attribute does not match type.
        """
        self.ra_name = ra_name
        self.dec_name = dec_name
        self.table = table
        return self

    def build(self):
        """Build and perform query.

        Returns
        -------
        query : string with built query.
        """
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

        query += """ORDER BY dist ASC"""
        return query

    def top(self, row_limit):
        """Set row limit for the query.

        Attributes
        ----------
        row_limit : int, optional
            Limit of rows to retrieve from the remote table.
            (default is -1, meaning all found rows)

        Returns
        -------
        query : Query class instance

        Raises
        ------
        ValueError if an attribute does not match type.
        """
        if not isinstance(row_limit, int):
            raise ValueError("row_limit must be int")
        self.row_limit = row_limit
        return self

    def get(self, **kwargs):
        """Build and performe query.

        Parameters:
        -----------
        Parameters that are passed through **kwargs to
        astroquery.gaia.Gaia.launch_job_async
        For example:
        dump_to_file : bool
            If True, results will be stored in file
            (default is False).
        output_file : str
            Name of the output file

        Returns
        -------
        octable : opencluster.OCTable
            Instance with query results,
            None if dump_to_file is True
        """
        query = self.build()
        job = Gaia.launch_job_async(query=query, **kwargs)
        if not kwargs.get("dump_to_file"):
            table = job.get_results()
            return table


def query_region(*, ra=None, dec=None, name=None, radius):
    """Make a cone search type query for retrieving data.

    Parameters
    ----------
    ra : int, float, optional
        Right ascention of the center of the cone search.
    dec : int, float, optional
        Declination of the center of the cone search.
    name : str, optional
        Name of the Simbad identifier to set the center of the cone search.
    radius : astropy.units.quantity.Quantity
        Radius of the cone search.

    Returns
    -------
    query : Query class instance

    Raises
    ------
    ValueError if an attribute does not match type, or if
    both name and ra & dec are provided.
    """
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
    """List available tables in Gaia catalogues.

    Returns
    -------
    tables : list of str
        Available tables names
    """
    available_tables = Gaia.load_tables()
    names = []
    for table in available_tables:
        name = table.get_qualified_name()
        index = name.index(".") + 1
        name = name[index:]
        names.append(name)
    return names


@checkargs
def remote_info(table: str):
    """Remote table description and column names.

    Parameters
    ----------
    table : str
        valid table name from Gaia catalogues.

    Returns
    -------
    description : str
        Short text describing contents of the remote table.
    cols : list of str
        Column names of the remote table.
    """
    table = Gaia.load_table(table)
    description = f"table = {table}"
    cols = [column.name for column in table.columns]
    return description, cols


@checkargs
def load_file(filepath_or_buffer: (str, io.IOBase)):
    """Load a xml VOT table file as a OCTable instance.

    Parameters
    ----------
    filepath_or_buffer : str or io.IOBase
        Path of file or opened file

    Returns
    -------
    octable : OCTable instance
    """
    table = (
        parse(filepath_or_buffer)
        .get_first_table()
        .to_table(use_names_over_ids=True)
    )
    return OCTable(table)


@checkargs
def load_remote(
    *,
    table: str = Gaia.MAIN_GAIA_TABLE,
    columns="*",
    filters=None,
    ra=None,
    dec=None,
    name=None,
    radius,
    limit=-1,
    **kwargs,
):
    """Retrieve remote data from Gaia catalogues.

    Parameters
    ----------
    table : str, optional
        Name of the Gaia catalogues table.
        (default is astroquery.gaia.Gaia.MAIN_GAIA_TABLE)
    columns : str or list of str
        If str, must be '*', indicating all columns
        If list of str, must be list of valid column names.
    filters : dict, optional
        Dictionary of filters: {'column_name' : 'column_filter'}
        column_name: str
            Valid column in the selected table
        column_filter: str
            Must start with '<', '>', '<=', '>=' and end with a numeric value.
    limit : int, optional
        Limit of rows to retrieve from the remote table.
        (default is -1, meaning all found rows)
    radius : u.quantity.Quantity
        Radius of the cone search.
    ra : int, float, optional
        Right ascention of the center of the cone search (default is None).
    dec : int, float, optional
        Declination of the center of the cone search (default is None).

    Returns
    -------
    octable : opencluster.OCTable
        Instance with query results,
        None if dump_to_file is True.
    """
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


def lognegexp_normal(par, x):
    """Calculate f(x) where f is parallax approximation function.

    It models the field as the sum of log and negative exp,
    and the cluster as a gaussian.

    Parameters
    ----------
        par : array of float
            Must contain in order: k, a, h, r, kc, u, s
        x : float

    Returns
    -------
    y : float
        y = f(x) =
        k.log(x/a) + h.e^(-x/r) + (kc/sqrt(2.pi.s)).e^(-((x-u)^2)/2.s^2)
    """
    return (
        (par[0] * np.log(x / par[1]))
        + (par[2] * np.exp(-x / par[3]))
        + (par[4] / np.sqrt(2 * np.pi * par[6]))
        * np.exp(-((x - par[5]) ** 2) / (2 * (par[6] ** 2)))
    )


def normalnegexp_normal():
    """Calculate f(x) where f is parallax approximation function.

    Alternative that uses piecewise defined function
    with normal|negative exponential for field, normal for cluster.
    """
    ...


def two_normal(par, x):
    """Calculate f(x) where f is proper motion approximation function.

    Uses 2 normal distributions, one for the field and one for the cluster.
    """
    return (
        par[4]
        / np.sqrt(2.0 * np.pi * par[1])
        * np.exp(-((x - par[0]) ** 2 / (2 * par[1] ** 2)))
    ) + (
        par[5]
        / np.sqrt(2.0 * np.pi * par[3])
        * np.exp(-((x - par[2]) ** 2 / (2 * par[3] ** 2)))
    )


@attrs(frozen=True)
class OCTable:
    table = attrib(validator=validators.instance_of(Table))

    def __getattr__(self, a):
        return getattr(self.table, a)

    def __getitem__(self, a):
        return self.table[a]

    def __iter__(self):
        return iter(self.table)

    def __len__(self):
        return len(self.table)

    @checkargs
    def fit_plx(
        self,
        initial_params,
        plx_col: str = "parallax",
        lower_lim: (float, int) = None,
        upper_lim: (float, int) = None,
        pdfs: str = "lognegexp_normal",
        bins: (int, str) = "fd",
        ftol: float = 1.0e-12,
        gtol: float = 1.0e-12,
        xtol: float = 1.0e-12,
        arenou_criterion: bool = True,
        bp_rp_col: str = "bp_rp",
        bp_rp_excess_factor_col: str = "phot_bp_rp_excess_factor",
        **kwargs,
    ):

        # get pandas dataframe from VOTable
        df = self.table.to_pandas()

        # select columns that are going to be used
        columns = [plx_col]
        if arenou_criterion:
            columns += [bp_rp_col, bp_rp_excess_factor_col]
        df = df[columns]

        # discard nans and infs
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any("columns")
        df = df[indices_to_keep]

        # apply arenou criterion
        if arenou_criterion:
            df = df.loc[
                (1 + 0.015 * df[bp_rp_col] ** 2 < df[bp_rp_excess_factor_col])
                & (
                    df[bp_rp_excess_factor_col]
                    < 1.3 + 0.06 * df[bp_rp_col] ** 2
                )
            ]

        # apply parallax limits filter
        if lower_lim is not None:
            df = df.loc[(df[plx_col] >= lower_lim)]
        else:
            lower_lim = df[plx_col].min()
        if upper_lim is not None:
            df = df.loc[(df[plx_col] <= upper_lim)]
        else:
            upper_lim = df[plx_col].max()

        plx = df[plx_col].to_numpy()

        his, bin_edges = np.histogram(
            plx, bins=30, density=True, range=(lower_lim, upper_lim)
        )
        bin_width = bin_edges[1] - bin_edges[0]
        bin_center = bin_edges - bin_width / 2
        # freq = his * bin_width
        his = his / np.sum(his)

        solution = opt.least_squares(
            fun=lambda params, x, y: lognegexp_normal(params, x) - y,
            x0=np.array(initial_params),
            method="lm",
            args=(bin_center[1:], his),
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
        )

        solution["histogram"] = his
        solution["bin_edges"] = bin_edges

        return solution

    @checkargs
    def fit_pm(
        self,
        initial_params_ra,
        initial_params_dec,
        pmra_col: str = "pmra",
        pmdec_col: str = "pmdec",
        plx_col: str = "parallax",
        bp_rp_excess_factor_col: str = "phot_bp_rp_excess_factor",
        g_mag: str = "phot_g_mean_mag",
        bp_rp_col: str = "bp_rp",
        pdfs: str = "two_normal",
        bins_pmra: (int, str) = "fd",
        bins_pmdec: (int, str) = "fd",
        ftol: float = 1.0e-12,
        gtol: float = 1.0e-12,
        xtol: float = 1.0e-12,
        plx_lower_lim: (float, int) = None,
        plx_upper_lim: (float, int) = None,
        arenou_criterion: bool = True,
        **kwargs,
    ):
        df = self.table.to_pandas()

        columns = [
            pmra_col,
            pmdec_col,
            plx_col,
            bp_rp_col,
            bp_rp_excess_factor_col,
        ]
        df = df[columns]

        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any("columns")
        df = df[indices_to_keep]

        # Filtro en magnitud

        df = df.loc[df[bp_rp_col] < 14]

        # Filtro fotométrico

        df = df.loc[
            (1 + 0.015 * df[bp_rp_col] ** 2 < df[bp_rp_excess_factor_col])
            & (df[bp_rp_excess_factor_col] < 1.3 + 0.06 * df[bp_rp_col] ** 2)
        ]

        # Filtro de paralaje

        if plx_lower_lim is not None:
            df = df.loc[(df[plx_col] >= plx_lower_lim)]
        else:
            plx_lower_lim = df[plx_col].min()
        if plx_upper_lim is not None:
            df = df.loc[(df[plx_col] <= plx_upper_lim)]
        else:
            plx_upper_lim = df[plx_col].max()

        pmra = df[pmra_col].to_numpy()
        pmdec = df[pmdec_col].to_numpy()

        # Ajuste en pmra

        his, bin_edges_ra = np.histogram(pmra, bins=bins_pmra, density=True)

        bin_width_ra = bin_edges_ra[1] - bin_edges_ra[0]
        bin_center_ra = bin_edges_ra - bin_width_ra / 2
        # freq_ra = his * bin_width_ra
        his = his / np.sum(his)
        solution_ra = opt.least_squares(
            fun=lambda params, x, y: two_normal(params, x) - y,
            x0=np.array(initial_params_ra),
            method="lm",
            args=(bin_center_ra[1:], his),
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
        )

        solution_ra["histogram"] = his
        solution_ra["bin_edges"] = bin_edges_ra

        # Ajuste en pmdec

        his, bin_edges_dec = np.histogram(pmdec, bins=bins_pmdec, density=True)

        bin_width_dec = bin_edges_dec[1] - bin_edges_dec[0]
        bin_center_dec = bin_edges_dec - bin_width_dec / 2
        # freq_dec = his * bin_width_dec
        his = his / np.sum(his)

        solution_dec = opt.least_squares(
            fun=lambda params, x, y: two_normal(params, x) - y,
            x0=np.array(initial_params_dec),
            method="lm",
            args=(bin_center_dec[1:], his),
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
        )

        solution_dec["histogram"] = his
        solution_dec["bin_edges"] = bin_center_dec

        return {"solution_pmra": solution_ra, "solution_pmdec": solution_dec}
