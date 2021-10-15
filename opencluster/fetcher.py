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
from astropy.io.votable import from_table, parse, writeto
from astropy.table.table import Table

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.utils.commons import coord_to_radec, radius_to_unit

from attr import attrib, attrs, validators

import numpy as np

def default_table():
    return 'gaiaedr3.gaia_source'

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
def simbad_search(identifier: str, fields=['coordinates',
    'parallax','propermotions','velocity', 'dimensions', 'diameter'], dump_to_file: bool =False,
    output_file: str=None, **kwargs):
    """Search an identifier in Simbad catalogues.

    Parameters
    ----------
    identifier : str
    fields : list of strings optional, default: ['coordinates',
    'parallax','propermotions','velocity']
        Fields to be included in the result.
    dump_to_file: bool optional, default False
    output_file: string, optional.
        Name of the file, default is the object identifier.

    Returns
    -------
    coordinates : astropy.coordinates.SkyCoord
        Coordinates of object if found, None otherwise
    result: votable
        Full result table

    Warns
    ------
    Identifier not found
        If the identifier has not been found in Simbad Catalogues.
    """
    simbad = Simbad()
    for f in fields:
        simbad.add_votable_fields(f)
    result = simbad.query_object(identifier, **kwargs)

    if result is None:
        warnings.warn("Identifier not found.")
        return None

    coord = ' '.join(
        np.array(simbad.query_object(identifier)[['RA','DEC']])[0]
        )
    
    if dump_to_file:
        if not output_file:
            output_file = identifier
        table = from_table(result)
        writeto(table=table, file=output_file)
    
    return SkyCoord(coord, unit=(u.hourangle, u.deg)), result


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
        If str, must be '*', indicating all columns. If list of str,
        must be list of valid column names.
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

    COUNT_QUERY_TEMPLATE = """
    SELECT COUNT(*)
    FROM
    {table_name}
    WHERE
    1 = CONTAINS(
        POINT('ICRS', {ra_column}, {dec_column}),
        CIRCLE('ICRS', {ra}, {dec}, {radius}))
    """

    table = attrib(default=default_table())
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
        if column_filters:
            if not isinstance(column_filters, dict):
                raise ValueError(
                    "column_filters must be dict: {'column': '> value'}"
                )
            if self.columns != "*":
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
    def from_table(
        self, table: str, ra_name: str = None, dec_name: str = None
    ):
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
        if ra_name:
            self.ra_name = ra_name
        if dec_name:
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
            ra_hours, dec = coord_to_radec(self.coords)
            ra = ra_hours * 15.0

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

        Parameters
        ----------
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

        print('launching query')
        print(query)
        print('this may take some time...')

        job = Gaia.launch_job_async(query=query, **kwargs)
        if not kwargs.get("dump_to_file"):
            table = job.get_results()
            return table

    def count(self, **kwargs):
        if self.radius is not None and self.coords is not None:
            ra_hours, dec = coord_to_radec(self.coords)
            ra = ra_hours * 15.0

        query = self.COUNT_QUERY_TEMPLATE.format(
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
        print('launching query')
        print(query)
        print('this may take some time...')
        job = Gaia.launch_job_async(query=query, **kwargs)

        if not kwargs.get("dump_to_file"):
            table = job.get_results()
            return table


def query_region(*, ra=None, dec=None, name=None, coord=None, radius):
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
    coord : astropy.coordinates.SkyCoord, optional
        Coords of center of cone search

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
    if not ((name is not None) ^ (ra is not None and dec is not None) ^ (coord is not None)):
        raise ValueError("'name' or 'ra' and 'dec' are required (not both)")
    if name is not None:
        if not isinstance(name, str):
            raise ValueError("name must be string")
        else:
            coord, _ = simbad_search(name)
    if (ra, dec) != (None, None):
        if not isinstance(ra, (float, int)) or not isinstance(
            dec, (float, int)
        ):
            raise ValueError("ra and dec must be numeric")
        else:
            coord = SkyCoord(ra, dec, unit=(u.degree, u.degree), frame="icrs")
    radius_deg = radius_to_unit(radius, unit="deg")
    query = Query(radius=radius_deg, coords=coord)
    return query


def list_remotes(only_names: bool = True, pattern: str = None, **kwargs):
    """List available tables in Gaia catalogues.

    Parameters
    ----------
    only_names : bool, optional, default is True
        Return only table names.
    pattern: str, optional, return only results
        that match pattern. Works only if only_names
        is True

    Returns
    -------
    tables : list of str if only_names=True
        Available tables names

    tables: vot table if only_names=False
    """
    available_tables = Gaia.load_tables(**kwargs)
    if only_names:
        names = []
        for table in available_tables:
            name = table.get_qualified_name()
            index = name.index(".") + 1
            name = name[index:]
            if not pattern or pattern in name:
                names.append(name)
        return names
    return available_tables


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
    coord=None,
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
        query_region(name=name, ra=ra, dec=dec, coord=coord, radius=radius)
        .from_table(table)
        .select(columns)
        .where(filters)
        .top(limit)
    )
    result = query.get(**kwargs)

    if not kwargs.get("dump_to_file"):
        return OCTable(result)


@attrs(frozen=True)
class OCTable:
    """Class that contains data offers fit methods.

    Attributes
    ----------
    table : astropy.table.table
    """

    table = attrib(validator=validators.instance_of(Table))

    def __getattr__(self, attr):
        """Get a named attribute from an table; getattr(x, 'y') is equivalent to x.y.

        All astropy.table.table attributs are can be accessed.
        """
        return getattr(self.table, attr)

    def __getitem__(self, attr):
        """Get item from contained astropy.table.table."""
        return self.table[attr]

    def __iter__(self):
        """Get iterable from contained astropy.table.table."""
        return iter(self.table)

    def __len__(self):
        """Get table length."""
        return len(self.table)

    def __dir__(self):
        """Get list of attribute names.

        Returns
        -------
        list of strings
            List of attribute names
        """
        return dir(self.table)

    def write_to(self, filepath):
        return writeto(self.table, filepath)

