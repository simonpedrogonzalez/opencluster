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

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse
from astropy.table.table import Table

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.utils.commons import coord_to_radec, radius_to_unit

from attr import attrib, attrs, validators

from matplotlib import pyplot as plt

import numpy as np

import scipy.optimize as opt


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
    columns = attrib(default="*")
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


def lognegexp_normal(par, x):
    return (
        (par[0] * np.log(x / par[1]))
        + (par[2] * np.exp(-x / par[3]))
        + (par[4] / np.sqrt(2 * np.pi * par[6]))
        * np.exp(-((x - par[5]) ** 2) / (2 * (par[6] ** 2)))
    )


def normalnegexp_normal():
    """alternative that uses piecewise defined function
    with normal|negative exponential for field, normal for cluster"""
    ...


# Esta función nos permite hacer un ajuste del campo
#  y el cúmulo en pm y en las posiciones
def two_normal(params, x):
    # return (((params[4]/(np.sqrt(2.np.pi*params[1])))
    # (np.exp(-((x-params[0])*2/(2(params[1])*2)))  ))
    #   +  (  (params[5]/(np.sqrt(2.*np.pi*params[3])))
    # (np.exp(-((x-params[2])*2/(2(params[3])**2))))  )  )
    ...


@attrs(frozen=True)
class OCTable:
    table = attrib(validator=validators.instance_of(Table))

    def __getattr__(self, a):
        return getattr(self.table, a)

    def densities_plot(self, bins=100):
        df = self.table.to_pandas()
        plot = df.hist(bins=bins, grid=True, legend=True)
        return plot

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

        # get the plx columns as a numpy array to start the procedure
        plx = df[plx_col].to_numpy()

        # ponerle nombre significativo a las variables,
        #  por ejemplo his por histograma,
        # bin_edges, por bordes de los bines
        # aca notar que la variable bins puede ser un
        # numero o bien un string que indica el
        # metodo (ver en el encabezado de la funcion)
        his, bin_edges = np.histogram(
            plx, bins=30, density=True, range=(lower_lim, upper_lim)
        )
        bin_width = bin_edges[1] - bin_edges[0]
        bin_center = bin_edges - bin_width / 2
        freq = his * bin_width
        his = his / np.sum(his)

        solution = opt.least_squares(
            # que signigica esta fun:
            # la funcion se pone así porque NO quiero incluir
            # el "- y" dentro de la funcion
            # porque después quiero poder usar lognegexp_normal para
            # sacar los "y" para graficar
            # la funcion. SI le pongo el - y adentro, solo me sirve
            # para calcular los residuos y
            # no la puedo reutilizar.
            # por eso pongo una lambda, que toma x, parametros e y,
            # y calcula el resultado de la funcion
            # para x con parametros=params, y luego le resta y
            fun=lambda params, x, y: lognegexp_normal(params, x) - y,
            x0=np.array(initial_params),
            method="lm",
            args=(bin_center[1:], his),
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
        )

        # armo un eje con puntos "x" para graficar
        x_fit = np.linspace(bin_edges[0] + 0.1, bin_edges[-1], 10000)
        # reutilizo la funcion para calcular los "y" del grafico
        y_fit = lognegexp_normal(solution.x, x_fit)

        # hago una figura con un sublplot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        # le agrego el histograma del paralaje con la misma
        # cantidad de bines que usé en el ajuste
        ax.bar(
            bin_edges[1:] - bin_width / 2, freq, 0.06, label="Plx distribution"
        )

        # le agrego el grafico de la funcion de ajuste
        ax.plot(x_fit, y_fit, lw=2, c="r", label="Parallax fit")

        fig.suptitle("Parallax fit")
        fig.legend()

        # devuelvo un diccionario que tiene los parametros de la solucion
        # y el grafico para ver el ajuste
        return {
            "params": solution.x,
            "fit": fig,
        }

    @checkargs
    def fit_pm(
        self,
        pmra_col: str = "pmra",  # tiene que recibir los nombres
        # de las columnas que tienen los datos
        pmdec_col: str = "pmdec",  # por defecto tienen
        # los nombres que vienen en las tablas de gaia
    ):
        # Esta funcion tiene que ajustar el movimiento propio
        # tiene que recibir como parámetro todos lo que necesites

        # HACER LOS AJUSTES (LEER EL METODO plx_fit y hacerlo lo más
        #  parecido posible en estructura y tener en cuenta
        # los comentarios que le puse)

        # LAS FUNCIONES DE AJUSTE QUE SEAN NECESARIAS DEFINIRLAS
        #  AFUERA DE LA CLASE OCTABLE (leer lognegexp_normal y
        #  fijarse donde esta)

        # DEVOLVER LO MISMO QUE EL METODO plx_fit:
        #  la solucion y 2 graficos:
        #  1 con el histograma y funcion de pmra y
        #  otro con el histograma y funcion de pmdec
        return None
