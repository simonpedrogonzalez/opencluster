import numpy as np
import astropy.units as u
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from opencluster.decorators import unsilence_warnings
from astropy.io.votable import parse


@unsilence_warnings()
def simbad_search(id):
    result = Simbad.query_object(id)
    ra = (
        np.array(result["RA"])[0]
        .replace(" ", "h", 1)
        .replace(" ", "m", 1) + "s"
    )
    dec = (
        np.array(result["DEC"])[0]
        .replace(" ", "d", 1)
        .replace(" ", "m", 1) + "s"
    )
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
    **kwargs
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
        if not isinstance(ra, (float, int)) or\
             not isinstance(dec, (float, int)):
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
        **kwargs
    )

    if not dump_to_file:
        table = job.get_results()
        return table


def load_VOTable(path):
    table = (
        parse(path, pedantic=False).
        get_first_table().
        to_table(use_names_over_ids=True)
    )
    return table
