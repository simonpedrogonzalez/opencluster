from opencluster.retriever import simbad_search, cone_search, load_VOTable
from astropy.coordinates import SkyCoord
from opencluster.exceptions import CatalogNotFoundException
from astropy.table.table import Table
from astropy.utils.diff import report_diff_values
import os


class TestDataRetriever:
    def test_correct_simbad_search(self):
        # check exact values
        assert simbad_search("ic2395").to_string("hmsdms") == SkyCoord(
            ra=130.62916667, dec=-48.1, frame="icrs", unit="deg"
        ).to_string("hmsdms")

    def test_non_existent_catalog_simbad_search(self):
        try:
            assert simbad_search("non existent table")
        except Exception as e:
            assert isinstance(e, CatalogNotFoundException)

    def test_cone_search(self):
        table = cone_search(name="ic2395", radius=0.1)
        assert isinstance(table, Table)
        table = cone_search(ra=130.62916667, dec=-48.1, radius=0.1)
        assert isinstance(table, Table)

    def test_wrong_params_conse_search(self):
        try:
            cone_search(ra=130.62916667, radius=0.1)
        except ValueError:
            assert True
        try:
            cone_search(name="ic2395", ra=130.62916667, radius=0.1)
        except ValueError:
            assert True
        try:
            cone_search(dec=130.62916667, radius=0.1)
        except ValueError:
            assert True
        try:
            cone_search(name=3, radius=0.1)
        except ValueError:
            assert True
        try:
            cone_search(name="ic2395", radius="0.1")
        except ValueError:
            assert True

    def test_loadVOTable(self):
        name = "ic2395"
        file = "./tests/" + name + ".vot"
        original_table = cone_search(name=name, radius=0.1)
        cone_search(
            name=name,
            radius=0.1,
            output_file=file,
            dump_to_file=True
        )
        loaded_table = load_VOTable(file)
        identical = report_diff_values(original_table, loaded_table)
        assert identical
        if os.path.exists(file):
            os.remove(file)
