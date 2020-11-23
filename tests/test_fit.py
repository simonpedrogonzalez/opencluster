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

import astropy.units as u

import numpy as np

from opencluster.opencluster import load_remote

import pytest


@pytest.fixture(scope="session", autouse=True)
def octable():
    return load_remote(
        name="ic2395",
        radius=u.Quantity("30", u.arcminute),
        filters={"phot_g_mean_mag": "<= 14", "parallax": ">0"},
    )
    # return load_file("opencluster/ic2395.vot")


class TestFit:
    def test_plx_fit(self, octable):

        result = octable.fit_plx(
            lower_lim=0,
            upper_lim=2,
            bins=30,
            initial_params=[4.0, 0.1, 0.1, 1.0, 0.07, 1.3, 0.1],
        )

        solution = np.array(
            [
                0.06431495,
                3.22492762,
                0.30726659,
                0.91553511,
                0.02719005,
                1.3648159,
                0.04344376,
            ]
        )

        np.testing.assert_array_almost_equal(result.x, solution)

    def test_pm_fit(self, octable):
        result = octable.fit_pm(
            initial_params_ra=[-4.0, 0.6, 0.2, 3.6, 6.0, 0.03],
            initial_params_dec=[4.0, 0.8, 0.5, 8.4, 5.0, 0.07],
            bins=60,
            plx_lower_lim=0,
            plx_upper_lim=2,
        )
        solution_pmra = np.array(
            [
                -4.31577558,
                0.64112027,
                -4.69865038,
                2.48052096,
                0.14776007,
                0.27382957,
            ]
        )
        solution_pmdec = np.array(
            [
                3.80299902,
                0.85536873,
                5.25105991,
                2.80916489,
                0.23382906,
                0.1937844,
            ]
        )

        np.testing.assert_array_almost_equal(
            result["solution_pmra"].x, solution_pmra
        )
        np.testing.assert_array_almost_equal(
            result["solution_pmdec"].x, solution_pmdec
        )
