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


class TestFit:
    def test_plx_fit(self):

        octable = load_remote(
            name="ic2395",
            radius=u.Quantity("30", u.arcminute),
            filters={"phot_g_mean_mag": "<= 14", "parallax": ">0"},
        )

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
        error = np.sum(np.square(solution - result.get("params")))
        assert error < 1.0e-15
