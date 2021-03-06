import astropy.units as u

from opencluster import region

votable = (
    region(ra=130.62916667, dec=-48.1, radius=u.Quantity("30", u.arcminute))
    .select(["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"])
    .where({"phot_g_mean_mag": "<15"})
    .top(55)
    .get()
)

region(name="ic2395", radius=u.Quantity("30", u.arcminute)).select(
    "*"
).from_table("public.hipparcos", ra_name="ra", dec_name="de").top(50).get(
    dump_to_file=True, output_file="test.vot"
)

from opencluster import load_file

octable = load_file("ic2395.vot")

octable.densities_plot()

result = octable.fit_plx(
    lower_lim=0,
    upper_lim=2,
    bins=30,
    initial_params=[4.0, 0.1, 0.1, 1.0, 0.07, 1.3, 0.1],
)
result.get("fit")
result.get("params")
