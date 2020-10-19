import os

from setuptools import setup


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="opencluster",
    version="0.0.1",
    author="Simón Pedro González, Antonio Alejo",
    author_email="simon.pedro.g@gmail.com",
    description=("mempership probabilities for open star clusters"),
    license="GPL-3",
    keywords="star cluster membership probabilities",
    url="http://packages.python.org/opencluster",
    packages=["opencluster", "tests"],
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "Cython",
        "scikit-learn",
        "astropy",
        "astroquery",
    ],
)
