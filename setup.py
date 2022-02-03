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
        "numpy==1.19.3",
        "matplotlib==3.4.1",
        "scipy==1.4.1",
        # "Cython",
        "astropy",
        "astroquery",
        "pandas",
        "hdbscan==0.8.27",
        "scikit-learn==0.23.1",
        "scikit-image==0.18.1",
        "rpy2==3.1.0",
        "seaborn==0.11.0",
        "attr==0.3.1",
        "statmodels==0.12.2",
        "KDEpy==1.0.10",
        "unidip==0.1.1"
    ],
)
