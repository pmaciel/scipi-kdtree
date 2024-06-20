from setuptools import setup, find_packages
import os

setup(
    name="scipy-kdtree",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
