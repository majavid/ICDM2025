from setuptools import setup, find_packages
import os

repository_root = os.path.abspath('../')
repository_name = os.path.basename(repository_root)

setup(
    name=repository_name,
    packages=find_packages(
        include=["src"],
        exclude=("tests")
    ),
)
