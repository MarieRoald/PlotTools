from setuptools import setup
from setuptools import find_packages


setup(
    name="plottools",
    version="0.0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
)
