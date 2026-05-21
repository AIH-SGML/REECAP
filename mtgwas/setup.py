from setuptools import setup, find_packages

setup(
    name="mtgwas",
    version="0.1",
    zip_safe=False,
    packages=find_packages(),
    author="Francesco Paolo Casale",
    author_email="paolo.casale@helmholtz-muenchen.de",
    description=("Univariate and multivariate models for GWAS"),
    include_package_data=True,
)

