import setuptools

setuptools.setup(
    name="uedge_dcp",
    packages=setuptools.find_packages(),
    install_requires=[
        "h5py",
        "mppl",
        "forthon",
        "uedge",
        "matplotlib",
        "numpy",
        "scipy",
        "pandas",
        "shapely",
        "netCDF4",
        "freeqdsk>=0.5",
    ],
    author="Dominic Power",
    author_email="power8@llnl.gov",
    url="https://github.com/plasdom/uedge_dcp",
    description="Some UEDGE tools",
    long_description=open("README.md").read(),
)
