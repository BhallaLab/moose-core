# Installing MOOSE

## Installing released version from PyPI using `pip`
MOOSE is available on PyPI. To install the latest release, run `pip install pymoose`.

## Installing from a binary wheel using `pip`
Binary wheels for MOOSE are available on its github page under `Releases`. You can download a wheel suitable for your platform and install it directly with `pip`. The last three components of the wheel filename indicate what platform it was built for: `*-{python-version}-{operating-system}_{architecture}.whl`.


For example, 

```
pymoose-4.1.0.dev0-cp312-cp312-linux_x86_64.whl
```

is built for CPython version 3.12 on Linux for 64 bit Intel CPU (x86_64). MOOSE also depends on GSL, and the specific version should match. This should be available in the release description. The above was with GSL 2.7. So you will need an environment with these. 

The easiest way to setup a custom environment is with Anaconda (or similar tools like mamba, miniforge, etc.) See https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html to find how to install conda or its variants.

To create an environment named `moose` with these using conda run:

```
conda  create -n moose pyton=3.12 gsl=2.7 -c conda-forge
```

in the terminal, and then activate this environment with

```
conda activate moose
```

Now install the wheel file with

```
pip install pymoose-4.1.0.dev0-cp312-cp312-linux_x86_64.whl
```

assuming you have the file in the current directory.

## Installing from github repository

To build `MOOSE` from source, you need `meson`, `ninja`, `pkg-config`, and `python-setuptools`. We
recommend to use Python 3.9 or higher.  To build and install with `pip`, you also need `meson-python`.

For platform specific instructions, see:
- Linux: UbuntuBuild.md
- MacOSX: AppleM1Build.md
- Windows: WindowsBuild.md

Briefly,
1. Before running the following command to build and install, make sure thatthe following packages are installed.

   - gsl-1.16 or higher.
   - python-numpy
   - pybind11 (if the setup fails to find pybind11, try running `pip install pybind11[global]`)
   - python-libsbml
   - pyneuroml

2. Now use `pip` to download and install `pymoose` from the [github repository](https://github.com/BhallaLab/moose-core).

```
$ pip install git+https://github.com/BhallaLab/moose-core --user
```

## Post installation

You can check that moose is installed and initializes correctly by running:
```
$ python -c "import moose; moose.le()"
```
This should show 
```
Elements under /
    /Msgs
    /clock
    /classes
    /postmaster
	
```

Now you can import moose in a Python script or interpreter with the statement:

    >>> import moose

## Uninstall

To uninstall moose, run

    $ pip uninstall pymoose
    
If you are building moose from source, make sure to get out of the source directory, or you may encounter a message like this:

    Found existing installation: pymoose {version}
    Can't uninstall 'pymoose'. No files were found to uninstall.



