# Installing MOOSE

## Installing released version from PyPI using `pip`
MOOSE is available on PyPI. To install the latest release, run `pip install pymoose`.

## Installing from a binary wheel using `pip`
Binary wheels for MOOSE are available on its github page under `Releases`. You can download a wheel suitable for your platform and install it directly with `pip`. The last three components of the wheel filename indicate what platform it was built for: `*-{python-version}-{operating-system}_{architecture}.whl`.


For example, 

```
pymoose-4.1.0.dev0-cp312-cp312-manylinux_2_28_x86_64.whl
```

is built for CPython version 3.12 on Linux for 64 bit Intel CPU (x86_64). MOOSE also depends on GSL, and the specific version should match. This should be available in the release description. The above was with GSL 2.7. So you will need an environment with these. 

The easiest way to setup a custom environment is with Anaconda (or similar tools like mamba, miniforge, micromamba, etc.) See https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html to find how to install conda or its variants. The channel `conda-forge` provides binary `gsl` package for all major platforms.

To create an environment named `moose` with these using conda run (*Note: in the commands below replace `conda` with `mamba` or `micromamba` if you are using one of those*):

```
conda  create -n moose python=3.12 gsl numpy vpython matplotlib -c conda-forge
```

in the terminal, and then activate this environment with

```
conda activate moose
```

Now install the wheel file with

```
pip install pymoose-4.1.0.dev0-cp312-cp312-manylinux_2_28_x86_64.whl
```

assuming you have the file in the current directory.

## Installing from source-code in github repository

To build `MOOSE` from source, you need `meson`, `ninja`, `pkg-config`, and `python-setuptools`. We
recommend to use Python 3.9 or higher.  To build and install with `pip`, you also need `meson-python`.

For platform specific instructions, see:
- Linux: UbuntuBuild.md
- MacOSX: AppleM1Build.md
- Windows: WindowsBuild.md

Briefly,
1. Build environment: make sure that the following packages are installed on your system.

   - gsl-1.16 or higher.
   - python-numpy
   - pybind11 (if the setup fails to find pybind11, try running `pip install pybind11[global]`)
   - python-libsbml
   - pyneuroml
   - clang compiler 18 or newer
   - meson
   - ninja
   - meson-python
   - python-setuptools
   - pkg-config

2. Now use `pip` to download and install `pymoose` from the [github repository](https://github.com/BhallaLab/moose-core).

```
$ pip install git+https://github.com/BhallaLab/moose-core --user
```

## Post installation

You can check that moose is installed and initializes correctly by running:
```
$ python -c "import moose; ch = moose.HHChannel('ch'); moose.le()"
```
This should show 
```
Elements under /
    /Msgs
    /clock
    /classes
    /postmaster
    /ch	
```

Now you can import moose in a Python script or interpreter with the statement:

    >>> import moose

## Uninstall

To uninstall moose, run

    $ pip uninstall pymoose
    
If you are building moose from source, make sure to get out of the source directory, or you may encounter a message like this:

    Found existing installation: pymoose {version}
    Can't uninstall 'pymoose'. No files were found to uninstall.



