# Installing MOOSE

## Installing released version using `pip`
MOOSE is available on PyPI. To install the latest release, run `pip install pymoose`.

## Building from source

To build `MOOSE` from source, you need `meson`, `ninja`, `pkg-config`, and `python-setuptools`. We
recommend to use Python 3.9 or higher.  To build and install with `pip`, you also need `meson-python`.

Before running the following command to build and install, make sure thatthe following packages are installed.

- gsl-1.16 or higher.
- python-numpy
- pybind11 (if the setup fails to find pybind11, try running `pip install pybind11[global]`)
- python-libsbml
- pyneuroml

On Ubuntu-16.04 or higher, these dependencies can be installed with:

```
sudo apt-get install python-pip python-numpy libgsl-dev g++ pybind11 meson ninja
```

And 

```
pip install python-libsbml
pip install pyneuroml
```

Now use `pip` to download and install `pymoose` from the [github repository](https://github.com/BhallaLab/moose-core).

```
$ pip install git+https://github.com/BhallaLab/moose-core --user
```

## Development build with `meson` and `ninja`

`pip`  builds `pymoose` with default options, it runs `meson` behind the scene.
If you are developing moose, want to build it with different options, or need to test
and profile it, `meson` and `ninja` based flow is recommended.

Install the required dependencies and download the latest source code of moose
from github.

```
    $ git clone https://github.com/BhallaLab/moose-core --depth 50 
    $ cd moose-core
    $ meson setup --wipe _build --prefix=`pwd`/_build_install -Duse_mpi=false -Dbuildtype=release
    $ ninja -v -C _build 
	$ meson install -C _build
```

This will build moose, in `moose-core/_build`  directory and install it as Python package in the `moose-core/_build_install` directory.

To rebuild, delete the `_build` directory and the generated `_build_install/` directory and continue the steps above starting with `meson setup ...`.

To make in debug mode replace the option `-Dbuildtype=release` with `-Dbuildtype=debug`

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



