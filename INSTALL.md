# Pre-built packages

## Linux

Use our repositories hosted at [Open Build Service](http://build.opensuse.org).
We have packages for Debian, Ubuntu, CentOS, Fedora, OpenSUSE/SUSE, RHEL,
Scientific Linux.  Visit the following page for instructions.

https://software.opensuse.org/download.html?project=home:moose&package=moose

## MacOSX

MOOSE is available via [homebrew](http://brew.sh).

    $ brew install homebrew/science/moose


# Building MOOSE from source

To build `MOOSE` from source, you can either use `cmake` (recommended) or GNU `make` based flow.

Download the latest source code of moose from github.

    $ git clone -b master https://github.com/BhallaLab/moose-core

## Install dependencies

For moose-core:

- gsl-1.16 or higher.
- libhdf5-dev (optional)
- python-dev
- python-numpy

On Ubuntu-12.04 or higher, these can be installed with:

    sudo apt-get install python-dev python-numpy libhdf5-dev cmake libgsl0-dev g++

__NOTE__ : On Ubuntu 12.04, gsl version is 1.15. You should skip `libgsl0-dev` install gsl-1.16 or higher manually.

SBML support is enabled by installing [python-libsbml](http://sbml.org/Software/libSBML/docs/python-api/libsbml-installation.html). Alternatively, it can be installed by using `python-pip`

    $ sudo pip install python-libsbml

## Use `cmake` to build moose:

    $ cd /path/to/moose-core
    $ mkdir _build
    $ cd _build
    $ cmake ..
    $ make  
    $ ctest --output-on-failure

This will build moose and its python extentions, `ctest` will run few tests to
check if build process was successful.

To install MOOSE into non-standard directory, pass additional argument `-DCMAKE_INSTALL_PREFIX=path/to/install/dir` to cmake.

### Python3

You just need to one command in previous set of instructions to following

    cmake -DPYTHON_EXECUTABLE=/opt/bin/python3 ..

### Install

    $ sudo make install
