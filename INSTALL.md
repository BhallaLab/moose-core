# Building MOOSE 

To build `MOOSE` from source, you need `cmake` and `python-setuptools`. We
recommend to use Python 3.8 or higher. 

Before running the following command to build and install, make sure that
followings are installed.

- gsl-1.16 or higher.
- python-numpy
- pybind11 (if the setup fails to find pybind11, try running `pip install pybind11[global]`)

On Ubuntu-16.04 or higher, these dependencies can be installed with:

```
sudo apt-get install python-pip python-numpy cmake libgsl-dev g++ pybind11
```

Now use `pip` to download and install `pymoose` from the [github repository](https://github.com/BhallaLab/moose-core).

```
$ pip install git+https://github.com/BhallaLab/moose-core --user
```

## Using cmake (For developers)

`pip`  builds `pymoose` with default options, it runs `cmake` behind the scene.
If you are developing moose, build it with different options, or needs to test
and profile it, `cmake` based flow is recommended.

Install the required dependencies and download the latest source code of moose
from github.

    $ git clone https://github.com/BhallaLab/moose-core --depth 50 
    $ cd moose-core
    $ mkdir _build
    $ cd _build
    $ cmake ..
    $ make -j3  
    $ ctest -j3 --output-on-failure

This will build moose, `ctest` will run few tests to check if build process was
successful.

To rebuild, delete the `_build` directory and the generated `_temp__build/` directory and recreate the `_build` directory:

    $ cd ..; rm -rf _build; rm -rf _temp__build
    
and continue the steps following that described above.

To make in debug mode use:
    $ cmake -DCMAKE_BUILD_TYPE=Debug ..

To make in debug mode with optimization turned off, use:
	$ cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_FLAGS_DEBUG="-g -O0" \
      -DCMAKE_CXX_FLAGS_DEBUG="-g -O0" ..


To make with profiling:
    $ cmake -DGPROF=ON -DCMAKE_BUILD_TYPE=Debug ..

To make with NSDF support (requires installation of libhdf5-dev):
    $ cmake -DWITH_NSDF=ON ..

To install MOOSE into non-standard directory, pass additional argument
`-DCMAKE_INSTALL_PREFIX=path/to/install/dir` to during configuration. E.g.,

   $ mkdir _build && cd _build    # inside moose-core directory.
   $ cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
   $ make && make install

Will build and install pymoose to `~/.local`.

To use a non-default python installation, set
`PYTHON_EXECUTATBLE=/path/to/python` e.g.,

    $ cmake -DPYTHON_EXECUTABLE=/opt/bin/python3 ..

## Post installation

Now you can import moose in a Python script or interpreter with the statement:

    >>> import moose
    >>> moose.test()   # will take time. Not all tests will pass.

## Uninstall

To uninstall moose, run

    $ pip uninstall pymoose
    
If you are building moose from source, make sure to get out of the source directory, or you may encounter a message like this:

    Found existing installation: pymoose {version}
    Can't uninstall 'pymoose'. No files were found to uninstall.




# Notes

- SBML support is enabled by installing
[python-libsbml](http://sbml.org/Software/libSBML/docs/python-api/libsbml-installation.html).
Alternatively, it can be installed by using `python-pip`

    $ sudo pip install python-libsbml


