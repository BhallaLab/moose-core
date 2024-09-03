# Building MOOSE on Windows with MSVC

## Virtual environment
You may want to use one of the virtual environment systems like Anaconda, Miniforge with Mamba, or Micromamba (https://mamba.readthedocs.io/en/latest/), which will allow you to create isolated Python environments. There  are binary packages for most of the requirements in the conda channels (conda-forge). In contrast, pip will actually download the package source code and try to build it locally, which opens up a chain of dependencies on various other libraries.

If you want to keep things slim, `micromamba` may be ideal because it is a single statically linked C++ executable and does not install any base environment.

In this guide, `conda` command can be replaced by `mamba` or `micromamba` if you are using one of those systems. 

To create an environment, open Anaconda command prompt (below we assume Windows CMD shell, you may need to change some commands for PowerShell) and enter

```
conda create -n moose meson gsl hdf5 cmake numpy matplotlib vpython doxygen pkg-config pybind11[global] -c conda-forge
```

This will create an environment name `moose`. In some terminals (windows cmd?) you may get an error for `pybind11[global]`. Put it inside quotes to work around it.

Then activate this environment for your build :

```
conda activate moose
```

## Requirements

You need to use Windows cmd shell (not powershell) for the following:

* Install either MS Visual Studio 2015 or newer or MS Visual Studio Build Tools.
  Add path to this folder in your PATH variable
* Install git for Windows
* [Skip] For MPI install MS-MPI (https://github.com/microsoft/Microsoft-MPI/releases/), the only free MPI for Windows
  - TODO: MPI-build on Windows is not supported yet
* [Skip] Install doxygen
* Install `pkg-config`

```
conda install pkg-config
```

* Get the environment variable for MS Visual Studio command line tools set up by running 

```
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

Gotcha: if you are on a 64 bit machine, the machine type is `x64`. MSVC comes with cross compilation support for various machine-os combos (x86, x86_64). You can initialize the architecture according to your specific case (see this [stackoverflow comment](https://stackoverflow.com/questions/78446613/whats-the-difference-in-visual-studio-between-amd64-x86-vs-x86-amd64)

"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" {combo}

* Clone `moose-core` source code using git
* Build moose
```
cd moose-core

meson setup --wipe _build --prefix=%CD%\\_build_install -D use_mpi=false --buildtype=release -Ddebug=false
ninja -v -C _build 
meson install -C _build
```

This will create `moose` module inside `moose-core/_build_install` directory. To make moose importable from any terminal, add this directory to your `PYTHONPATH` environment variable. 


For standard installation you can simply run `pip install .` in the `moose-core` directory.

To build a wheel, you need `build` and `meson-python` modules:

```
conda install meson-python
conda install build
```

In a terminal, `cd` to `moose-core` and run the following:

```
python -m build
```

# Debug build
Debug build tries to link with debug build of Python, and this is not
readily available on Windows, unless you build the Python interpreter
(CPython) itself from sources in debug mode. Therefore, debug build of moose will fail at the linking stage complaining that the linker could not find `python3x_d.lib`.

The workaround, as pointed out by Ali Ramezani [here](https://stackoverflow.com/questions/66162568/lnk1104cannot-open-file-python39-d-lib), is to make a copy of `python3x.lib` named `python3x_d.lib` in the same directory (`libs`). After that, you can run meson setup as follows:

`meson setup --wipe _build --prefix=%CD%\\_build_install -D use_mpi=false --buildtype=debug --optimization=0`

and then go through the rest of the steps.

A free graphical debugger available for MS Windows is WinDbg. You can
use it to attach to a running Python process and set breakpoints at
target function/line etc.

In WinDbg command line you find the moose module name with
`lm m _moose*`

The will show something like `_moose_cp311_win_amd64` when your build produced `_moose.cp311-win_amd64.lib`.

Now you can set a breakpoint to a class function with the module name as prefix as follows:

`bp _moose_cp311_win_amd64!ChanBase::setGbar`
