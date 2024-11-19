# Building MOOSE on Windows with MSVC

## Virtual environment
You may want to use one of the virtual environment systems like Anaconda, Miniforge with Mamba, or Micromamba (https://mamba.readthedocs.io/en/latest/), which will allow you to create isolated Python environments. There  are binary packages for most of the requirements in the conda channels (conda-forge). In contrast, pip will actually download the package source code and try to build it locally, which opens up a chain of dependencies on various other libraries.

If you want to keep things slim, `micromamba` may be ideal because it is a single statically linked C++ executable and does not install any base environment.

In this guide, `conda` command can be replaced by `mamba` or `micromamba` if you are using one of those systems. 

To create an environment, open Anaconda command prompt (below we assume Windows CMD shell, you may need to change some commands for PowerShell) and enter

```
conda create -n moose meson ninja meson-python gsl hdf5 cmake numpy matplotlib vpython doxygen pkg-config clang pybind11[global] -c conda-forge
```

*Note: Please make sure you are using the `conda-forge` channel (`-c conda-forge`) for installing `GSL` and not the anaconda `dafaults`. The latter causes linking error*

This will create an environment name `moose`. In some terminals (windows cmd?) you may get an error for `pybind11[global]`. Put it inside quotes to work around it.

Then activate this environment for your build :

```
conda activate moose
```

## Requirements

You need to use Windows cmd shell (not powershell) for the following:

* Install either MS Visual Studio 2019 or MS Visual Studio Build Tools 2019, including the Windows SDK.
  Add path to this folder in your PATH variable
* Install the LLVM compiler infrastructure (https://releases.llvm.org/download.html). You can either install it directly, adding its bin folder to the `PATH` environment variable, or install it with winget from the commandline: `winget install llvm`
  To add it to PATH in PowerShell, run: `$env:PATH="$env:PATH;C:\Program Files\LLVM\bin"`
* Install git for Windows
* [Skip] For MPI install MS-MPI (https://github.com/microsoft/Microsoft-MPI/releases/), the only free MPI for Windows
  - TODO: MPI-build on Windows is not supported yet
* [Skip] Install doxygen
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

meson setup --wipe _build --prefix=%CD%\\_build_install -Duse_mpi=false --buildtype=release
ninja -v -C _build 
meson install -C _build
```

This will create `moose` module inside `moose-core/_build_install` directory. To make moose importable from any terminal, add this directory to your `PYTHONPATH` environment variable. 

*Note: for PowerShell, pass the absolute-path of the `moose-core\_build_install` directory instead of  `%CD%\\_build_install` to the `--prefix` option*

Meson provides many builtin options: https://mesonbuild.com/Builtin-options.html. Meson options are supplied in the command line to `meson setup` in the format `-Doption=value`.

  - **Buildtype**
	If you want a developement build with debug enabled, pass `-Dbuildtype=debug` in the `meson setup`.


	```
	meson setup --wipe _build --prefix=%CD%\\_build_install -Duse_mpi=false -Dbuildtype=debug
	```

	You can either use `buildtype` option alone or use the two options `debug` and `optimization` for finer grained control over the build. According to `meson` documentation `-Dbuildtype=debug` will create a debug build with optimization level 0 (i.e., no optimization, passing `-O0 -g` to GCC), `-Dbuildtype=debugoptimized`  will create a debug build with optimization level 2 (equivalent to `-Ddebug=true -Doptimization=2`), `-Dbuildtype=release` will create a release build with optimization level 3 (equivalent to `-Ddebug=false -Doptimization=3`), and `-Dbuildtype=minsize` will create a release build with space optimization (passing `-Os` to GCC).
	
  - **Optimization level**
	
	To set optimization level, pass `-Doptimization=level`, where level can be `plain`, `0`, `g`, `1`, `2`, `3`, `s`.



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
(CPython) itself from sources in debug mode. Therefore, debug build of
moose will fail at the linking stage complaining that the linker could
not find `python3x_d.lib`.

The workaround, as pointed out by Ali Ramezani
[here](https://stackoverflow.com/questions/66162568/lnk1104cannot-open-file-python39-d-lib),
is to make a copy of `python3x.lib` named `python3x_d.lib` in the same
directory (`libs`). After that, you can run meson setup as follows:

```
meson setup --wipe _build --prefix=%CD%\\_build_install -Duse_mpi=false --buildtype=debug
```

and then go through the rest of the steps.

A free graphical debugger available for MS Windows is WinDbg (https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/). You can
use it to attach to a running Python process and set breakpoints at
target function/line etc.

In WinDbg command line you find the moose module name with
`lm m _moose*`

The will show something like `_moose_cp311_win_amd64` when your build produced `_moose.cp311-win_amd64.lib`.

Now you can set a breakpoint to a class function with the module name as prefix as follows:

`bp _moose_cp311_win_amd64!ChanBase::setGbar`
