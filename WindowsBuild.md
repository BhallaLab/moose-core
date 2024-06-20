# Building MOOSE on Windows with MSVC

You may want to use one of the virtual environment systems like Anaconda, Miniforge with Mamba, or Micromamba, which will allow you to create isolated Python environments. There  are binary packages for most of the requirements in the conda channels (conda-forge). In contrast, pip will actually download the package source code and try to build it locally, which opens up a chain of dependencies on various other libraries.

To create an environment, open Anaconda command prompt (below we assume Windows CMD shell, you may need to change some commands for PowerShell) and enter
```
conda create -n moose
```

Then switch to this environment for your build 
```
conda activate moose
```

## Requirements
* Install either MS Visual Studio 2015 or newer or MS Visual Studio Build Tools.
  Add path to this folder in your PATH variable
* Install git fow Windows
* Install vcpkg (https://github.com/microsoft/vcpkg)

```
      git clone https://github.com/microsoft/vcpkg
      .\vcpkg\bootstrap-vcpkg.bat
      .\vcpkg\vcpkg integrate install
```

* Install cmake
  Using conda (mamba)
```
conda install cmake
```

* Install GSL using vcpkg (enter the following in the command line):

```
      .\vcpkg\vcpkg install gsl:x64-windows
```
	  
* Install HDF5 (for NSDF support)

```
.\vcpkg\vcpkg.exe install hdf5:x64-windows
```

* Install pybind11 (https://pybind11.readthedocs.io/en/stable/installing.html)

```
.\vcpkg\vcpkg.exe install pybind11
```

* Install doxygen

  ```
  .\vcpkg\vcpkg.exe install doxygen:x64-windows
  ```

* Install python package requirements (any of these that you don't have already installed). 
```
pip install numpy
pip install matplotlib
pip install vpython
```

* Get the environment variable for MS BuildTools set up by running 

```
C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\LaunchDevCmd.bat
```


* Clone `moose-core` source code using git
* Build moose
```
cd moose-core
pip install .
```

* Rename files

  * The build process will create everything in the folder `_temp__build`
  * The `python` subdirectory contains the moose python module.
  * The binary library for moose is created as `_temp__build\python\moose\Release\moose.pyd`
  * Move this file to `_temp__build\python\moose\_moose.pyd`
  * Add `_temp__build\python`  to your `PYTHONPATH` environment variable and you are ready to go.
