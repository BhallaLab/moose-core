# Building MOOSE on Ubuntu (possibly in WSL)
0. Install GNU build tools

```
sudo apt install build-essential
```

1. Install conda/mamba/micromamba (in all the commands below `conda` can be replaced by `mamba` or `micromamba` respectively)

2. Create an environment with required packages

```
conda create -n moose meson gsl hdf5 cmake numpy matplotlib vpython doxygen pybind11[global] pkg-config -c conda-forge
```

3. Activate the environment

```
conda activate moose
```

4. Clone `moose-core` source code using git
5. Build moose
```
cd moose-core

meson setup --wipe _build --prefix=`pwd`/_build_install -Duse_mpi=false -Dbuildtype=release
ninja -v -C _build 
meson install -C _build
```

This will create `moose` module inside `moose-core/_build_install` directory. To make moose importable from any terminal, add this directory to your `PYTHONPATH` environment variable. For standard installation you can simply run `pip install .` in the `moose-core` directory.

Meson provides many builtin options: https://mesonbuild.com/Builtin-options.html. Meson options are supplied in the command line to `meson setup` in the format `-Doption=value`.

  - **Buildtype**
	If you want a developement build with debug enabled, pass `-Dbuildtype=debug` in the `meson setup`.


	```
	meson setup --wipe _build --prefix=`pwd`/_build_install -Duse_mpi=false -Dbuildtype=debug -Ddebug=true
	```

	You can either use `buildtype` option alone or use the two options `debug` and `optimization` for finer grained control over the build. According to `meson` documentation `-Dbuildtype=debug` will create a debug build with optimization level 0 (i.e., no optimization, passing `-O0 -g` to GCC), `-Dbuildtype=debugoptimized`  will create a debug build with optimization level 2 (equivalent to `-Ddebug=true -Doptimization=2`), `-Dbuildtype=release` will create a release build with optimization level 3 (equivalent to `-Ddebug=false -Doptimization=3`), and `-Dbuildtype=minsize` will create a release build with space optimization (passing `-Os` to GCC).
	
  - **Optimization level**
	
	To set optimization level, pass `-Doptimization=level`, where level can be `plain`, `0`, `g`, `1`, `2`, `3`, `s`.

6. For a Python development build so that your edits to the Python source code are included, run:

```
python -m pip install --no-build-isolation --editable .
```

7. To build a wheel (for distribution), you need `build` and `meson-python` modules:

```
conda install meson-python
conda install build
```

In a terminal, `cd` to `moose-core` and run the following:

```
python -m build
```
