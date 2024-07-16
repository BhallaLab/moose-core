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

meson setup --wipe _build --prefix=%CD%\\_build_install -D use_mpi=false --buildtype=release -Ddebug=false
ninja -v -C _build 
meson install -C _build
```

This will create `moose` module inside `moose-core/_build_install` directory. To make moose importable from any terminal, add this directory to your `PYTHONPATH` environment variable. For standard installation you can simply run `pip install .` in the `moose-core` directory.

6. For a development build so that your edits to the Python source code are included, run:

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
