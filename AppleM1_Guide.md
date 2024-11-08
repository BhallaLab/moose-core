- Install homebrew: https://brew.sh/
- Set up required development environment
  - Install command line tools for XCode
  - Install build dependencies by running these commands in a terminal
  ```
          brew install gsl
          brew install hdf5
          brew install graphviz
          brew install cmake
          brew install doxygen
  ```
  
- Install anaconda/miniconda/micromamba/miniforge. For example, for micromamba, run
```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

  in command line
- Update micromamba: `micromamba self-update`
- Restart terminal and create an environment with necessary packages: 

```
micromamba create -n moose hdf5 graphviz pytables numpy matplotlib vpython lxml doxygen setuptools wheel pybind11[global]
```
- Activate the moose environment: `micromamba activate moose`
- Install libsbml: `pip install python-libsbml`
- Install moose from github: `pip install git+https://github.com/BhallaLab/moose-core.git`
