# Building MOOSE on MacOS with Apple M1 CPU
- Install homebrew: https://brew.sh/
- Set up required development environment
  - Install command line tools for XCode
  - Install build dependencies by running these commands in a terminal
  ```
          brew install gsl
  ```
  
- Install anaconda/miniconda/micromamba/miniforge. For example, for micromamba, run
```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

  in command line
- Update micromamba: `micromamba self-update`
- Restart terminal and create an environment with necessary packages: 

```
micromamba create -n moose numpy matplotlib vpython lxml meson ninja meson-python gsl setuptools pybind11[global] pkg-config -c conda-forge
```
- Activate the moose environment: `micromamba activate moose`
- Install moose from github: `pip install git+https://github.com/BhallaLab/moose-core.git`
