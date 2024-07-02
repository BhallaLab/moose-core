# On WSL or git bash

- If not already setup, open PowerShell and install it:
```
wsl --install
```

- The above will install Ubuntu. Now open the terminal on WSL/Ubuntu and install `g++` and `make`
```
sudo apt-get install g++
sudo apt-get install make
sudo apt-get install qtbase5-dev
sudo apt install firefox
```

- Install micromamba:

https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

- Update micromamba

```
micromamba self-update
```

- Create environment with various required packages

```
micromamba create -n moose pip h5py numpy vpython scipy matplotlib gsl hdf5 jupyter pyqt cmake  pybind11[global] jupyterlab-vpython
```

- Activate the environment

```
micromamba activate moose
```

- Build and install pymoose

```
pip install git+https://github.com/BhallaLab/moose-core.git
```

You need to do the above only once. After that, each time to use moose, open WSL Ubuntu terminal and do the following

```
micromamba activate moose
```

and you are ready to start running moose scripts.


NOTE: When running moose examples that use vpython for 3D visualization, you may run into a message like `"gio: http://localhost:39143: Operation not supported` and nothing much happens. To see the visualization, start `firefox` (or any other browser you installed on WSL) from another WSL terminal and enter the web address (http://localhost:39143 in this case) in its  address bar.
