[![Python package](https://github.com/BhallaLab/moose-core/actions/workflows/pymoose.yml/badge.svg)](https://github.com/BhallaLab/moose-core/actions/workflows/pymoose.yml)

# MOOSE

MOOSE is the Multiscale Object-Oriented Simulation Environment. It is designed
to simulate neural systems ranging from subcellular components and biochemical
reactions to complex models of single neurons, circuits, and large networks. 
MOOSE can operate at many levels of detail, from stochastic chemical 
computations, to multicompartment single-neuron models, to spiking neuron
network models.

MOOSE is multiscale: It can do all these calculations together. For example
it handles interactions seamlessly between electrical and chemical signaling.
MOOSE is object-oriented. Biological concepts are mapped into classes, and
a model is built by creating instances of these classes and connecting them
by messages. MOOSE also has classes whose job is to take over difficult
computations in a certain domain, and do them fast. There are such solver
classes for stochastic and deterministic chemistry, for diffusion, and for 
multicompartment neuronal models.

MOOSE is a simulation environment, not just a numerical engine: It provides
data representations and solvers (of course!), but also a scripting interface
with Python, graphical displays with Matplotlib, PyQt, and VPython, and 
support for many model formats. These include SBML, NeuroML, GENESIS kkit 
and cell.p formats, HDF5 and NSDF for data writing.

This is the core computational engine of [MOOSE
simulator](https://github.com/BhallaLab/moose). This repository
contains C++ codebase and python interface called `pymoose`. For more
details about MOOSE simulator, visit https://moose.ncbs.res.in .


----------
# Installation

See [INSTALLATION.md](INSTALLATION.md) for instructions on installation.

Have a look at examples, tutorials and demo here
https://github.com/BhallaLab/moose-examples.

# Build 

To build `pymoose`, follow instructions given in
[INSTALLATION.md](INSTALLATION.md) and for platform specific
information see:
- Linux: [UbuntuBuild.md](UbuntuBuild.md)
- MacOSX: [AppleM1Build.md](AppleM1Build.md)
- Windows: [WindowsBuild.md](WindowsBuild.md)

# ABOUT VERSION 4.1.0, `Jhangri`

[`Jhangri`](https://en.wikipedia.org/wiki/Imarti) is an Indian sweet
in the shape of a flower. It is made of white-lentil (*Vigna mungo*)
batter, deep-fried in ornamental shape to form the crunchy, golden
body, which is then soaked in sugar syrup lightly flavoured with
spices.

This release has the following major changes:

1. Improved support for reading NeuroML2 models
2. `HHGate2D`: separate `xminA`, `xminB`, etc. for `A` and `B` tables
   replaced by single `xmin`, `xmax`, `xdivs`, `ymin`, `ymax`, and
   `ydivs` fields for both tables.
2. Build system switched from cmake to meson
2. Native binaries for Windows
6. Updated to conform to c/c++-17 standard
7. Various bugfixes

# LICENSE

MOOSE is released under GPLv3.


