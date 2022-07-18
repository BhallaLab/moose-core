[![Python package](https://github.com/BhallaLab/moose-core/actions/workflows/pymoose.yml/badge.svg)](https://github.com/BhallaLab/moose-core/actions/workflows/pymoose.yml)

# Installation

This repository is sufficient for using MOOSE as a python module. We provide python package via `pip`.

    $ pip install pymoose --user 

To install `nightly` build:

    $ pip install pymoose --user --pre --upgrde
    
Have a look at examples, tutorials and demo here https://github.com/BhallaLab/moose-examples.

# Build 

To build `pymoose`, follow instructions given here at https://github.com/BhallaLab/moose-core/blob/master/INSTALL.md 


----------
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

This is the core computational engine of [MOOSE simulator](https://github.com/BhallaLab/moose). This repository contains
C++ codebase and python interface called `pymoose`. For more details about MOOSE simulator, visit https://moose.ncbs.res.in .

# ABOUT VERSION 4.0.0, Jalebi

Jalebi is an Indian sweet involving a golden twisting tube like a hyper-pretzel,
of crunchy batter soaked in sugar syrup lightly flavoured with spices and
sometimes lemon.

This release has the following major changes:

1. A major under-the-hood change to numerics for chemical calculations,
eliminating the use of 'zombie' objects for the solvers. This simplifies
and cleans up the code and object access, but doesn't alter runtimes.

2. Another major under-the-hood change to use pybind11 as a much cleaner
way to interface the parser with the C++ numerical code.

3. Addition of a thread-safe and faster parser based on ExprTK

4. Resurrected objects for handling simulation output saving using HDF5
format. There is an HDFWriter class, an NSDFWriter, and a new NSDFWriter2.
The latter two implement storage in NSDF, Neuronal Simulation Data Format,
Ray et al Neuroinformatics 2016. NSDF is built on HDF5 and builds up a
specification designed to ensure ready replicability as well as self-
description of model output.

5. Multiple enhancements to rdesigneur, including vastly improved 3-D
graphics output using VPython.

6. Various bugfixes

# LICENSE

MOOSE is released under GPLv3.


