.. A cookbook for MOOSE
.. Lists all the snippets in Demos/snippets directory

MOOSE Cookbook   
==============

What is MOOSE and what is it good for?
--------------------------------------
MOOSE is the Multiscale Object-Oriented Simulation Environment. It is designed to simulate neural systems ranging from subcellular components and biochemical reactions to complex models of single neurons, circuits, and large networks. MOOSE can operate at many levels of detail, from stochastic chemical computations, to multicompartment single-neuron models, to spiking neuron network models.

MOOSE is multiscale: It can do all these calculations together. One of its major uses is to make biologically detailed models that combine electrical and chemical signaling.

MOOSE is object-oriented. Biological concepts are mapped into classes, and a model is built by creating instances of these classes and connecting them by messages. MOOSE also has numerical classes whose job is to take over difficult computations in a certain domain, and do them fast. There are such solver classes for stochastic and deterministic chemistry, for diffusion, and for multicompartment neuronal models.

MOOSE is a simulation environment, not just a numerical engine: It provides data representations and solvers (of course!), but also a scripting interface with Python, graphical displays with Matplotlib, PyQt, and OpenGL, and support for many model formats. These include SBML, NeuroML, GENESIS kkit and cell.p formats, HDF5 and NSDF for data writing.

Loading and running models
--------------------------
This section of the documentation explains how to load and run predefined
models in MOOSE.

Hello, MOOSE: Load, run and display existing models.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: helloMoose

The Hodgkin-Huxley demo
^^^^^^^^^^^^^^^^^^^^^^^
This is a self-contained graphical demo implemented by Subhasis Ray,
closely based on the 'Squid' demo by Mark Nelson which ran in GENESIS.
The demo has built-in documentation and may be run from the 
``Demos/squid``
subdirectory of MOOSE.

Start, Stop, and Speed
^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: startstop

Accessing and tweaking parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: tweakingParameters

Storing simulation output
^^^^^^^^^^^^^^^^^^^^^^^^^
Here we'll show how to store and dump from a table and also using HDF5.

.. automodule:: hdfdemo

Chemical Signaling models
-------------------------
This section of the documentation explains how to do operations specific
to the chemical signaling.

Running with different numerical methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: switchKineticSolvers

Changing volumes
^^^^^^^^^^^^^^^^
.. automodule:: scriptGssaVols.py

Feeding tabulated input to a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finding steady states
^^^^^^^^^^^^^^^^^^^^^

Making a dose-response curve
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building a chemical model from parts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Oscillation models
^^^^^^^^^^^^^^^^^^

Bistability models
^^^^^^^^^^^^^^^^^^

Reaction-diffusion models
-------------------------

Reaction-diffusion in a cylinder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: cylinderDiffusion

Reaction-diffusion in a neuron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reaction-diffusion + motor + transport
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Turing model
^^^^^^^^^^^^^^

A spatial bistable model
^^^^^^^^^^^^^^^^^^^^^^^^

Single cell models
------------------

Loading, modifying, saving
^^^^^^^^^^^^^^^^^^^^^^^^^^
Explicit vs. implict methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Integrate-and-fire models
^^^^^^^^^^^^^^^^^^^^^^^^^
The HH model
^^^^^^^^^^^^
Analyzing spike trains
^^^^^^^^^^^^^^^^^^^^^^

Network models
--------------
Connecting two cells together
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Regular and plastic synapses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Providing random input to a cell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A recurrent integrate-and-fire network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A recurrent integrate-and-fire network with plastiicty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A feed-forward network with random input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using compartmental models in networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multiscale models
-----------------
A multiscale HH model with chemical feedback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Graphics
--------
Displaying time-series plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Animation of values along an axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using MOOGLI widgets to display a neuron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HDF5 Writer
-----------
.. automodule:: hdfdemo

		
