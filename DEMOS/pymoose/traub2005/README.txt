* This directory contains moose version of the cell models from the paper

  Traub, R.D. et al. Single-Column Thalamocortical Network Model
  Exhibiting Gamma Oscillations, Sleep Spindles, and Epileptogenic
  Bursts. J Neurophysiol 93, 2194-2232 (2005).

  The pymoose models were developed with help from both the original
  fortran code and the neuron version of the models.

  
  
- Code is organized into the following directories:

nrn: 

Contains test scripts for doing a current clamp experiment on each
cell. If you want to run these scripts you need to download the neuron
model from modelDB:

http://senselab.med.yale.edu/ModelDb/ShowModel.asp?model=82894

and unpack the contents. This will create a directory called: nrntraub.
Move the contents to nrn directory and then compile the mod files
using:
cd nrn
nrnivmodl mod

nrn/data:

Results of running test_{XYZ}.hoc (Vm_{XYZ}.plot and Ca_{XYZ}/plot) is
saved in this directory. For each cell the data files have already
been provided in gzipped format. So you don't need to actually run the
NEURON scripts to do the pymoose simulation.


py:

 - *.p : each .p file contains a prototype cell definition used by readcell
    to create a prototype model in /library.

 - config.py: basic configuration settings for the simulation

 - trbutil.py: utility for plotting and reading data files.

 - channel.py: channel base class

 - kchans.py : all K+ channel definitions

 - nachans.py: all Na+ channel definitions

 - compartment.py: an extension of compartment class with some
   utilities like insertion of recording tables by field name, setting
   conductance values by specifying the densities

 - cachans.py: Ca2+ channels

 - capool.py: extended version of CaConc

 - archan.py: the combined cation current

 - cell.py: base class for all the cells.

 - deepbasket.py: deep basket cells

 - deepLTS: deep LTS interneurons

 - deepaxoaxonic.py: deep axoaxonic cell

 - nontuftedRS.py: nontufted regular spiking cell.

 - nRT.py: cell from thalamic reticular nucleus

 - simulation.py: utility for controlling the simulation

 - spinystellate.py: spiny stellate cells from layer 4

 - supaxoaxonic.py: superficial axoaxonic cell

 - supbasket.py: superficial basket cell

 - supLTS: superficial LTS cell

 - suppyrFRB.py: superficial FRB puramidal cell

 - suppyrRS.py: superficial regular spiking pyramidal cell

 - tcr.py: thalamocortical relay cell

 - tuftedIB.py: tufted intrinsically bursting neuron

 - tuftedRS.py: tufted regular spiking cell

py/data:
	Contains the simulation results.

----------------------------------------------
Subhasis Ray / NCBS, Bangalore / 2009-10-17
