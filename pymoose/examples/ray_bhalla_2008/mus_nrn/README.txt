This directory contains files to run a simulation combining a NEURON 
model with a GENESIS model simulated on MOOSE using Python as the
glue language.

The GENESIS model is taken from the doqcs database:
http://doqcs.ncbs.res.in/template.php?&y=accessiondetails&an=77

which corresponds to the paper:
S. M. Ajay and U. S. Bhalla, 
A role for ERKII in synaptic pattern selectivity on the time-scale of 
minutes. Eur J Neurosci 20, 2671-2680 (2004).

The NEURON model is taken from modeldb:
http://senselab.med.yale.edu/modeldb/ShowModel.asp?model=3263

Hippocampal CA3 pyramidal neuron model from the paper 
M. Migliore, E. Cook, D.B. Jaffe, D.A. Turner and D. Johnston, Computer
simulations of morphologically reconstructed CA3 hippocampal neurons, J.
Neurophysiol. 73, 1157-1168 (1995). 

Running:
You need PyMOOSE compiled with GSL support in order to run this demo. 

1. Compile the mod files for the NEURON model.
Under unix systems:
to compile the mod files use the command 
nrnivmodl 

2. Use the command:
python moosenrn.py
to run the simulation and generate the plots.


