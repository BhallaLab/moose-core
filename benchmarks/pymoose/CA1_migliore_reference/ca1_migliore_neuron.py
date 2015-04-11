"""ca1_migliore_neuron.py: 

    Run this model in NEURON.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"


from neuron import h
h.load_file('celbild.hoc')
cb = h.CellBuild(0)
cb.manage.neuroml('./Generated.net.xml')
cb.cexport(1)
