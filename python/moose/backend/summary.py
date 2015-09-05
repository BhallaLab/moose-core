"""summary.py: 

    Print summary of model in moose.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

from . import backend 
import pprint

def summary():
    """Print summary of elements in moose"""
    elems = backend.moose_elems
    elems.populateStoreHouse()
    txt = {}
    txt["#Compartments"] = len(elems.compartments)
    txt["#Channels"] = len(elems.channels)
    txt["#SynChans"] = len(elems.synchans)
    txt["#Pulsegens"] = len(elems.pulseGens)
    txt["#Msgs"] = len(elems.msgs)
    txt["#Tables"] = len(elems.tables)
    pprint.pprint(txt)
    return txt


