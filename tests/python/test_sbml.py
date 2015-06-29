"""
Test SBML capabilities of PyMOOSE
"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
sys.path.append('../../python')
import moose
import moose.utils as mu
import sys
import pylab

def main():
    modelname = './chem_models/mkp1_feedback_effects_acc4.xml'
    modelType = modelname.split(".")[-1]
    if modelType == "xml":
        model = moose.readSBML(modelname, '/model')
    else:
        raise IOError("Input is not XML model. Wrong extension %s" % modelType)
    tables = moose.wildcardFind('/##[TYPE=Table2]')
    records = {}
    for t in tables: records[t.path.split('/')[-1]] = t
    c = moose.Clock('/clock')
    moose.reinit()
    moose.start(200)
    check(tables)

def check(tables):
    assert len(tables) > 0, "No moose.Table2 created."
    for t in tables:
        assert len(t.vec) > 100
    # TODO: Add more tests here.


if __name__ == '__main__':
    main()
